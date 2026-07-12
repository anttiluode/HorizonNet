"""
Horizon LM Autopsy
==================
Experiment 3 (char-LM) certified <1% of tokens. Two hypotheses:

  H_slow:  the weight-tied attention block genuinely does NOT reach a fixed
           point on language. The DEQ premise fails. Certificate is right to
           refuse. Nothing to fix.

  H_blind: the block DOES converge, but the ONLINE ESTIMATOR cannot see it,
           because updates are non-monotone (84% of sequences re-accelerate)
           and rho_hat = max(last 4 ratios) reads ~1 off a single bump.
           The certificate is broken, not the network.

These are distinguished by ONE measurement the previous runs never made:
the TRUE distance to the fixed point, ||z* - z_n||, with z* taken from a
deep reference (300 iterations). The online bound never sees this. We do.

Reported per token:
  true_err_n   = ||z_300 - z_n||_F                      (ground truth)
  bound_n      = rho_hat/(1-rho_hat) * d_n              (what the certificate believes)
  tightness    = bound_n / true_err_n                   (>>1 => estimator slack)
  oracle_halt  = last iteration at which the last-position argmax changed, +1
                 (the minimal depth a clairvoyant halting rule would use)
  cert_halt    = the soft certificate's halt

If oracle_halt << cert_halt, the compute was there to be saved and the
certificate simply could not prove it. That is H_blind, and it is fixable.
"""
import math, json, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(1)

text = open("shakespeare.txt").read()
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n_tr = int(0.95 * len(data)); train_data, val_data = data[:n_tr], data[n_tr:]
T = 32

def get_batch(split, n):
    src = train_data if split == "train" else val_data
    ix = torch.randint(0, len(src) - T - 1, (n,))
    return (torch.stack([src[i:i + T] for i in ix]),
            torch.stack([src[i + 1:i + T + 1] for i in ix]))

D, H, FFD = 64, 4, 128
CAUSAL = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)

class TiedLMBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Parameter(torch.randn(T, D) * 0.02)
        self.attn = nn.MultiheadAttention(D, H, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(D, FFD), nn.GELU(), nn.Linear(FFD, D))
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.inj = nn.Linear(D, D); self.head = nn.Linear(D, V); self.alpha = 0.5
    def u(self, x): return self.inj(self.emb(x) + self.pos)
    def step(self, z, u):
        a = self.attn(z, z, z, attn_mask=CAUSAL, need_weights=False)[0]
        h = self.ln1(z + a + u)
        return (1 - self.alpha) * z + self.alpha * self.ln2(h + self.ff(h))
    def forward(self, x, iters):
        u = self.u(x); z = torch.zeros(x.shape[0], T, D)
        for _ in range(iters): z = self.step(z, u)
        return self.head(z)

net = TiedLMBlock()
opt = torch.optim.Adam(net.parameters(), lr=2e-3)
t0 = time.time()
for step in range(2500):
    x, y = get_batch("train", 32)
    loss = F.cross_entropy(net(x, 10).reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0:
        print(f"step {step:4d} loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
torch.save(net.state_dict(), "horizon_lm.pt")
L_read = torch.linalg.matrix_norm(net.head.weight, 2).item()
print(f"L_readout = {L_read:.3f}")

# ---------------- deep trajectory, full record ----------------
DEEP, N, WIN = 300, 300, 4
x, y = get_batch("val", N)
with torch.no_grad():
    u = net.u(x); z = torch.zeros(N, T, D)
    Z, d_hist, am_hist, mg_hist = [], [], [], []
    for it in range(1, DEEP + 1):
        zn = net.step(z, u)
        d_hist.append((zn - z).flatten(1).norm(dim=1).numpy())
        lg = net.head(zn[:, -1, :])
        t2 = lg.topk(2, 1).values
        mg_hist.append((t2[:, 0] - t2[:, 1]).numpy())
        am_hist.append(lg.argmax(1).numpy())
        Z.append(zn.clone()); z = zn
    zstar = Z[-1]
    true_err = np.stack([(zstar - Zi).flatten(1).norm(dim=1).numpy() for Zi in Z], 1)  # N x DEEP

d = np.stack(d_hist, 1); am = np.stack(am_hist, 1); mg = np.stack(mg_hist, 1)

# does it converge at all?
conv = dict(
    true_err_it10  = float(np.median(true_err[:, 9])),
    true_err_it20  = float(np.median(true_err[:, 19])),
    true_err_it60  = float(np.median(true_err[:, 59])),
    true_err_it150 = float(np.median(true_err[:, 149])),
    true_rho_tail  = float(np.median((true_err[:, 199] / np.maximum(true_err[:, 149], 1e-12)) ** (1 / 50))),
)

# oracle halt: last iteration the argmax changed (relative to final), +1
final = am[:, -1]
oracle = np.zeros(N, dtype=int)
for i in range(N):
    diffs = np.where(am[i] != final[i])[0]
    oracle[i] = (diffs[-1] + 2) if len(diffs) else 1

# certificate (plain online estimator, cap 60) -- reproduce experiment 3
def certify(cap=60):
    halt = np.zeros(N, dtype=int); rho_at = np.full(N, np.nan); bnd_at = np.full(N, np.nan)
    for it in range(WIN + 1, cap + 1):
        ratios = d[:, it - WIN:it] / np.maximum(d[:, it - WIN - 1:it - 1], 1e-12)
        rho = ratios.max(1)
        b = np.where(rho < 0.999, rho / np.maximum(1 - rho, 1e-9) * d[:, it - 1], np.inf)
        fire = (halt == 0) & (2 * L_read * b < mg[:, it - 1])
        halt[fire] = it; rho_at[fire] = rho[fire]; bnd_at[fire] = b[fire]
    halt[halt == 0] = cap
    return halt, rho_at, bnd_at
cert_halt, rho_at, _ = certify()

# tightness of the bound at a fixed probe depth (it=20), where truth is known
it = 20
ratios = d[:, it - WIN:it] / np.maximum(d[:, it - WIN - 1:it - 1], 1e-12)
rho20 = ratios.max(1)
bound20 = np.where(rho20 < 0.999, rho20 / np.maximum(1 - rho20, 1e-9) * d[:, it - 1], np.inf)
tight = bound20 / np.maximum(true_err[:, it - 1], 1e-12)

res = dict(
    convergence=conv,
    oracle_halt_mean=float(oracle.mean()),
    oracle_halt_median=float(np.median(oracle)),
    oracle_halt_le_10=float((oracle <= 10).mean()),
    cert_halt_mean=float(cert_halt.mean()),
    certified_frac=float((~np.isnan(rho_at)).mean()),
    bound_tightness_at_20_median=float(np.median(tight[np.isfinite(tight)])),
    rho_hat_at_20_median=float(np.median(rho20)),
    rho_true_tail=conv["true_rho_tail"],
    L_readout=L_read,
    margin_median_final=float(np.median(mg[:, -1])),
    wasted_depth_mean=float((cert_halt - oracle).mean()),
)
print(json.dumps(res, indent=2))
json.dump(res, open("horizon_lm_autopsy.json", "w"), indent=2)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for i in range(30):
    ax[0].semilogy(true_err[i], color="#2a9d8f", lw=0.7, alpha=0.6)
    ax[0].semilogy(d[i], color="#e76f51", lw=0.7, alpha=0.4)
ax[0].set_title("TRUE error ‖z*−z_n‖ (teal) vs update ‖z_n−z_{n−1}‖ (orange)")
ax[0].set_xlabel("iteration"); ax[0].grid(alpha=0.3)
ax[1].hist(oracle, bins=range(0, 62), color="#2a9d8f", edgecolor="k", label="oracle halt")
ax[1].hist(cert_halt, bins=range(0, 62), color="#e76f51", alpha=0.6, edgecolor="k", label="certified halt")
ax[1].legend(); ax[1].set_xlabel("iteration"); ax[1].set_title("Depth actually needed vs depth the certificate demanded")
ax[1].grid(alpha=0.3)
fin = tight[np.isfinite(tight)]
ax[2].hist(np.log10(fin), bins=40, color="#264653", edgecolor="k")
ax[2].set_xlabel("log10( bound / true error )  at iteration 20")
ax[2].set_title(f"Bound slack (median {np.median(fin):.1f}×)"); ax[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("horizon_lm_autopsy.png", dpi=140)
print("saved")
