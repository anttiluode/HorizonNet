"""
Directional Horizon Certificate
===============================
The autopsy located the slack. It is not the contraction estimate (bound is
only 2.2x loose vs the true fixed-point error). It is the readout constant.

Standard certificate:
    |delta margin| <= ||w_a - w_b||_2 * ||e_n||        e_n = z* - z_n
                   <= 2*L * rho/(1-rho) * d_n           L = ||W||_2

That first inequality is Cauchy-Schwarz against a WORST-CASE direction. But
the error e_n is not worst-case. For a map iterated to a fixed point, the
error obeys e_n = sum_{k>=n} (z_{k+1} - z_k), and near the fixed point each
update is J applied to the last, so the updates -- and hence e_n -- align with
the DOMINANT SLOW EIGENVECTOR of J. This is just power iteration.

So the last update direction v_n = (z_n - z_{n-1})/||.|| is an online, free
estimate of the error direction. Replace the worst-case constant with the
projected one:

    |delta margin| <= |(w_a - w_b) . v_n| * rho/(1-rho) * d_n

Nothing else changes. Everything is measured at the LAST position (the only
one the LM readout uses), so d_n and e_n are D-dimensional there.

We verify the assumption instead of assuming it:
  * alignment(n) = cos(e_n, v_n) measured against a 300-iteration reference.
    If this is not near 1, the directional certificate is unjustified and we
    report that and stop.
  * ever-flip violations of the directional certificate, checked against the
    full 300-iteration trajectory (the strongest violation test available).

No retraining: loads the checkpoint from the autopsy run.
"""
import json
import numpy as np
import torch, torch.nn as nn

torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(1)

text = open("shakespeare.txt").read()
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
val_data = data[int(0.95 * len(data)):]
T, D, H, FFD = 32, 64, 4, 128
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

net = TiedLMBlock(); net.load_state_dict(torch.load("horizon_lm.pt")); net.eval()
W = net.head.weight.detach()                      # V x D
L_global = torch.linalg.matrix_norm(W, 2).item()

DEEP, N, WIN, CAP = 300, 400, 4, 60
ix = torch.randint(0, len(val_data) - T - 1, (N,))
x = torch.stack([val_data[i:i + T] for i in ix])
y_last = torch.stack([val_data[i + T] for i in ix]).numpy()

with torch.no_grad():
    u = net.u(x); z = torch.zeros(N, T, D)
    ZL, dL, AM, MG, TOP2 = [], [], [], [], []
    for it in range(DEEP):
        zn = net.step(z, u)
        zl, zl_prev = zn[:, -1, :], z[:, -1, :]              # last position only
        dL.append((zl - zl_prev).norm(dim=1).numpy())
        ZL.append(zl.clone())
        lg = net.head(zl)
        t2 = lg.topk(2, 1)
        TOP2.append(t2.indices.numpy()); MG.append((t2.values[:, 0] - t2.values[:, 1]).numpy())
        AM.append(lg.argmax(1).numpy())
        z = zn
    Zstar = ZL[-1]

d = np.stack(dL, 1); am = np.stack(AM, 1); mg = np.stack(MG, 1); top2 = np.stack(TOP2, 1)
Zs = torch.stack(ZL, 1)                                       # N x DEEP x D

# ---- verify the assumption: does the error align with the last update? ----
E = Zstar.unsqueeze(1) - Zs                                   # N x DEEP x D  (true error)
Vdir = Zs - torch.cat([torch.zeros(N, 1, D), Zs[:, :-1]], 1)  # N x DEEP x D  (last update)
cos = torch.nn.functional.cosine_similarity(E, Vdir, dim=2).numpy()
align = {f"it{k}": float(np.median(cos[:, k - 1])) for k in (5, 10, 20, 40, 60)}
print("alignment cos(e_n, v_n):", json.dumps(align, indent=2))

# ---- certificates ----
def run_cert(directional):
    halt = np.zeros(N, dtype=int); pred = np.full(N, -1); flip = np.zeros(N, bool)
    for it in range(WIN + 1, CAP + 1):
        i = it - 1
        ratios = d[:, i - WIN + 1:i + 1] / np.maximum(d[:, i - WIN:i], 1e-12)
        rho = ratios.max(1)
        err_bnd = np.where(rho < 0.999, rho / np.maximum(1 - rho, 1e-9) * d[:, i], np.inf)
        if directional:
            v = Vdir[:, i, :]
            v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
            wa = W[torch.from_numpy(top2[:, i, 0])]; wb = W[torch.from_numpy(top2[:, i, 1])]
            coef = ((wa - wb) * v).sum(1).abs().numpy()       # |(w_a - w_b) . v_n|
        else:
            coef = np.full(N, 2 * L_global)
        fire = (halt == 0) & (coef * err_bnd < mg[:, i])
        halt[fire] = it; pred[fire] = am[fire, i]
        done = (halt > 0) & (halt < it)
        flip[done] |= (am[done, i] != pred[done])
    cert = halt > 0
    # strongest violation test: any disagreement anywhere out to iteration 300
    for i in range(CAP, DEEP):
        flip[cert] |= (am[cert, i] != pred[cert])
    halt[~cert] = CAP; pred[~cert] = am[~cert, -1]
    return halt, pred, cert, flip

res = {}
for name, flag in [("worst_case_L", False), ("directional", True)]:
    halt, pred, cert, flip = run_cert(flag)
    res[name] = dict(
        certified_frac=float(cert.mean()),
        mean_halt=float(halt[cert].mean()) if cert.any() else None,
        median_halt=float(np.median(halt[cert])) if cert.any() else None,
        everflip_violations_to_300=float(flip[cert].mean()) if cert.any() else None,
        acc_certified=float((pred[cert] == y_last[cert]).mean()) if cert.any() else None,
    )
ref = am[:, -1]
res["shared"] = dict(n=N, L_global=L_global, deep_ref=DEEP,
                     acc_ref=float((ref == y_last).mean()),
                     median_coef_directional=None)

# how much smaller is the directional coefficient, really?
i = 19
v = Vdir[:, i, :]; v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
wa = W[torch.from_numpy(top2[:, i, 0])]; wb = W[torch.from_numpy(top2[:, i, 1])]
cd = ((wa - wb) * v).sum(1).abs().numpy()
res["shared"]["median_coef_directional"] = float(np.median(cd))
res["shared"]["coef_worst_case"] = 2 * L_global
res["shared"]["coefficient_shrink"] = float(2 * L_global / np.median(cd))
res["alignment"] = align
print(json.dumps(res, indent=2))
json.dump(res, open("horizon_directional.json", "w"), indent=2)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for k in (5, 10, 20, 40):
    ax[0].hist(cos[:, k - 1], bins=40, alpha=0.5, label=f"iter {k}")
ax[0].set_xlabel("cos(true error, last update)"); ax[0].legend()
ax[0].set_title("Does the error align with the update direction?"); ax[0].grid(alpha=0.3)
hw, _, cw, _ = run_cert(False); hd, _, cdd, _ = run_cert(True)
ax[1].hist(hd[cdd], bins=range(0, CAP + 2), color="#2a9d8f", edgecolor="k", label="directional")
ax[1].hist(hw[cw], bins=range(0, CAP + 2), color="#e76f51", edgecolor="k", alpha=0.7, label="worst-case L")
ax[1].legend(); ax[1].set_xlabel("certified halt iteration")
ax[1].set_title(f"directional: {100*res['directional']['certified_frac']:.0f}% certified\n"
                f"worst-case: {100*res['worst_case_L']['certified_frac']:.1f}%")
ax[1].grid(alpha=0.3)
ax[2].hist(cd, bins=40, color="#264653", edgecolor="k")
ax[2].axvline(2 * L_global, color="r", ls="--", label=f"2·‖W‖₂ = {2*L_global:.1f}")
ax[2].set_xlabel("|(w_a − w_b)·v_n|  (directional coefficient)")
ax[2].set_title("The coefficient the certificate actually needs"); ax[2].legend(); ax[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("horizon_directional.png", dpi=140)
print("saved")
