"""
Horizon LM Probe (Experiment 3)
===============================
The kill-or-confirm test named in the README's next steps: does the soft
Banach certificate fire on REAL TOKENS?

Setup: a weight-tied CAUSAL transformer block (DEQ-style, input injection,
damping 0.5, no Lipschitz constraint anywhere) trained as a character-level
language model on tiny Shakespeare. At inference the block is iterated to a
fixed point while an online estimator tracks d_n = ||z_n - z_{n-1}||_F,
fits rho_eff conservatively (max of last WIN ratios), and forms the soft
a posteriori bound

    ||z* - z_n||_F <= rho/(1-rho) * d_n            (valid if rho < 1)

pushed through the readout Lipschitz constant to bound how far the
last-position next-character logits can still move. Halt when

    2 * L_read * bound < (top-1 logit - top-2 logit).

New vs experiment 2 (all three address ledger items / next steps):
  1. REAL TEXT. Margins now come from language statistics ('q'->'u' is a
     landslide, a word boundary is near-tied), not a synthetic dial.
  2. EVER-FLIP CHECK (stronger violation test). Experiment 2 only compared
     the certified answer against the final reference. Here we keep
     iterating after every halt and flag a violation if the argmax differs
     at ANY later iteration up to the reference depth. This is exactly the
     non-monotone-convergence failure mode the ledger worried about.
  3. SAFETY-MARGINED ESTIMATOR run alongside the plain one
     (rho_safe = min(rho + MARGIN, 0.995)), as next-step 4 demands.

Honest scope: one seed, one scale (d=64, T=32, ~120k chars of training on
1 CPU core), last-position certification only. This decides whether the
certificate CAN fire on real tokens, not whether it is production-useful.
"""
import math, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(1)

# ---------------- data: real text ----------------
text = open("shakespeare.txt").read()
chars = sorted(set(text))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n_tr = int(0.95 * len(data))
train_data, val_data = data[:n_tr], data[n_tr:]
T = 32                                   # context length

def get_batch(split, n):
    src = train_data if split == "train" else val_data
    ix = torch.randint(0, len(src) - T - 1, (n,))
    x = torch.stack([src[i:i + T] for i in ix])
    y = torch.stack([src[i + 1:i + T + 1] for i in ix])
    return x, y

# ---------------- model: weight-tied causal block ----------------
D, H, FFD = 64, 4, 128
CAUSAL = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)

class TiedLMBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(V, D)
        self.pos  = nn.Parameter(torch.randn(T, D) * 0.02)
        self.attn = nn.MultiheadAttention(D, H, batch_first=True)
        self.ff   = nn.Sequential(nn.Linear(D, FFD), nn.GELU(), nn.Linear(FFD, D))
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.inj  = nn.Linear(D, D)
        self.head = nn.Linear(D, V)
        self.alpha = 0.5

    def u(self, x):
        return self.inj(self.emb(x) + self.pos)

    def step(self, z, u):
        a = self.attn(z, z, z, attn_mask=CAUSAL, need_weights=False)[0]
        h = self.ln1(z + a + u)
        f = self.ln2(h + self.ff(h))
        return (1 - self.alpha) * z + self.alpha * f

    def logits(self, z):                 # per-position next-char logits
        return self.head(z)

    def forward(self, x, iters):
        u = self.u(x); z = torch.zeros(x.shape[0], T, D)
        for _ in range(iters):
            z = self.step(z, u)
        return self.logits(z)

net = TiedLMBlock()

# ---------------- train (10 unrolled weight-tied iterations) ----------------
TRAIN_ITERS, STEPS, B = 10, 2500, 32
opt = torch.optim.Adam(net.parameters(), lr=2e-3)
t0 = time.time()
for step in range(STEPS):
    x, y = get_batch("train", B)
    logits = net(x, TRAIN_ITERS)
    loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 250 == 0 or step == STEPS - 1:
        with torch.no_grad():
            xv, yv = get_batch("val", 256)
            lv = net(xv, TRAIN_ITERS)
            vloss = F.cross_entropy(lv.reshape(-1, V), yv.reshape(-1)).item()
            vacc = (lv.argmax(-1) == yv).float().mean().item()
        print(f"step {step:4d}  train {loss.item():.3f}  val {vloss:.3f} "
              f"acc {vacc:.3f}  ({time.time()-t0:.0f}s)", flush=True)

# readout Lipschitz for the LAST position:
# delta logits_last = W @ delta z_last, ||delta z_last|| <= ||delta z||_F
with torch.no_grad():
    L_read = torch.linalg.matrix_norm(net.head.weight, 2).item()

# ---------------- inference probe ----------------
MAX_IT, N_TEST, WIN, SAFETY = 60, 400, 4, 0.05
@torch.no_grad()
def probe(x):
    n = x.shape[0]
    u = net.u(x)
    z = torch.zeros(n, T, D)
    d_hist = np.zeros((n, MAX_IT))
    halted   = np.zeros((n, 2), dtype=int)        # [plain, safe] halt iters
    pred     = np.full((n, 2), -1, dtype=int)
    rho_halt = np.full((n, 2), np.nan)
    flipped  = np.zeros((n, 2), dtype=bool)       # ever-flip after halt
    reaccel  = np.zeros(n, dtype=bool)            # ratio>1 after iter 8
    argmax_final = None
    for it in range(1, MAX_IT + 1):
        z_new = net.step(z, u)
        d = (z_new - z).flatten(1).norm(dim=1).numpy()
        d_hist[:, it - 1] = d
        lg = net.logits(z_new)[:, -1, :]           # last position only
        top2 = lg.topk(2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).numpy()
        am = lg.argmax(1).numpy()
        if it > 8:
            prev = d_hist[:, it - 2]
            reaccel |= (d > prev + 1e-12)
        if it >= WIN + 1:
            ratios = d_hist[:, it - WIN:it] / np.maximum(d_hist[:, it - WIN - 1:it - 1], 1e-12)
            rho = ratios.max(1)
            for k, rr in enumerate([rho, np.minimum(rho + SAFETY, 0.995)]):
                ok = (rr < 0.999) & (2 * L_read * (rr / (1 - rr)) * d < margin)
                fire = ok & (halted[:, k] == 0)
                halted[fire, k] = it
                pred[fire, k] = am[fire]
                rho_halt[fire, k] = rho[fire]
        # ever-flip: any later argmax differing from the certified one
        for k in range(2):
            done = (halted[:, k] > 0) & (halted[:, k] < it)
            flipped[done, k] |= (am[done] != pred[done, k])
        z = z_new
        argmax_final = am
    for k in range(2):                              # never-certified -> cap
        miss = halted[:, k] == 0
        halted[miss, k] = MAX_IT
        pred[miss, k] = argmax_final[miss]
    return halted, pred, rho_halt, flipped, reaccel, d_hist, argmax_final

x, y = get_batch("val", N_TEST)
halt, pred, rho_h, flip, reacc, d_hist, ref = probe(x)
y_last = y[:, -1].numpy()

# final margins + entropy of the reference distribution (difficulty dial)
with torch.no_grad():
    lg_ref = net(x, MAX_IT)[:, -1, :]
    p_ref = lg_ref.softmax(1)
    ent = (-(p_ref * p_ref.clamp_min(1e-12).log()).sum(1)).numpy()
    t2 = lg_ref.topk(2, 1).values
    margin_ref = (t2[:, 0] - t2[:, 1]).numpy()

res = {}
for k, name in enumerate(["plain", "safety"]):
    cert = ~np.isnan(rho_h[:, k])
    res[name] = dict(
        certified_frac=float(cert.mean()),
        mean_halt=float(halt[cert, k].mean()) if cert.any() else None,
        halt_le_train_depth=float((halt[cert, k] <= TRAIN_ITERS).mean()) if cert.any() else None,
        viol_final=float((pred[cert, k] != ref[cert]).mean()) if cert.any() else None,
        viol_everflip=float(flip[cert, k].mean()) if cert.any() else None,
        rho_eff_median=float(np.nanmedian(rho_h[:, k])),
    )
res["shared"] = dict(
    n=N_TEST, ref_iters=MAX_IT, L_readout=L_read, vocab=V,
    acc_ref=float((ref == y_last).mean()),
    acc_cert_plain=float((pred[:, 0] == y_last).mean()),
    reaccel_frac=float(reacc.mean()),
    spearman_halt_vs_entropy=float(
        np.corrcoef(np.argsort(np.argsort(halt[:, 0])),
                    np.argsort(np.argsort(ent)))[0, 1]),
)
print(json.dumps(res, indent=2))

# concrete linguistic anchors: earliest vs latest certified halts
itos = {i: c for c, i in stoi.items()}
def show(i):
    ctx = "".join(itos[int(c)] for c in x[i]).replace("\n", "\\n")
    return f'...{ctx[-20:]}" -> "{itos[int(ref[i])]}"  halt {halt[i,0]:2d}  margin {margin_ref[i]:.2f}'
order = np.argsort(halt[:, 0])
anchors = {"earliest": [show(i) for i in order[:5]],
           "latest":   [show(i) for i in order[-5:]]}
print(json.dumps(anchors, indent=2))
res["anchors"] = anchors
json.dump(res, open("horizon_lm_probe.json", "w"), indent=2)

# ---------------- plots ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

for i in range(40):
    ax[0].semilogy(np.arange(1, MAX_IT + 1), d_hist[i], lw=0.7, alpha=0.5)
ax[0].set_xlabel("iteration"); ax[0].set_ylabel(r"$\|z_n-z_{n-1}\|_F$")
ax[0].set_title(f"Update norms on real text (40 seqs)\n"
                f"median ρ_eff at halt = {res['plain']['rho_eff_median']:.3f}")

cert = ~np.isnan(rho_h[:, 0])
ax[1].hist(halt[cert, 0], bins=range(0, MAX_IT + 2), color="#3a7ca5", edgecolor="k")
ax[1].axvline(MAX_IT, color="r", ls="--", label=f"reference ({MAX_IT})")
ax[1].axvline(TRAIN_ITERS, color="g", ls=":", label=f"train depth ({TRAIN_ITERS})")
ax[1].set_title(f"Certified halts: {100*res['plain']['certified_frac']:.1f}% of tokens\n"
                f"mean {res['plain']['mean_halt']:.1f} it, "
                f"ever-flip {100*res['plain']['viol_everflip']:.2f}%")
ax[1].set_xlabel("halt iteration"); ax[1].legend()

# adaptivity vs language difficulty (entropy of reference distribution)
q = np.quantile(ent, np.linspace(0, 1, 7))
mids, depths = [], []
for a, b in zip(q[:-1], q[1:]):
    m = (ent >= a) & (ent <= b) & cert
    if m.any():
        mids.append((a + b) / 2); depths.append(halt[m, 0].mean())
ax[2].plot(mids, depths, "o-", color="#a23b72")
ax[2].set_xlabel("next-char entropy (nats)  (high = hard)")
ax[2].set_ylabel("mean certified halt depth")
ax[2].set_title("Thinks longer on genuinely uncertain\nlanguage? (real-token adaptivity)")
for a in ax: a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("horizon_lm_probe.png", dpi=140)
print("saved plot + json")
