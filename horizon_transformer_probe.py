"""
Horizon Transformer Probe
=========================
Question (Gemini's): does the resolution-horizon certificate survive contact
with attention?

Setup: a weight-tied Transformer block iterated to a fixed point (DEQ-style,
with input injection). NO spectral-norm constraint -- contraction is not
guaranteed by construction. At inference we run an ONLINE estimator:

  d_n = ||z_n - z_{n-1}||_F                       (state-update sequence)
  rho_hat_n = conservative estimate from recent ratios d_k/d_{k-1}
  soft Banach bound:  ||z* - z_n|| <= rho_hat/(1-rho_hat) * d_n   (if rho_hat<1)

Push the bound through the readout Lipschitz constant L to get a logit
perturbation bound, halt when 2*L*bound < logit margin. Because rho_hat is
empirical, the certificate is SOFT: we must measure its violation rate
against a deep reference, not assume it.

Task: majority vote over a binary sequence (length 25). Difficulty has a
natural spectrum: |#1s - #0s| = vote margin. Hard tokens = near-tied votes.

Honest ledger of what this can and cannot show is in the results file.
"""
import math, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0); np.random.seed(0)
DEV = "cpu"

# ---------------- task ----------------
SEQ = 25
def make_batch(n):
    x = torch.randint(0, 2, (n, SEQ))
    s = x.sum(1)
    y = (s > SEQ // 2).long()          # majority (SEQ odd -> no ties)
    diff = (2 * s - SEQ).abs()          # vote margin = difficulty (small=hard)
    return x, y, diff

# ---------------- model ----------------
D, H, FFD = 64, 4, 128
class TiedBlock(nn.Module):
    """One weight-tied transformer block f(z, x) iterated to a fixed point."""
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(2, D)
        self.pos  = nn.Parameter(torch.randn(SEQ, D) * 0.02)
        self.attn = nn.MultiheadAttention(D, H, batch_first=True)
        self.ff   = nn.Sequential(nn.Linear(D, FFD), nn.GELU(), nn.Linear(FFD, D))
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.inj  = nn.Linear(D, D)
        self.head = nn.Linear(D, 2)
        self.alpha = 0.5                # damping z <- (1-a)z + a f(z,x)

    def u(self, x):                     # input injection, computed once
        return self.inj(self.emb(x) + self.pos)

    def step(self, z, u):
        h = self.ln1(z + self.attn(z, z, z, need_weights=False)[0] + u)
        f = self.ln2(h + self.ff(h))
        return (1 - self.alpha) * z + self.alpha * f

    def logits(self, z):
        return self.head(z.mean(1))     # mean-pool readout

    def forward(self, x, iters):
        u = self.u(x); z = torch.zeros(x.shape[0], SEQ, D)
        for _ in range(iters):
            z = self.step(z, u)
        return self.logits(z)

net = TiedBlock().to(DEV)

# ---------------- train (12 unrolled iterations, weight-tied) ----------------
opt = torch.optim.Adam(net.parameters(), lr=3e-4)
TRAIN_ITERS = 12
for step in range(1200):
    x, y, _ = make_batch(64)
    loss = F.cross_entropy(net(x, TRAIN_ITERS), y)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 300 == 0:
        with torch.no_grad():
            xa, ya, _ = make_batch(512)
            acc = (net(xa, TRAIN_ITERS).argmax(1) == ya).float().mean().item()
        print(f"step {step:4d} loss {loss.item():.4f} acc@12 {acc:.3f}")

# readout Lipschitz: logits = W @ mean(z); ||dlogits|| <= ||W||_2 * ||dz||_F/sqrt(T)
with torch.no_grad():
    W = net.head.weight
    L_read = torch.linalg.matrix_norm(W, 2).item() / math.sqrt(SEQ)

# ---------------- inference probe with online estimator ----------------
MAX_IT, N_TEST, WIN = 150, 1000, 4
@torch.no_grad()
def probe(x):
    """Returns per-sample: halt iter, certified pred, ref pred, rho trace."""
    n = x.shape[0]; u = net.u(x)
    z = torch.zeros(n, SEQ, D); z_prev = z.clone()
    d_hist = [[] for _ in range(n)]
    halted = torch.zeros(n, dtype=torch.long); pred_halt = torch.full((n,), -1)
    rho_at_halt = torch.zeros(n)
    for it in range(1, MAX_IT + 1):
        z_new = net.step(z, u)
        d = (z_new - z).flatten(1).norm(dim=1)           # ||z_n - z_{n-1}||
        for i in range(n):
            d_hist[i].append(d[i].item())
        logit = net.logits(z_new)
        top2 = logit.topk(2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1])
        for i in range(n):
            if halted[i]: continue
            hs = d_hist[i]
            if len(hs) >= WIN + 1:
                ratios = [hs[-k] / max(hs[-k - 1], 1e-12) for k in range(1, WIN + 1)]
                rho = max(ratios)                        # conservative: worst recent ratio
                if rho < 0.999:
                    bound = rho / (1 - rho) * hs[-1]     # soft Banach a posteriori
                    if 2 * L_read * bound < margin[i].item():
                        halted[i] = it
                        pred_halt[i] = logit[i].argmax().item()
                        rho_at_halt[i] = rho
        z_prev, z = z, z_new
        if halted.all(): break
    # reference: run to MAX_IT
    ref = net(x, MAX_IT).argmax(1)
    for i in range(n):                                    # never-certified -> ref answer at cap
        if halted[i] == 0:
            halted[i] = MAX_IT; pred_halt[i] = ref[i]; rho_at_halt[i] = float("nan")
    return halted, pred_halt, ref, rho_at_halt, d_hist

x, y, diff = make_batch(N_TEST)
halt, pred, ref, rho_h, d_hist = probe(x)

certified = torch.tensor([not math.isnan(r) for r in rho_h])
agree = (pred == ref)
res = dict(
    n=N_TEST,
    certified_frac=certified.float().mean().item(),
    mean_halt=halt[certified].float().mean().item() if certified.any() else None,
    mean_halt_all=halt.float().mean().item(),
    ref_iters=MAX_IT,
    violation_rate=(~agree[certified]).float().mean().item() if certified.any() else None,
    acc_ref=(ref == y).float().mean().item(),
    acc_cert=(pred == y).float().mean().item(),
    rho_eff_median=float(np.nanmedian(rho_h.numpy())),
    L_readout=L_read,
)
# difficulty adaptivity: halt depth vs vote margin
uniq = sorted(set(diff.tolist()))
depth_by_diff = {int(m): halt[(diff == m) & certified].float().mean().item()
                 for m in uniq if ((diff == m) & certified).any()}
res["depth_by_difficulty"] = depth_by_diff
print(json.dumps(res, indent=2))

# ---------------- plots ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

# (1) update norms: does attention even contract?
for i in range(0, 40):
    ax[0].semilogy(d_hist[i], lw=0.7, alpha=0.5)
ax[0].set_xlabel("iteration"); ax[0].set_ylabel(r"$\|z_n - z_{n-1}\|_F$")
ax[0].set_title(f"Update norms (40 samples)\nmedian ρ_eff at halt = {res['rho_eff_median']:.3f}")

# (2) halt histogram
hh = halt[certified].numpy() if certified.any() else halt.numpy()
ax[1].hist(hh, bins=range(0, int(hh.max()) + 2), color="#3a7ca5", edgecolor="k")
ax[1].axvline(MAX_IT, color="r", ls="--", label=f"reference ({MAX_IT})")
viol = res["violation_rate"]
ax[1].set_title(f"Certified halts: {100*res['certified_frac']:.1f}% of samples\n"
                f"mean {res['mean_halt']:.1f} it, violations {100*viol:.2f}%")
ax[1].set_xlabel("halt iteration"); ax[1].legend()

# (3) difficulty adaptivity
ms = sorted(depth_by_diff); ax[2].plot(ms, [depth_by_diff[m] for m in ms], "o-", color="#a23b72")
ax[2].set_xlabel("vote margin |#1 − #0|  (small = hard)")
ax[2].set_ylabel("mean certified halt depth")
ax[2].set_title("Does the network think longer on hard tokens?")
for a in ax: a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("horizon_transformer_probe.png", dpi=140)
json.dump(res, open("horizon_transformer_probe.json", "w"), indent=2)
print("saved plot + json")
