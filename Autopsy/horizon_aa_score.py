"""
Is the bulk determined by the boundary?
=======================================
Asymptotic Alignment (AA) score, after Anil et al. 2022 (arXiv:2211.09961).

Operational content of the holographic claim, stated so it can be falsified:

    Fix the boundary data (the input injection u(x)).
    Start the interior state z_0 from several DIFFERENT initializations.
    Iterate the same tied operator to its fixed point from each.

    If the boundary determines the bulk, all trajectories land on the SAME z*,
    regardless of path. cos(z*_a, z*_b) -> 1.

    If they land in different places, the interior depends on the PATH, not the
    boundary. The system is not holographic in the only sense that has content.

This is exactly path independence. It is also exactly the unstated precondition
of the Horizon certificate: a Banach bound proves "the current answer equals the
infinite-depth one," which is only worth proving if the infinite-depth one is
unique and is the answer the model was trained to give.

We measure, on the SAME checkpoint that failed the certificate:
  AA_state  : cos similarity between fixed points from different inits
  AA_logit  : cos similarity between the resulting next-char logit vectors
  AA_argmax : fraction of tokens where the PREDICTION agrees across inits
  and the same three at the training depth (K=10), for contrast.

Control: a genuinely path-independent operator has AA ~ 1 by construction.
Anil et al. report AA ~ 0.96-0.99 for path-independent DEQs.
"""
import json
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(7); np.random.seed(7); torch.set_num_threads(1)

text = open("shakespeare.txt").read()
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
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
K, DEEP, N = 10, 250, 1500

ix = torch.randint(0, len(val_data) - T - 1, (N,))
x = torch.stack([val_data[i:i + T] for i in ix])
y_last = torch.stack([val_data[i + T] for i in ix]).numpy()

# initializations: the canonical one (zeros) + Gaussian noise at 3 scales
INITS = {
    "zeros":       lambda: torch.zeros(N, T, D),
    "gauss_0.1":   lambda: torch.randn(N, T, D) * 0.1,
    "gauss_1.0":   lambda: torch.randn(N, T, D) * 1.0,
    "gauss_5.0":   lambda: torch.randn(N, T, D) * 5.0,
}

@torch.no_grad()
def run(z0):
    u = net.u(x); z = z0
    zK = None
    for it in range(1, DEEP + 1):
        z = net.step(z, u)
        if it == K:
            zK = z[:, -1, :].clone()
    return zK, z[:, -1, :]              # state at training depth K, and at fixed point

runs = {}
for name, mk in INITS.items():
    torch.manual_seed(hash(name) % 10000)
    zK, zs = run(mk())
    runs[name] = dict(zK=zK, zstar=zs,
                      lgK=net.head(zK), lg=net.head(zs))
    print(f"{name:12s}  acc@K={(runs[name]['lgK'].argmax(1).numpy()==y_last).mean():.4f}  "
          f"acc@fp={(runs[name]['lg'].argmax(1).numpy()==y_last).mean():.4f}", flush=True)

base = "zeros"
res = {"n": N, "train_depth": K, "deep": DEEP, "pairs": {}}
for name in INITS:
    if name == base: continue
    a, b = runs[base], runs[name]
    cs = lambda p, q: F.cosine_similarity(p, q, dim=1).median().item()
    res["pairs"][f"{base} vs {name}"] = dict(
        AA_state_fixedpoint=cs(a["zstar"], b["zstar"]),
        AA_logit_fixedpoint=cs(a["lg"], b["lg"]),
        AA_argmax_fixedpoint=float((a["lg"].argmax(1) == b["lg"].argmax(1)).float().mean()),
        AA_state_at_K=cs(a["zK"], b["zK"]),
        AA_argmax_at_K=float((a["lgK"].argmax(1) == b["lgK"].argmax(1)).float().mean()),
        rel_dist_fixedpoint=float(((a["zstar"] - b["zstar"]).norm(dim=1) /
                                   a["zstar"].norm(dim=1)).median()),
    )

fp_aa = [v["AA_state_fixedpoint"] for v in res["pairs"].values()]
fp_am = [v["AA_argmax_fixedpoint"] for v in res["pairs"].values()]
res["summary"] = dict(
    AA_score_mean=float(np.mean(fp_aa)),
    prediction_agreement_across_inits=float(np.mean(fp_am)),
    verdict=("PATH INDEPENDENT (boundary determines bulk)" if np.mean(fp_aa) > 0.95
             else "PATH DEPENDENT (bulk depends on trajectory, not boundary)"),
    reference_PI_DEQ_AA="0.96-0.99 (Anil et al. 2022)",
)
print(json.dumps(res, indent=2))
json.dump(res, open("horizon_aa_score.json", "w"), indent=2)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
for name in INITS:
    if name == base: continue
    c = F.cosine_similarity(runs[base]["zstar"], runs[name]["zstar"], dim=1).numpy()
    ax[0].hist(c, bins=50, alpha=0.55, label=f"vs {name}")
ax[0].axvline(0.96, color="g", ls="--", lw=2, label="path-independent DEQ (Anil et al.)")
ax[0].set_xlabel("cos( z*(from zeros), z*(from noise) )")
ax[0].set_title(f"Does the boundary determine the bulk?\nAA = {res['summary']['AA_score_mean']:.3f}")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3); ax[0].set_xlim(-1, 1.05)

names = [n for n in INITS if n != base]
xs = np.arange(len(names))
ax[1].bar(xs - 0.2, [res["pairs"][f"{base} vs {n}"]["AA_argmax_at_K"] for n in names],
          0.4, label=f"prediction agrees @ depth {K}", color="#2a9d8f")
ax[1].bar(xs + 0.2, [res["pairs"][f"{base} vs {n}"]["AA_argmax_fixedpoint"] for n in names],
          0.4, label="prediction agrees @ fixed point", color="#e76f51")
ax[1].set_xticks(xs); ax[1].set_xticklabels(names); ax[1].set_ylim(0, 1)
ax[1].axhline(1.0, color="k", lw=0.8)
ax[1].set_title("Same boundary, different path → same answer?")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("horizon_aa_score.png", dpi=140)
print("saved")
