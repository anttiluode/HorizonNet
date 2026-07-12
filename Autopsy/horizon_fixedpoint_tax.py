"""
The Fixed-Point Tax
===================
Accident from the directional run: the certified-at-depth-9 predictions were
MORE accurate (0.421) than the 300-iteration reference (0.385).

That should be impossible under the Horizon Net premise. The premise is:
x* (infinite depth) is the answer; early halting is a compute saving that
we certify costs us nothing. If x* is WORSE than the depth-10 iterate, then
the certificate is proving agreement with a degraded answer, and "halting
early" was never a saving -- it was the whole model.

Hypothesis: a weight-tied net trained by unrolling K=10 steps is not a DEQ.
Training optimizes the K-th ITERATE, not the fixed point. Nothing in the
loss says the fixed point should be good. On an easy task (majority vote,
100% acc) there is no headroom to see the gap. On a graded task (language)
there is.

Measurement: val loss and next-char accuracy vs iteration depth, 1..250,
on 4000 held-out contexts. If accuracy peaks near the training depth and
DECAYS toward the fixed point, the premise is broken for unrolled models.

Control: also report ||z_n - z*|| on the same axis, to show the model is
walking steadily TOWARD the fixed point while getting worse.
"""
import json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(1); np.random.seed(1); torch.set_num_threads(1)

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
TRAIN_ITERS = 10

N, DEEP = 4000, 250
ix = torch.randint(0, len(val_data) - T - 1, (N,))
x = torch.stack([val_data[i:i + T] for i in ix])
y = torch.stack([val_data[i + 1:i + T + 1] for i in ix])

accs, losses, dists, states = [], [], [], []
with torch.no_grad():
    u = net.u(x); z = torch.zeros(N, T, D)
    for it in range(1, DEEP + 1):
        z = net.step(z, u)
        lg = net.head(z)
        accs.append((lg.argmax(-1) == y).float().mean().item())
        losses.append(F.cross_entropy(lg.reshape(-1, V), y.reshape(-1)).item())
        states.append(z[:, -1, :].clone())
    zstar = states[-1]
    dists = [float((zstar - s).norm(dim=1).median()) for s in states]

accs, losses = np.array(accs), np.array(losses)
best = int(accs.argmax()) + 1
res = dict(
    train_depth=TRAIN_ITERS,
    best_depth=best,
    acc_at_train_depth=float(accs[TRAIN_ITERS - 1]),
    acc_at_best=float(accs[best - 1]),
    acc_at_60=float(accs[59]),
    acc_at_fixed_point=float(accs[-1]),
    loss_at_train_depth=float(losses[TRAIN_ITERS - 1]),
    loss_at_60=float(losses[59]),
    loss_at_fixed_point=float(losses[-1]),
    acc_drop_train_to_fp=float(accs[TRAIN_ITERS - 1] - accs[-1]),
    loss_rise_train_to_fp=float(losses[-1] - losses[TRAIN_ITERS - 1]),
    dist_to_fp_at_train_depth=dists[TRAIN_ITERS - 1],
    dist_to_fp_at_fp=dists[-1],
    n=N,
)
print(json.dumps(res, indent=2))
json.dump(res, open("horizon_fixedpoint_tax.json", "w"), indent=2)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
it = np.arange(1, DEEP + 1)
ax[0].plot(it, accs, color="#2a9d8f", lw=2)
ax[0].axvline(TRAIN_ITERS, color="g", ls=":", lw=2, label=f"training depth ({TRAIN_ITERS})")
ax[0].axhline(accs[-1], color="r", ls="--", lw=1, label=f"fixed point ({accs[-1]:.3f})")
ax[0].set_xscale("log"); ax[0].set_xlabel("iteration depth (log)")
ax[0].set_ylabel("next-char accuracy")
ax[0].set_title("The Fixed-Point Tax\naccuracy PEAKS at training depth, decays toward x*")
ax[0].legend(); ax[0].grid(alpha=0.3)

a2 = ax[1].twinx()
ax[1].plot(it, losses, color="#e76f51", lw=2, label="val loss")
a2.semilogy(it, dists, color="#264653", lw=1.5, ls="--", label="‖z_n − z*‖")
ax[1].axvline(TRAIN_ITERS, color="g", ls=":", lw=2)
ax[1].set_xscale("log"); ax[1].set_xlabel("iteration depth (log)")
ax[1].set_ylabel("val loss", color="#e76f51"); a2.set_ylabel("distance to fixed point", color="#264653")
ax[1].set_title("Converging steadily toward x* — and getting worse doing it")
ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("horizon_fixedpoint_tax.png", dpi=140)
print("saved")
