"""
The Optimum, Not The Limit
==========================
Two questions, one sweep.

Q1 (the law). Weight-tied nets trained by unrolling K steps: is the accuracy
    peak PINNED TO K? Train K in {5,10,20,40}, sweep inference depth 1..300.
      - peak tracks K            -> the model is a K-step algorithm at every
                                    scale; the fixed point is an artifact.
      - peak sits at ~11 always  -> it is the task/architecture, not the
                                    unrolling. Story collapses. Report that.

Q2 (the halting condition). Rewrite the target of the certificate.

    INFINITE target (Horizon Net as published):
        ||x* - z_n||  <=  d_n * rho/(1-rho)
    FINITE target (this work) -- bound the distance to the iterate training
    actually optimized, z_K:
        ||z_K - z_n|| <= sum_{k=n}^{K-1} ||z_{k+1}-z_k||
                      <= d_n * rho*(1 - rho^{K-n})/(1-rho)

    Identical machinery. One extra factor (1 - rho^{K-n}), which is small when
    n is close to K. Halt when 2*L*bound < margin. Certified prediction is then
    provably the DEPTH-K prediction -- the model that was trained -- rather
    than provably the equilibrium prediction, which we have shown is worse.

    Violation test: disagreement with argmax at depth K (exactly computable).
    Oracle: earliest n whose argmax matches argmax_K and never deviates before
    K. This is the ceiling for ANY halting rule targeting z_K.

Also per model: AA score (path independence of the fixed point), rho_eff.

Honest scope: one seed per K (plus 2 extra seeds at K=10), D=64, char-level
Shakespeare, 1 CPU core. Enough to see whether the effect is there and scales.
Not enough to claim a constant.
"""
import json, time, math, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(1)

text = open("shakespeare.txt").read()
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n_tr = int(0.95 * len(data)); train_data, val_data = data[:n_tr], data[n_tr:]
T, D, H, FFD = 32, 64, 4, 128
CAUSAL = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
STEPS, B, DEEP, NSWEEP, NCERT, WIN = 1800, 32, 300, 2000, 500, 4

def get_batch(split, n, gen=None):
    src = train_data if split == "train" else val_data
    ix = torch.randint(0, len(src) - T - 1, (n,), generator=gen)
    return (torch.stack([src[i:i + T] for i in ix]),
            torch.stack([src[i + 1:i + T + 1] for i in ix]))

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

def train(K, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = TiedLMBlock()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    t0 = time.time()
    for s in range(STEPS):
        x, y = get_batch("train", B)
        loss = F.cross_entropy(net(x, K).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"  [K={K} seed={seed}] trained in {time.time()-t0:.0f}s, loss {loss.item():.3f}", flush=True)
    return net

@torch.no_grad()
def depth_sweep(net, K):
    """accuracy + loss vs inference depth, on fixed val set"""
    g = torch.Generator().manual_seed(1234)
    x, y = get_batch("val", NSWEEP, g)
    u = net.u(x); z = torch.zeros(NSWEEP, T, D)
    accs, losses = [], []
    for it in range(1, DEEP + 1):
        z = net.step(z, u)
        lg = net.head(z)
        accs.append((lg.argmax(-1) == y).float().mean().item())
        losses.append(F.cross_entropy(lg.reshape(-1, V), y.reshape(-1)).item())
    return np.array(accs), np.array(losses)

@torch.no_grad()
def aa_score(net):
    """path independence of the fixed point: same boundary, different interior"""
    g = torch.Generator().manual_seed(99)
    x, _ = get_batch("val", 400, g)
    u = net.u(x)
    fps = []
    for scale in (0.0, 1.0, 5.0):
        torch.manual_seed(int(scale * 13) + 3)
        z = torch.randn(400, T, D) * scale if scale > 0 else torch.zeros(400, T, D)
        for _ in range(DEEP): z = net.step(z, u)
        fps.append(z[:, -1, :].clone())
    return float(np.mean([F.cosine_similarity(fps[0], f, dim=1).median().item()
                          for f in fps[1:]]))

@torch.no_grad()
def certificates(net, K):
    """finite-K target vs infinite target, same Banach machinery"""
    g = torch.Generator().manual_seed(555)
    x, y = get_batch("val", NCERT, g)
    L = torch.linalg.matrix_norm(net.head.weight, 2).item()
    u = net.u(x); z = torch.zeros(NCERT, T, D)
    d, am, mg = [], [], []
    for it in range(DEEP):
        zn = net.step(z, u)
        zl, zp = zn[:, -1, :], z[:, -1, :]
        d.append((zl - zp).norm(dim=1).numpy())
        lg = net.head(zl); t2 = lg.topk(2, 1).values
        mg.append((t2[:, 0] - t2[:, 1]).numpy()); am.append(lg.argmax(1).numpy())
        z = zn
    d = np.stack(d, 1); am = np.stack(am, 1); mg = np.stack(mg, 1)
    target_K   = am[:, K - 1]                     # the trained model's answer
    target_inf = am[:, -1]                        # the equilibrium's answer
    y_last = y[:, -1].numpy()

    out = {}
    for name, target, cap in [("finite_K", target_K, K), ("infinite", target_inf, DEEP)]:
        halt = np.zeros(NCERT, dtype=int); pred = np.full(NCERT, -1)
        for it in range(WIN + 1, cap + 1):
            i = it - 1
            if name == "finite_K" and it >= K: break   # no saving past K-1
            ratios = d[:, i - WIN + 1:i + 1] / np.maximum(d[:, i - WIN:i], 1e-12)
            rho = np.clip(ratios.max(1), 0, 0.9999)
            tail = rho / (1 - rho)
            if name == "finite_K":
                tail = tail * (1 - rho ** (K - it))     # <-- the whole change
            bnd = tail * d[:, i]
            fire = (halt == 0) & (2 * L * bnd < mg[:, i])
            halt[fire] = it; pred[fire] = am[fire, i]
        cert = halt > 0
        halt[~cert] = cap; pred[~cert] = target[~cert]
        out[name] = dict(
            certified_frac=float(cert.mean()),
            mean_halt=float(halt[cert].mean()) if cert.any() else None,
            compute_saved=float(1 - halt.mean() / cap),
            violations_vs_target=float((pred[cert] != target[cert]).mean()) if cert.any() else None,
            acc_certified=float((pred == y_last).mean()),
        )
    # oracle for the finite-K target: earliest n with argmax stable through K
    stable = (am[:, :K] == target_K[:, None])
    oracle = np.array([K - np.argmin(stable[i, :K][::-1]) if not stable[i].all() else 1
                       for i in range(NCERT)])
    out["oracle_finite_K"] = dict(
        mean_halt=float(oracle.mean()),
        max_compute_saved=float(1 - oracle.mean() / K),
        frac_settled_by_half_K=float((oracle <= max(1, K // 2)).mean()),
    )
    out["L_readout"] = L
    out["acc_at_K"] = float((target_K == y_last).mean())
    out["acc_at_fixedpoint"] = float((target_inf == y_last).mean())
    return out

RUNS = [(5, 0), (10, 0), (20, 0), (40, 0), (10, 1), (10, 2)]
results = {}
for K, seed in RUNS:
    tag = f"K{K}_s{seed}"
    print(f"=== {tag} ===", flush=True)
    net = train(K, seed)
    accs, losses = depth_sweep(net, K)
    peak = int(accs.argmax()) + 1
    r = dict(
        K=K, seed=seed,
        peak_depth=peak,
        acc_at_peak=float(accs[peak - 1]),
        acc_at_K=float(accs[K - 1]),
        acc_at_fixedpoint=float(accs[-1]),
        loss_at_K=float(losses[K - 1]),
        loss_at_fixedpoint=float(losses[-1]),
        transient_premium_acc=float(accs[K - 1] - accs[-1]),
        transient_premium_nats=float(losses[-1] - losses[K - 1]),
        AA=aa_score(net),
        cert=certificates(net, K),
        acc_curve=[float(a) for a in accs],
    )
    results[tag] = r
    print(json.dumps({k: v for k, v in r.items() if k != "acc_curve"}, indent=2), flush=True)
    json.dump(results, open("horizon_optimum.json", "w"), indent=2)
    torch.save(net.state_dict(), f"lm_{tag}.pt")

print("ALL DONE")
