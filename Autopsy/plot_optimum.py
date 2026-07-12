import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = json.load(open("horizon_optimum.json"))
fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))

# (a) accuracy vs depth, peak tracks K
cols = {5: "#e76f51", 10: "#2a9d8f", 20: "#264653", 40: "#a23b72"}
for tag, v in r.items():
    if v["seed"] != 0: continue
    K = v["K"]; c = np.array(v["acc_curve"])
    ax[0].plot(np.arange(1, len(c) + 1), c, color=cols[K], lw=1.8, label=f"trained K={K}")
    ax[0].plot(v["peak_depth"], c[v["peak_depth"] - 1], "o", color=cols[K], ms=9,
               mec="k", mew=1.2, zorder=5)
    ax[0].axvline(K, color=cols[K], ls=":", lw=1, alpha=0.6)
ax[0].set_xscale("log"); ax[0].set_xlabel("inference depth (log)")
ax[0].set_ylabel("next-char accuracy")
ax[0].set_title("Competence peaks at the training depth\n(dots = peak, dotted = K)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

# (b) transient premium vs K
Ks, prem, aa = [], [], []
for tag, v in r.items():
    Ks.append(v["K"] + (v["seed"] * 0.4)); prem.append(v["transient_premium_acc"]); aa.append(v["AA"])
ax[1].semilogx(Ks, prem, "o", ms=10, color="#e76f51", mec="k")
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_xlabel("training depth K (log)")
ax[1].set_ylabel("acc@K − acc@fixed point")
ax[1].set_title("Transient premium collapses as K grows\nthe model becomes a true equilibrium model")
ax[1].grid(alpha=0.3)

# (c) THE TENSION: certificate fires where the model is bad
accK = [v["acc_at_K"] for v in r.values()]
cf = [100 * v["cert"]["finite_K"]["certified_frac"] for v in r.values()]
sz = [40 + 4 * v["K"] for v in r.values()]
sc = ax[2].scatter(accK, cf, s=sz, c=[v["K"] for v in r.values()], cmap="viridis_r",
                   edgecolor="k", zorder=5)
for v in r.values():
    ax[2].annotate(f"K={v['K']}", (v["acc_at_K"], 100 * v["cert"]["finite_K"]["certified_frac"]),
                   textcoords="offset points", xytext=(8, 6), fontsize=9)
ax[2].set_xlabel("model quality (accuracy at K)")
ax[2].set_ylabel("% of tokens the certificate can halt")
ax[2].set_title("The certificate fires ONLY on bad models\nbecause it can only detect wasted compute")
ax[2].grid(alpha=0.3)

# (d) the proof gap
tags = list(r.keys())
xs = np.arange(len(tags))
orc = [r[t]["cert"]["oracle_finite_K"]["mean_halt"] for t in tags]
crt = [r[t]["cert"]["finite_K"]["mean_halt"] or r[t]["K"] for t in tags]
kk = [r[t]["K"] for t in tags]
ax[3].bar(xs - 0.26, orc, 0.26, label="oracle: answer already settled", color="#2a9d8f")
ax[3].bar(xs, crt, 0.26, label="certificate: can prove it", color="#e76f51")
ax[3].bar(xs + 0.26, kk, 0.26, label="K: what you actually pay", color="#adb5bd")
ax[3].set_xticks(xs); ax[3].set_xticklabels(tags, rotation=35, fontsize=8)
ax[3].set_ylabel("iterations")
ax[3].set_title("The proof gap\nknowing (teal) vs proving (orange) vs paying (grey)")
ax[3].legend(fontsize=8); ax[3].grid(alpha=0.3)

plt.tight_layout(); plt.savefig("horizon_optimum.png", dpi=140)
print("saved")
