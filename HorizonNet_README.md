# Horizon Net

**Certified anytime inference: a weight-tied network that stops computing the moment it can prove — or measure — that thinking longer cannot change its answer.**

No hype clause, up front: every mathematical ingredient here is old (Banach 1922, Lipschitz margins, deep equilibrium models). What this repo contributes is a specific weld of those ingredients, two working implementations, and measured results with the failure modes logged. Whether the weld itself is new is an open question; see *Open questions* below.

---

## Where this came from

This did not start as machine learning. It started as a rendering trick in a WebGL game, [Antti's Brain 2](https://github.com/anttiluode/AnttisBrain2): a hall of convex mirrors that shows apparently infinite recursive reflections while computing only **four** bounces. Four is not a hack — it is the provably correct truncation depth, because each bounce both dims the image (amplitude contraction) and shrinks it (spatial contraction), and after four applications the remaining structure falls below one pixel. The infinite series genuinely converges at 4 *in the only norm a retina can measure*.

Stated once as a principle: **when you want the appearance of unbounded recursive structure, don't simulate the structure — implement one contracting operator and truncate at the observation horizon.**

Horizon Net is that principle wearing a neural network. The "observer" is the readout layer; the "resolution" is the decision margin; the "bounce count" is the iteration depth. The network iterates a weight-tied cell and halts when the mathematically bounded distance to the infinite-depth answer can no longer flip the decision.

## The math in five lines

For a contracting update map `f` with rate ρ < 1, iterated as `x_{n+1} = f(x_n)`, the Banach a posteriori bound gives

```
‖x* − x_n‖ ≤ ρ/(1−ρ) · ‖x_n − x_{n−1}‖
```

where `x*` is the (never computed) infinite-depth fixed point. Push this through the readout's Lipschitz constant `L` to bound how much the logits can still move, and halt when

```
2 · L · ρ/(1−ρ) · ‖x_n − x_{n−1}‖  <  (top-1 logit − top-2 logit)
```

At that moment the infinite-depth network provably agrees with the current one. Halting is a theorem firing per-sample, not a heuristic.

## The two experiments

### 1. `horizon_net.py` — hard certificate (contraction guaranteed)

Weight-tied cell with spectral norm pinned to ρ = 0.8, so contraction is a construction-time guarantee. Results (n = 2000, six ρ settings):

- Mean halt at **6.2 iterations** vs a 400-iteration reference, **100.0% prediction agreement** — which is the theorem checking out, not a discovery. The certificate is exact.
- **Resolution-horizon law confirmed:** demanding `b` bits of output precision costs depth *linear* in `b`. Measured 0.63 iterations/bit; the empirical contraction (ρ_eff ≈ 0.38, far stronger than the 0.8 guarantee, thanks to tanh saturation) predicts 0.72.
- **Logged negative:** per-sample difficulty adaptivity is weak (~1 iteration between easy and hard). The theory explains it: depth ∝ log(1/margin), and strong contraction compresses the difficulty spectrum. If you want a network that visibly thinks longer on hard inputs, you want *weaker* contraction.

### 2. `horizon_transformer_probe.py` — soft certificate (contraction estimated online)

The question that decides whether any of this matters beyond toys: does the certificate survive an architecture with **no Lipschitz guarantee**? A weight-tied transformer block (attention + FFN + LayerNorms, DEQ-style input injection, damping 0.5) is iterated to a fixed point while an online estimator tracks the update-norm sequence `‖z_n − z_{n−1}‖`, fits ρ_eff on the fly (max of the last 4 ratios, deliberately conservative), and forms a *soft* Banach certificate. Task: majority vote over binary sequences, chosen for its clean difficulty dial (vote margin), not realism. Results (n = 1000):

- **Contraction emerged without being imposed:** median ρ_eff = 0.79. Training through 12 unrolled iterations apparently pressures the map toward contraction on its own.
- **Certificate fired for 100% of samples, mean halt 9.7 vs 150-iteration reference, 0 violations** — which, because ρ_eff is empirical, honestly means *violation rate < 0.3% at 95% confidence* (rule of three), not zero.
- **Difficulty adaptivity appeared**, exactly where experiment 1's negative result predicted it would: halt depth falls monotonically from 11 iterations on near-tied votes to 7 on landslides. The log law is compressed (spread 4 vs naive 12) by readout saturation squashing the margin spectrum — the third appearance of saturation-beats-the-bound in this lineage, after tanh in experiment 1 and lensing-beats-mirror-demagnification in the game.

![transformer probe results](horizon_transformer_probe.png)

## Honest ledger

What these results are **not**:

- **Not tested beyond toy tasks.** Both tasks are small and the transformer task is easy (reference accuracy 100%). Nothing here demonstrates the certificate fires usefully on language modeling or any real workload.
- **The 65× / 15× speedups are against deliberately absurd references** (400 and 150 iterations). Against a sensibly chosen fixed depth, the win is not speed — it is the *certificate*: knowing, per sample, when the fixed depth would have been too shallow, and cutting compute only when it is provably (or measurably) safe.
- **The soft certificate can be fooled in principle** by non-monotone convergence (plateau, then re-acceleration after the estimator has already fired). It did not happen in 1000 samples; it can happen. A production version needs a safety margin on ρ_eff or continued monitoring on a validation stream.
- **The hard certificate constrains capacity.** Pinning the spectral norm restricts what the cell can represent. Flat accuracy across ρ in experiment 1 is a property of that easy task, not a free lunch.
- **Single seeds, single scales.** No robustness claims.
- **Prior-art status unresolved.** Adjacent work exists and is close: DEQ residual tolerances (heuristic stopping, no margin coupling), ACT and PonderNet (learned heuristic halting, no bound), Lipschitz-margin certificates (static robustness, not anytime), monotone-operator DEQs (guarantee by construction — the assumption experiment 2 deliberately drops). The specific weld — *a posteriori contraction bound × decision margin = certified anytime halting, with the contraction rate estimated online on an unconstrained architecture* — is the part we have not found stated. **"Not found" is not "doesn't exist."** A serious literature pass is step one before any strong claim.

Bugs confessed, because the run only exists because it caught them: the first version's certificate could never fire (a stray epsilon made each class fail comparison against itself), and the first theory line used the guaranteed ρ where the empirical rate belonged.

## Open questions / next steps

1. **Literature pass** targeting the weld specifically (search terms: certified early exit, anytime inference guarantees, a posteriori error bounds in implicit models, DEQ stopping criteria).
2. **Non-toy transformer:** measure ρ_eff on a weight-tied block of a small language model and see whether the soft certificate fires on real tokens — the kill-or-confirm test.
3. **Widen the adaptivity:** the compressed log law predicts that spreading logit margins (temperature at training time) should widen the easy/hard depth gap toward the theoretical 12. Testable in an afternoon.
4. **Failure-mode hunt:** construct inputs with non-monotone convergence and measure how badly the online estimator can be fooled; add the safety margin the ledger demands.

## Files

```
horizon_net.py                        experiment 1: hard certificate, MLP cell
horizon_net_results.md / .json / .png results + raw numbers + plots
horizon_transformer_probe.py          experiment 2: soft certificate, attention block
horizon_transformer_probe_results.md  results with full ledger
horizon_transformer_probe.json / .png raw numbers + plots
```

## Reproducing

```bash
pip install numpy matplotlib          # experiment 1
python horizon_net.py

pip install torch                     # experiment 2 (CPU is fine, ~3 min total)
python horizon_transformer_probe.py
```

Both scripts are seeded. If your numbers differ materially from the ones above, that is a result — please open an issue.

---

*Part of the PerceptionLab research program. Motto in force throughout: do not hype, do not lie, just show.*
