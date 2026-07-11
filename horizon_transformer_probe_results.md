# Horizon Transformer Probe — Results

**Question (Gemini's):** does the resolution-horizon certificate survive contact with attention, when contraction is *estimated on the fly* rather than guaranteed by construction?

**Answer from this run: yes, on this task, with the certificate now soft rather than proven — and the honest metric for a soft certificate is its measured violation rate, which was 0/1000.**

## Setup

- Weight-tied transformer block (4-head attention + FFN + LayerNorms, d=64), DEQ-style input injection, damping α=0.5, iterated to a fixed point. **No spectral-norm constraint anywhere** — nothing in the architecture guarantees contraction.
- Task: majority vote over a binary sequence of length 25. Natural difficulty spectrum: vote margin |#1 − #0| (1 = near-tie = hard, 17 = landslide = easy).
- Online estimator during inference: track d_n = ‖z_n − z_{n−1}‖_F, take ρ̂ = max of the last 4 ratios (conservative), form the soft Banach bound ρ̂/(1−ρ̂)·d_n, push through the readout Lipschitz constant (‖W‖₂/√T = 0.168), halt when 2·L·bound < logit margin.
- Reference: 150 iterations. n = 1000 test sequences.

## Results

1. **Contraction emerged without being imposed.** Median ρ_eff at halt = 0.788. Training a weight-tied block through 12 unrolled iterations apparently pressures the map toward contraction on its own (the damping helps). This was the open question — attention has no global Lipschitz guarantee, and the update norms still decay geometrically after a short transient (left panel).

2. **The soft certificate fired for 100% of samples, mean halt 9.7 vs 150 reference (~15×), with 0 violations in 1000.** Since ρ̂ is empirical, zero observed violations means violation rate < 0.3% at 95% confidence (rule of three) — *not* zero. This is the correct claim for a soft certificate and it is weaker than the hard version's theorem, on purpose.

3. **Per-sample difficulty adaptivity appeared — the thing the hard-contraction MLP version lacked.** Certified halt depth falls monotonically from 11 iterations at vote margin 1 (near-tie) to 7 at margin 17 (landslide). The network provably-enough "thinks longer" on hard tokens, by ~4 iterations. Curious detail: within a difficulty class, halts are essentially identical across sequences — the halting dynamics are dominated by the margin, not by individual sequence structure.

4. **The log law is compressed, again by saturation.** Naively, depth spread should be log(17/1)/log(1/0.788) ≈ 12 iterations; observed spread is 4. Same phenomenon as Horizon Net and as the lensing-beats-mirror effect: the trained readout compresses logit margins, so the difficulty spectrum arrives pre-squashed. Weaker readout saturation ⇒ more visible adaptivity. (Prediction, testable: retrain with a temperature that spreads logit margins and the depth spread should widen toward the log law.)

## Honest ledger

- **Toy task, and an easy one** (reference accuracy 100%). Majority vote was chosen for its clean difficulty dial, not its realism. Nothing here says the certificate fires usefully on language modeling.
- **The certificate is soft.** ρ̂ from 4 recent ratios can be fooled by non-monotone convergence (plateau then re-acceleration). It didn't happen here; it can happen in principle. A production version should include a safety margin on ρ̂ or a Popper clause: keep monitoring one extra step after halting on a validation stream.
- **Single seed, single architecture size.** No claim of robustness across scales.
- **L_readout ignores LayerNorm's Jacobian** between z and the pooled representation in the strictest sense — folded into "soft."
- **The 15× speedup is against an absurd 150-iteration reference.** Against a sensibly chosen fixed depth (say 12, what it was trained with), the win is not speed, it's the *certificate*: you know, per-sample, when 12 would have been too few.

## Where this sits

The weld being tested: **online empirical contraction estimate × Banach a posteriori bound × decision margin = certified-enough anytime inference, on an architecture with no Lipschitz guarantee.** Adjacent prior art: DEQ residual tolerances (heuristic, no margin coupling), ACT/PonderNet (learned heuristic halting), Lipschitz-margin robustness certificates (static, not anytime), monotone-operator DEQs (guarantee by construction, the thing we deliberately dropped). The specific online-soft-certificate weld is the part the literature pass needs to target before any strong claim.

## Files

- `horizon_transformer_probe.py` — full experiment, seeded
- `horizon_transformer_probe.png` — three panels: update-norm decay, halt histogram, depth vs difficulty
- `horizon_transformer_probe.json` — raw numbers
