# Experiment 3: the kill-or-confirm test, and what it killed

*Do not hype. Do not lie. Just show.*

README next-step #2 asked: **does the soft certificate fire on real tokens?**

It does not. And the reason is not the certificate.

---

## The result, in one line

The soft Banach certificate certifies **<1% of characters** on a weight-tied causal
transformer trained on tiny Shakespeare. Replicated independently on two machines
(0.5% and 1.0%, ρ_eff 0.90 and 0.88, L_read 6.3 and 7.8). This is a clean kill.

But the autopsy overturned the obvious diagnosis, and the second autopsy overturned
the premise of the whole repo.

---

## Autopsy 1 — the estimator is innocent

The suspicion was that non-monotone updates (84% of sequences re-accelerate after
iteration 8) blind the conservative estimator ρ̂ = max(last 4 ratios), which reads
≈1 off a single bump and refuses to fire even at the fixed point. *Fix the
oscillation, fix the certificate.*

Wrong. Measured against a 300-iteration reference (the true `z*`, which the online
bound never sees):

| quantity | value |
|---|---|
| true ρ from the actual fixed point (tail) | **0.959** |
| ρ̂ from the online estimator, iter 20 | 0.976 |
| **bound / true error ‖z* − z_n‖** | **2.2×** |

The bound is *tight*. The estimator is barely conservative. The block genuinely
converges — slowly, at ρ ≈ 0.96. Damping interventions ("monotone gating") smooth
the curves and buy **nothing**, because oscillation was never the disease.

Arithmetic of the failure: ρ/(1−ρ) = 24, times 2L = 12.7, gives a coefficient of
**304**. Median logit margin on language is **0.79**. So the certificate demands
‖z_n − z_{n−1}‖ < 0.0026 — which arrives around iteration **150**. The 60-iteration
cap was never going to see it.

**And yet:** the *oracle* halt — the last iteration at which the predicted character
ever changes — has median **8.5**, with 54% of tokens settled by iteration 10.

The answer is decided at ~8. The theorem needs ~150 to say so. A 15–20× gap between
knowing and proving, on a bound that is only 2.2× loose. **The slack is not in the
contraction estimate. It is in `L = ‖W‖₂`** — a worst-case response of a 65-way head
to a worst-case direction, applied to an error that is not worst-case.

## Attempted fix — the directional certificate (FALSIFIED)

Near a fixed point the error `e_n = Σ_{k≥n}(z_{k+1} − z_k)` should align with the
dominant slow eigenvector of the Jacobian — power iteration. So the last update
direction `v_n` is a free online estimate of the error direction, and the worst-case
constant can be replaced by the projected one `|(w_a − w_b)·v_n|`.

The pre-registered check was: **measure cos(e_n, v_n) against the deep reference.
If it is not ≈1, the fix is unjustified and we say so.**

| | worst-case L | directional |
|---|---|---|
| certified fraction | 25% | **98.5%** |
| median halt | 50 | **9** |
| coefficient | 12.66 | 0.17 (**75× smaller**) |
| **ever-flip violations (to iter 300)** | 0.0% | **34.0%** |
| **cos(e_n, v_n)** | — | **0.35** |

The alignment assumption is false. The directional rule fires beautifully and is
**wrong one time in three**. It is a heuristic early-exit wearing a theorem's coat.
Killed by its own check. Logged as a negative — it is a *good heuristic* (accuracy
is preserved) and a *worthless certificate*, and the whole point of this repo is the
difference between those two things.

## Autopsy 2 — the fixed point is not the answer

Fallout from the directional run: predictions certified at depth 9 were *more*
accurate (0.421) than the 300-iteration reference (0.385). That should be impossible
under the Horizon Net premise, which treats `x*` as ground truth and early halting as
free compute.

Depth sweep, n = 4000 held-out contexts, single forward trajectory:

| depth | next-char accuracy | val loss |
|---|---|---|
| **10 (training depth)** | **0.4381** | **1.925** |
| 11 (peak) | 0.4386 | — |
| 60 | 0.3933 | 2.130 |
| **250 (fixed point)** | **0.3912** | **2.141** |

**The network converges steadily toward its own fixed point and gets monotonically
worse doing it.** −4.7 accuracy points, +0.216 nats (≈24% worse perplexity).

Training unrolls K = 10 steps. The loss optimizes the **K-th iterate**. Nothing in the
loss ever sees `x*`. For an unrolled weight-tied model, the fixed point is not the
model — it is an extrapolation artifact.

## What that means for the certificate

**Horizon Net has an unstated precondition, and it is the whole ballgame.**

The certificate proves *"the infinite-depth answer equals the current one."* That is
only worth proving if the infinite-depth answer is the one you want. Which explains
the entire lineage:

- **Experiment 1** (fixed random contracting cell + ridge readout): no unrolled
  training exists, so the fixed point *is* the model, by construction. Certificate
  perfect. **The precondition held for free.**
- **Experiment 2** (majority vote, 100% reference accuracy): saturated. No headroom
  for degradation to show. **The precondition was untested, not satisfied.**
- **Experiment 3** (language, graded, genuinely uncertain): precondition **violated**,
  certificate correctly refuses to certify agreement with a degraded answer.

## Prior art (found, and it changes the plan)

This is **path independence**, and it is known:
Anil, Pokle, Liang, Treutlein, Wu, Bai, Kolter, Grosse (2022),
*Path Independent Equilibrium Models Can Better Exploit Test-Time Computation*,
[arXiv:2211.09961](https://arxiv.org/abs/2211.09961). They report that models with
low Asymptotic Alignment (AA) score degrade when inference depth exceeds training
depth, and that non-path-independent networks' in-distribution performance falls with
deeper inference. They give the diagnostic (AA score) and the interventions
(phantom gradients, IFT gradients, mixed initialization).

**The fixed-point tax is not new. Claiming it would be a lie.** We walked into a
known wall from the outside. What is arguably new — and is the actual contribution of
this experiment — is the *weld*:

> **Path independence is the precondition of certified anytime inference, and the AA
> score is its precondition check.** A Banach/margin certificate on a
> non-path-independent model is certifying agreement with an answer the training loss
> never endorsed. Check AA first; certify second.

That claim also needs a literature pass before it is asserted. But it is a sharper
statement of what Horizon Net is *for* than the repo currently makes, and it turns a
null result into a specification.

## Next steps (now specified, not guessed)

1. **Train for path independence** (phantom gradients or IFT, per Anil et al.), then
   re-run this exact probe. Prediction: the depth-accuracy curve flattens or rises,
   and the certificate becomes worth firing.
2. **Spectral-normalize the readout head.** L = 6.33 enters the bound multiplicatively
   and turns into a coefficient of 304. Pinning ‖W‖₂ = 1 divides that by 6.3 for free.
   Nobody pulled this lever. It costs one line.
3. **Report the AA score alongside every certificate.** A certificate without it is
   uninterpretable.
4. **The pixel that isn't there.** The optical origin worked because the observer's
   resolution is a *hard, known* quantity: one pixel. In a classifier the "pixel" is
   the logit margin. Language has no pixels — margins are continuous, small, and
   genuinely ambiguous (median 0.79). The resolution-horizon analogy breaks exactly
   where the observer's grid stops existing. Worth stating in the README.

## Honest ledger

- Single seed, single scale (D=64, T=32, 2500 steps, 1 CPU core). The tax magnitude
  (−4.7 pts / +0.22 nats) is **one model**. It needs seeds, depths, and a second task
  before it is a number rather than an anecdote.
- The 0% ever-flip violation for the worst-case certificate is over **101 certified
  samples** out of 400. Low-n. "0%" means "<3% at 95% confidence," not zero.
- Experiment 2's headline "0 violations, 100% certified" should be re-audited under
  the ever-flip test used here (disagreement at *any* later depth, not just the final
  one). It was never run against that standard.
- The oracle halt is computed against a 300-iteration reference and is therefore a
  statement about *this* trajectory, not a deployable rule. It is a measurement of
  available headroom, not a method.
- No claim is made that the certificate cannot work on language. The claim is that it
  cannot work on *this* model, for a reason that is now named and has a known fix.

## Files

```
horizon_lm_probe.py          exp 3: soft certificate on char-LM (the kill)
horizon_lm_autopsy.py        true fixed-point error vs bound; oracle halt
horizon_directional.py       directional certificate + its falsification
horizon_fixedpoint_tax.py    accuracy vs depth out to the fixed point
```
