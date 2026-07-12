# The Certificate Can Only Find the Waste

### Certified halting in weight-tied networks recovers compute that was already doing nothing

**Antti Luode, PerceptionLab** · *with Claude (Anthropic)*
Working paper. Single-scale, char-level, CPU. Read the ledger before the abstract.

*Do not hype. Do not lie. Just show.*

---

## Abstract

Horizon Net proposed certified anytime inference: iterate a weight-tied cell and halt
the moment a Banach a posteriori bound proves that the infinite-depth answer cannot
differ from the current one. On synthetic tasks it worked — 100% agreement, 65× fewer
iterations than a 400-step reference. We tried it on a character-level language model
and it certified **less than 1% of tokens**. This paper is the autopsy, and the autopsy
reaches further than the method.

We show, on weight-tied causal transformers trained by unrolling K steps:

1. **Competence is pinned to the training depth.** Accuracy peaks at exactly K
   (K=5: peak 5; K=10: peak 10, three seeds), then *decays monotonically* toward the
   fixed point. The equilibrium is not the model; it is where the model has finished
   forgetting.
2. **The fixed point is nevertheless perfectly path-independent** (AA = 1.000 from
   initializations up to σ=5). The bulk *is* determined by the boundary. It is
   determined, unique, and worse. Path independence is necessary for the certificate
   to mean anything and utterly insufficient for it to be worth anything.
3. **Retargeting the certificate at z_K instead of x\*** — a one-factor change,
   (1 − ρ^(K−n)) — makes the bound up to 13× tighter and, on the good models,
   *still fires zero times*.
4. **The reason is structural, and it generalizes past this method.** A contraction
   certificate can only fire once the state has stopped moving. A state that has
   stopped moving is a state that has stopped computing. Therefore:

> **Certified anytime halting cannot compress useful computation. It can only recover
> compute that was already being wasted.** The savings it reports are a measurement of
> a model's idleness, not a speedup of its work.

The certificate fires on **94% of tokens for our worst model** (K=40, acc 0.282) and
**0% for our best** (K=10, acc 0.429). That is the instrument working correctly. What
it correctly reports, on a model that is not wasting compute, is *zero*.

We propose the certificate be repurposed accordingly: not as an accelerator, but as a
**compute-utilization probe** — the certified fraction of a weight-tied model is a
sound, per-token measurement of how much of its inference does nothing.

---

## 1. Where this came from

The lineage is a rendering trick. In a WebGL raymarcher (*Antti's Brain 2*), a hall of
convex mirrors shows apparently infinite recursive reflections while computing four
bounces. Four is not a budget — it is the correct truncation depth, because each bounce
both dims and shrinks the image, and after four the residual falls below one pixel. The
infinite series converges at 4 *in the only norm a retina can measure*.

Horizon Net was that principle wearing a neural network: iterate a contracting cell,
halt when the bounded distance to the infinite-depth fixed point can no longer flip the
decision. For a contracting map with rate ρ,

$$\|x^* - x_n\| \le \frac{\rho}{1-\rho}\,\|x_n - x_{n-1}\|,$$

which pushed through a readout with Lipschitz constant L gives the halting rule

$$2L\,\frac{\rho}{1-\rho}\,\|x_n - x_{n-1}\| \;<\; \text{(top-1 logit} - \text{top-2 logit)}.$$

Halting is a theorem firing per-sample, not a heuristic. That distinction is the entire
point of the method, and we will hold it to that standard here.

**A substitution happened somewhere in that translation, and it is the bug.** The
raymarcher *truncates a transient* — it never computes the fixed point of the rendering
equation and never wanted to. Horizon Net *converges to a fixed point and proves it can
stop early*. Those are not the same sentence. Everything below follows from the
difference.

## 2. Setup

Weight-tied causal transformer block, DEQ-style input injection, damping α = 0.5, no
Lipschitz constraint anywhere:

$$z_{n+1} = (1-\alpha)z_n + \alpha\,\mathrm{LN}_2\big(h + \mathrm{FF}(h)\big),\quad
h = \mathrm{LN}_1\big(z_n + \mathrm{Attn}(z_n) + u(x)\big)$$

D = 64, 4 heads, context T = 32, char-level tiny Shakespeare, vocab 65. Trained by
unrolling K steps and backpropagating through all of them; 1800 Adam steps, lr 2e-3,
single CPU core. K ∈ {5, 10, 20, 40}; three seeds at K = 10. All measurements on held-out
text, last-position next-character prediction.

## 3. The fixed point is not the model

Sweeping inference depth from 1 to 300 on a model trained at depth K:

| K | peak depth | acc @ K | acc @ fixed point | Δacc | Δnats | AA score | train loss |
|---|---|---|---|---|---|---|---|
| 5 | **5** | 0.425 | 0.206 | **0.220** | 1.869 | 0.841 | 1.95 |
| 10 (s0) | **10** | 0.429 | 0.386 | 0.043 | 0.187 | 1.000 | 1.94 |
| 10 (s1) | **10** | 0.427 | 0.388 | 0.039 | 0.179 | 1.000 | 1.87 |
| 10 (s2) | **10** | 0.429 | 0.386 | 0.043 | 0.195 | 1.000 | 1.94 |
| 20 | 50 | 0.324 | 0.324 | 0.0001 | 0.001 | 1.000 | 2.32 |
| 40 | 36 | 0.282 | 0.282 | 0.0001 | 0.000 | 1.000 | 2.48 |

**Peak depth equals K, exactly, in all four runs where training succeeded** (K=5, and
K=10 across three seeds; premium 0.042 ± 0.002, acc 0.428 ± 0.001). The K=5 model's
equilibrium is catastrophic: loss 3.84, against 4.17 for a uniform distribution over 65
characters. Its fixed point is very nearly noise.

**Transient premium — acc(K) − acc(x\*) — decays monotonically in K:** 0.220 → 0.042 →
0.0001 → 0.0001. As the training depth grows, the model converges (in the ordinary
sense) toward being a genuine equilibrium model, one whose answer *is* its fixed point.

### 3.1 Path independence holds, and does not help

We measured the Asymptotic Alignment score (Anil et al., 2022): fix the boundary data
u(x), start the interior state z₀ from zeros and from Gaussian noise at σ = 0.1, 1.0,
5.0, and iterate each to its fixed point.

**AA = 1.000.** Cosine similarity between fixed points, relative distance ≤ 3×10⁻⁵.
Accuracy at the fixed point: 0.3973, 0.3980, 0.3987, 0.3967 across the four
initializations — agreement to a tenth of a percent.

The boundary determines the bulk. The holographic well-posedness statement holds
*exactly*. And the bulk it determines is uniformly worse than the transient.

Meanwhile the transient is path-*dependent*: at depth 10, accuracy is 0.461 from the
zero init (the one training used) and 0.328 from σ=5 noise, with only 53% of predictions
agreeing between them.

> **The competence lives in the transient and only from the trained initial condition.
> Convergence destroys it.** The attractor is path-independent *because* it is a sink,
> and being a sink means it forgets where you came from — including the initialization
> that carried the model's skill.

This is in tension with the reading of Anil et al. under which path independence enables
models to exploit test-time compute. Our model is maximally path-independent and its
performance *degrades* with depth beyond K. We flag this as an observation, not a
refutation: their tasks (prefix-sum, mazes) are algorithmic with exact answers and
harder OOD instances; language modeling has no such structure, and our AA is our own
operationalization of their score, not their exact metric. **The tension needs someone
to resolve it properly. It is not resolved here.**

## 4. Retargeting the halting condition

If the fixed point is not the model, the certificate is pointed at the wrong object. So
point it at z_K, the iterate training actually optimized. Same machinery, finite sum:

$$\|z_K - z_n\| \;\le\; \sum_{k=n}^{K-1}\|z_{k+1}-z_k\|
\;\le\; d_n\,\frac{\rho\,(1-\rho^{\,K-n})}{1-\rho},
\qquad d_n = \|z_n - z_{n-1}\|.$$

The infinite bound is recovered at K → ∞. The change is exactly one factor,
**(1 − ρ^(K−n))**, which at ρ = 0.96 and K−n = 2 equals 0.078 — a **13× tighter bound,
free**. The certified prediction is now provably the *depth-K* prediction: the model
that was trained.

Halting rule, unchanged otherwise: halt at n when 2·L·bound < margin_n.

## 5. What the retargeted certificate does

| K | acc @ K | **certified fraction** | mean halt | violations vs z_K | compute saved | **oracle could save** |
|---|---|---|---|---|---|---|
| 5 | 0.425 | **0.0%** | — | — | 0% | 58% |
| 10 | 0.429 | **0.0%** | — | — | 0% | 58% |
| 20 | 0.324 | 42.2% | 18.5 / 20 | 0.0% | 3.1% | 80% |
| 40 | 0.282 | **94.4%** | 29.4 / 40 | 0.0% | **25.1%** | 87% |

Where it fires, it is sound: **zero violations** against the exact depth-K target, at
every K. The theorem is correct. The bound is correct. The implementation is correct.

And it is useless exactly where the model is good.

**Plot the certified fraction against model quality and it is a clean monotone
anticorrelation.** 94% on the worst model. 0% on the best.

## 6. Why: the certificate is a waste detector

The mechanism is transparent once stated, and it is not specific to Banach bounds.

A contraction certificate fires when `d_n = ‖z_n − z_{n−1}‖` is small enough that the
bounded residual cannot flip the decision. But `d_n` small **means the state has stopped
moving**. A state that has stopped moving is a state that has stopped computing —
subsequent iterations do nothing.

Therefore the interval the certificate skips, from halt to K, is by construction an
interval in which the network was already idle. **The certificate does not compress
computation. It discovers that computation had already ended.**

This explains every result in the Horizon Net lineage at once:

- **Experiment 1** (MLP, ρ pinned to 0.8, 400-iteration reference): halted at 6.2 for a
  "65× speedup." Those 394 skipped iterations were doing nothing. The reference was
  absurd — the ledger already confessed this — but we can now say *why every impressive
  reference must be absurd*. An impressive saving is a confession of a wasteful baseline.
- **Experiment 2** (majority vote, 150-iteration reference): identical structure.
- **Experiment 3** (LM at K=10): the model computes right up to K and stops. Nothing is
  wasted. The certificate reports **zero savings, correctly.** "No savings available"
  was the true answer all along.
- **K=20, K=40:** the model finished computing by iteration ~29 of 40 and idled the
  rest. The certificate found the idleness and reported 25%. That is a real,
  sound measurement — of a defect.

### 6.1 The proof gap

The oracle — the earliest iteration whose argmax matches the depth-K argmax and never
deviates before K — has mean **4.2 at K = 10**. The answer is settled at iteration 4.
The certificate cannot prove it at 9.

That gap is not slack in the bound. We verified separately that the Banach bound is only
**2.2× loose** against the true distance to the fixed point, and the online contraction
estimate is nearly exact (ρ̂ = 0.976 vs true ρ = 0.959). The bound is tight. **The gap is
between the state having settled and the decision having settled, and they are not the
same event.** The decision settles first, by a factor of two, and no bound on state
motion can see it.

Capturing that headroom requires reasoning in *decision space*, not state space — which
means learned or heuristic halting (ACT, PonderNet), the very family Horizon Net defined
itself against. There is roughly 58% of compute lying on the floor at K=10, and no sound
method we know of can pick it up.

### 6.2 A failed fix, logged

We attempted a directional certificate: near a fixed point the error `e_n = Σ_{k≥n}
(z_{k+1} − z_k)` should align with the dominant slow eigenvector, so the last update
direction `v_n` estimates the error direction and the worst-case constant ‖W‖₂ can be
replaced by the projected `|(w_a − w_b)·v_n|`. This shrinks the coefficient **75×** and
certifies **98.5% of tokens at median depth 9**.

The pre-registered check was cos(e_n, v_n) against a deep reference. It came back
**0.35**, not ≈1. The assumption is false. The rule violates its own guarantee on
**34% of tokens.** It is an excellent heuristic early-exit and a worthless certificate,
and it is reported here because the difference between those two things is the only
thing this repository is about.

## 7. What this means

**Anytime inference, done soundly, is a null concept on a well-specified unrolled
model.** If you train at depth K, the model computes to depth K, and there is nothing to
skip. Every sound saving you can demonstrate is evidence that you chose K too large.

The certificate should therefore change jobs:

> **The certified fraction of a weight-tied model is a sound, per-token measurement of
> its inference-time idleness.** Report it as a diagnostic. A model that certifies 94% of
> tokens at 29/40 iterations is telling you, with a proof, that a quarter of its
> inference does nothing — and that K should be smaller.

That is a real instrument. It has a theorem behind it, zero violations in every
configuration we ran, and it answers a question practitioners actually have.

And it reunites the method with the rendering trick that started it. **The four-bounce
truncation in the raymarcher was correct because the observer has a hard resolution
limit — a pixel.** The proof gap there is *zero*, because "below one pixel" and "below
the decision threshold" are the same event. A network has no pixel. Its decision settles
while its state wanders on, and a bound on the operator cannot certify a threshold that
lives in the observer.

> **The resolution horizon is real. But it lives in the retina, not in the mirror.**
> Horizon Net tried to derive the horizon from the operator's contraction rate. The game
> got it from the eye. That is the missing piece and the specification for whatever comes
> next: a certificate that contracts in *decision space*, not state space.

## 8. Honest ledger

- **The K=20 and K=40 arms are confounded.** Both trained worse (loss 2.32, 2.48 vs
  1.94) — backprop through 20–40 tied steps at lr 2e-3 and 1800 steps is hard. So
  "became a true equilibrium model" and "was undertrained" arrive together, and we
  **cannot** separate them. The claim *"the certificate fires only on bad models"* is
  therefore suggestive, not established. **The control that decides it:** train K=40 to
  convergence (more steps, gradient clipping, or phantom gradients / IFT), and re-measure.
  If a *good* K=40 model still certifies 94%, the tension is an artifact and Horizon Net
  is partly rescued. If a good K=40 model recovers a transient premium and stops
  certifying, the thesis stands. **This experiment has not been run.** It is the next
  thing to do and it can be done in an afternoon.
- **peak = K is 4 runs** (K=5 ×1, K=10 ×3). Not four *values* of K. The claim is
  supported at two training depths, one of them replicated. K=20's peak was 50 and
  K=40's was 36; both are confounded per above.
- **One scale, one architecture, one corpus.** D=64, T=32, char-level Shakespeare, one
  CPU core. Nothing here demonstrates anything about models of practical size. The
  *mechanism* in §6 is architecture-independent by argument; the *numbers* are not.
- **The AA score is our operationalization**, not a re-implementation of Anil et al.
  The tension in §3.1 may dissolve under their exact metric.
- **"Zero violations"** at K=20 and K=40 is over 211 and 472 certified samples. Rule of
  three: violation rate < 1.4% and < 0.6% at 95% confidence. Not zero.
- **Experiment 2's original headline** (100% certified, 0 violations, majority vote)
  was never tested against the *ever-flip* standard used here — disagreement at *any*
  later depth, not merely the final one. It should be re-audited before it is cited.
- **§6 is an argument, not a theorem.** "A state that has stopped moving has stopped
  computing" is intuitively airtight and formally unproven. Someone should either prove
  it or find the counterexample. A network could in principle sit at a near-fixed point
  while a tiny residual drives a decision flip — our own directional experiment shows
  decisions moving while the state barely does. **That is precisely the case that would
  break the argument, and we observed it.** Treat §6 as a conjecture with strong evidence,
  not a result.

## 9. Prior art

- Bai, Kolter, Koltun (2019), *Deep Equilibrium Models* — fixed-point inference,
  heuristic residual stopping tolerances, no margin coupling.
- Anil, Pokle, Liang, Treutlein, Wu, Bai, Kolter, Grosse (2022), *Path Independent
  Equilibrium Models Can Better Exploit Test-Time Computation*
  ([arXiv:2211.09961](https://arxiv.org/abs/2211.09961)) — the AA score; reports that
  low-AA models degrade past training depth. Our high-AA model *also* degrades, which we
  flag as unresolved.
- Graves (2016), *Adaptive Computation Time*; Banino et al. (2021), *PonderNet* — learned
  halting, no bound. §6.1 argues these are the only family that can reach the headroom
  a sound certificate must leave on the floor.
- Banach (1922) — the bound.

We have **not** found the specific claim of §6 stated. *"Not found" is not "doesn't
exist,"* and a proper literature pass on certified early exit and anytime-inference
guarantees is step one before this is asserted anywhere it matters.

## 10. Reproducing

```
horizon_lm_probe.py        the original kill: soft certificate on char-LM, <1% certified
horizon_lm_autopsy.py      true fixed-point error vs bound; bound is 2.2x tight
horizon_directional.py     the falsified fix (cos = 0.35, 34% violations)
horizon_fixedpoint_tax.py  accuracy vs depth to the fixed point
horizon_aa_score.py        path independence: AA = 1.000
horizon_optimum.py         the K-sweep: peak = K, and the retargeted certificate
```

Seeded. CPU. Total wall-clock for the full sweep ≈ 45 min on one core. If your numbers
differ materially, that is a result — open an issue.

---

*Part of the PerceptionLab research program. The result we wanted was a faster network.
The result we got was a proof that the speedup we were chasing is a measure of our own
waste. We are publishing the second one.*

---

## Files in this folder

| file | what it does | the number |
|---|---|---|
| `horizon_lm_probe.py` | soft certificate on a char-level LM — the kill-or-confirm test | **<1% certified** |
| `horizon_lm_autopsy.py` | true fixed-point error vs the bound; oracle halt depth | bound is **2.2× tight**; oracle **8.5**, proof needs **~150** |
| `horizon_directional.py` | a proposed fix, and its pre-registered falsification | 98.5% certified, **34% violations**, cos = 0.35 |
| `horizon_fixedpoint_tax.py` | accuracy vs inference depth, out to the fixed point | **0.438 → 0.391** |
| `horizon_aa_score.py` | path independence: same boundary, four different interiors | **AA = 1.000** |
| `horizon_optimum.py` | the K-sweep + the retargeted finite-K halting condition | **peak = K**; cert fires **94% on the worst model, 0% on the best** |
| `plot_optimum.py` | the four-panel figure | — |

Figures: `horizon_optimum.png` (the main result), `horizon_aa_score.png`,
`horizon_fixedpoint_tax.png`, `horizon_lm_autopsy.png`, `horizon_directional.png`.
Raw numbers: `horizon_optimum.json`.

Requires `shakespeare.txt` (tiny Shakespeare) in this folder.

```bash
pip install torch numpy matplotlib
python horizon_optimum.py     # ~45 min, one CPU core, seeded
python plot_optimum.py
```

← back to [Horizon Net](../README.md)
