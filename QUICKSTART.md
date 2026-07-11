# Quick Start: Running Horizon Net Experiments

## One-liner setup

```bash
git clone https://github.com/anttiluode/HorizonNet
cd HorizonNet
pip install -r requirements.txt
```

## Experiment 1: Hard Certificate (10 minutes, CPU only)

```bash
python horizon_net.py
```

**What it does:**
- Trains a weight-tied neural network with spectral norm pinned at ρ=0.8
- Tests on MNIST (2000 test samples)
- Measures how many iterations it takes before a Banach bound proves the answer is locked in
- Result: mean 6.2 iterations vs 400-iteration reference, 100.0% agreement

**Output files:**
- `horizon_net.json` — raw numbers
- Plots (when matplotlib is configured)

**What to read after:**
- `horizon_net_results.md` — full ledger of what worked and what the limits are

---

## Experiment 2: Soft Certificate on Transformers (3–5 minutes, CPU only)

```bash
python horizon_transformer_probe.py
```

**What it does:**
- Builds a weight-tied transformer block (NO spectral norm constraint)
- Trains it on majority-vote classification over binary sequences (easier task, cleaner difficulty dial)
- During inference, estimates the contraction rate ρ_eff *on the fly* from recent update norms
- Uses that soft contraction estimate to form a Banach bound and halt when it's safe
- Tests on 1000 sequences

**Output files:**
- `horizon_transformer_probe.json` — raw numbers
- `horizon_transformer_probe.png` — three panels showing the certificate behavior
- Plots and logs printed to stdout

**What to read after:**
- `horizon_transformer_probe_results.md` — honest assessment of what the soft certificate means and where it can break

---

## Understanding the Results

Both experiments should run to completion and print JSON results. Key numbers to look for:

**Experiment 1 (hard):**
- `mean_halt_rho_0_8`: should be close to 6.2 iterations
- Accuracy should be 1.0 (100% on MNIST)
- No prediction violations

**Experiment 2 (soft):**
- Left panel: update norms should decay geometrically
- Center panel: histogram should show most halts between 5–15 iterations, with 0 violations out of 1000
- Right panel: harder samples (smaller vote margin) should require more iterations

If your numbers differ materially, that's a result — please file an issue with the numbers and your system specs.

---

## Customization Flags (Edit Code)

### For Experiment 1
- Line ~25: change `d=128` to adjust cell width
- Line ~25: change `TRAIN_ITERS=12` to train for more unrolled steps
- Line ~87: change `RHO_SETTINGS` to test different contraction rates

### For Experiment 2
- Line ~22: change `SEQ=25` to vary sequence length
- Line ~36: change `H=4` to adjust attention heads
- Line ~14: change contraction rate in the damping formula

---

## Prerequisites

- **CPU:** Both experiments run on CPU in 3–15 minutes depending on hardware
- **GPU:** Not required, but torch will use CUDA if available
- **Python 3.8+**
- No CUDA/cuDNN needed

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'torch'"**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**"MNIST download fails"** (experiment 1 only)
The script tries to download MNIST to `/tmp`. If that fails, edit line ~20 in `horizon_net.py` and change the `root` parameter to a directory you control.

**Results differ from the reported numbers**
- Differences < 5% are noise
- Differences > 10%: check your PyTorch version and CPU count (multithreading can introduce variance in neural network operations)

---

## What Each File Does

| File | Purpose |
|------|---------|
| `README.md` | Overview, the math, and the claims |
| `horizon_net.py` | Experiment 1: hard certificate on MNIST |
| `horizon_net_results.md` | Full results + ledger for experiment 1 |
| `horizon_transformer_probe.py` | Experiment 2: soft certificate on transformers |
| `horizon_transformer_probe_results.md` | Full results + ledger for experiment 2 |
| `requirements.txt` | Python dependencies |
| `LICENSE` | MIT |

---

## Next Steps After Running

1. **Read the honest ledgers** in the `*_results.md` files — these list exactly what is known, what is guessed, and what is still a toy.
2. **Check the open questions** in the README — they are ordered by priority (literature pass first, small-LM probe second).
3. **Reproduce with different hyperparameters** — the codebase is short and meant to be modified for exploration.

---

*The motto: do not hype, do not lie, just show.*
