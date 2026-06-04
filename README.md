# Surgical Skill Assessment — Formal Verification

A faithful reproduction of **Ismail Fawaz et al., "Accurate and interpretable
evaluation of surgical skills from kinematic data using fully convolutional
neural networks"** (IJCARS 2019, DOI 10.1007/s11548-019-02039-4), extended with
**formal robustness verification** of the OSATS regression model using two independent
verifiers: **n2v** (Star-set reachability) and **alpha-beta-CROWN** (GPU-accelerated
bound propagation).

Target venue: **FORMATS / ARCH / SAIV workshop at CAV**.

---

## Overview

The base project trains the paper's **SurgicalFCN** on JIGSAWS Suturing kinematics
and reproduces the paper's classification and regression results under both
cross-validation schemes (LOSO and LOUO). The stretch goal formally certifies four
robustness properties of the regression model across all five LOSO folds, using
held-out trials as anchors so every certificate is a generalization claim rather than
an in-sample one.

---

## Repository structure

```
.
├── config.py                        # all paths and constants — single source of truth
├── model.py                         # SurgicalFCN (paper-faithful architecture)
├── model_flat.py                    # SurgicalFCNFlat (block-diagonal reformulation, verifier-friendly)
├── model_flat_regression.py         # regression variant of the flat model
├── dataset.py                       # load_data / load_osats_data / LOSO / LOUO splits
├── train.py                         # LOSO classification + regression training
├── evaluate.py                      # evaluation helpers and results formatter
├── export_onnx.py                   # export classification model to ONNX
├── export_onnx_regression.py        # export regression fold models to ONNX (T-aware)
├── generate_property.py             # VNN-LIB property for classification model
├── generate_property_regression.py  # VNN-LIB properties + verification pipeline
├── inspect_osats_prediction.py      # quick per-trial prediction inspection
├── verify.sh                        # unified verification entry point (all modes)
├── setup_abcrown.sh                 # one-time alpha-beta-CROWN install
├── abcrown_config.yaml              # alpha-beta-CROWN verification strategy
├── requirements.txt                 # project venv dependencies
├── data/                            # JIGSAWS kinematics (not in repo)
├── models/                          # trained checkpoints (not in repo)
├── onnx/                            # exported ONNX models (not in repo)
├── properties/                      # generated VNN-LIB property files (not in repo)
├── results/                         # all verification and training results
└── n2v/                             # vendored n2v verifier (not in repo — see Setup)
```

---

## Requirements

### Hardware
- **GPU strongly recommended** for alpha-beta-CROWN (tested on RTX 4050 Laptop, 6 GB VRAM)
- CPU-only works for n2v; alpha-beta-CROWN will fall back to CPU but is significantly slower

### Software
- Python 3.12
- PyTorch 2.12, NumPy 2.4, SciPy 1.17, ONNX 1.21 (see `requirements.txt`)
- **For alpha-beta-CROWN only:** Python 3.11 (`sudo apt install python3.11 python3.11-venv`)
- NVIDIA drivers with CUDA 12.8+ support

---

## Setup

### 1. Clone and install project dependencies

```bash
git clone <this-repo>
cd surgical-skill-assessment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Install n2v (Star-set verifier)

```bash
git clone https://github.com/jax-ml/n2v.git n2v
pip install -e n2v/
```

### 3. Get the JIGSAWS data

Download the **Suturing** task kinematics from the
[JIGSAWS dataset](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/)
and place them at:

```
data/
└── Suturing/
    └── kinematics/
        └── AllGestures/
            ├── Suturing_B001.txt
            ├── Suturing_B002.txt
            ...
```

### 4. (Optional) Install alpha-beta-CROWN for GPU-accelerated verification

```bash
sudo apt install python3.11 python3.11-venv   # deadsnakes PPA if not found
bash setup_abcrown.sh
```

This clones alpha-beta-CROWN into the repo root, creates an isolated Python 3.11 venv
at `envs/abcrown/`, and installs PyTorch 2.8 + CUDA 12.8 into it. The project venv is
not affected. Takes ~5 minutes on a good connection (~2 GB download).

---

## Reproducing the results

### Step 1 — Train

```bash
# Classification (LOSO + LOUO, ~13 models × 1000 epochs)
python3 train.py

# Regression (LOSO, 5 fold models × 1000 epochs each)
python3 train.py --regression
```

Training results are written to `results/results.txt`. Fold checkpoints are saved to
`models/best_model_regression_fold{1-5}.pth`.

### Step 2 — Verify

All verification goes through the unified entry point:

```bash
bash verify.sh [--T N] [--method exact|approx] [--verifier n2v|abcrown] [--no-search]
```

| Flag | Values | Default | Notes |
|---|---|---|---|
| `--T N` | any integer | 10 | timestep window length |
| `--method` | `exact`, `approx` | `exact` | n2v only; `approx` skips exact-star escalation |
| `--verifier` | `n2v`, `abcrown` | `n2v` | requires `setup_abcrown.sh` for `abcrown` |
| `--no-search` | flag | off | skip certified-ε binary search; emit verdicts only |

Results land in `results/regression_results[_T{N}][_approx][_abcrown][_verdicts].txt`.
No combination of flags overwrites another run's output.

**Recommended runs in order:**

```bash
# Baseline: n2v exact-star, T=10 (the primary result)
bash verify.sh

# Cross-check: abcrown verdicts at T=10 — should match baseline exactly
bash verify.sh --verifier abcrown --no-search

# Higher T with abcrown
bash verify.sh --verifier abcrown --T 70 --no-search
```

---

## Architecture

**SurgicalFCN** — three-stage grouped convolutional network followed by Global
Average Pooling (GAP):

```
Input: (batch, 76, T)  — 76 kinematic channels, T timesteps

Layer 1:  20 × Conv1d(sub-cluster → 8 filters, k=3)    [20 sub-clusters of 4 channels each]
          → concat → (batch, 160, T)
Layer 2:  4  × Conv1d(40 → 16 filters, k=3)            [4 manipulator groups]
          → concat → (batch, 64, T)
Layer 3:  Conv1d(64 → 32 filters, k=3)                 [shared]
          → (batch, 32, T)
GAP:      mean over T dimension → (batch, 32)
Head:     Linear(32 → num_classes)                      [6 outputs for regression, 3 for classification]
```

GAP makes the model **input-length agnostic** — full variable-length trials during
training, fixed T-timestep windows during verification. This is why both tasks use the
same architecture.

**SurgicalFCNFlat** — mathematically equivalent reformulation using standard dense
`Conv1d` / `Linear` layers with block-diagonal weights. Required for verification
because n2v cannot handle the `Slice/Concat/ReduceMean` ops that the grouped
architecture produces in ONNX.

---

## Formal verification

### What we certify

All properties target **Y₄** (Overall Performance), the most clinically meaningful
OSATS sub-score. `X_0 … X_{76T-1}` are the flattened input (channel-first,
76 channels × T timesteps). Each fold model is verified on trials it **never trained
on** (held-out by LOSO fold index).

| # | Property | Anchor | Assertion (violation) | `unsat` means |
|---|---|---|---|---|
| 1 | Noise robustness | Expert (D_fold) | Y₄ deviates >δ from clean prediction | model is stable under da Vinci encoder noise |
| 2 | Monotonicity | Novice (B_fold) | Y₄ ≥ L (expert floor) | novice score provably below expert floor |
| 3 | Segmentation | Expert (D_fold) | Y₄ deviates >δ (wider ε at boundaries) | robustness holds at trial segment boundaries |
| 4 | Range floor | Expert (D_fold) | Y₄ ≤ L | expert score provably above floor L |

Properties 2 and 4 share the same floor L and compose into a **skill-ordering
certificate**: `novice ≤ L ≤ expert` — a formal guarantee that the model ranks expert
above novice over the entire ε-ball.

### VNN-LIB format

Properties are written as standard VNN-LIB files (`properties/`). The two-sided
disjunction (properties 1 and 3) uses **bare comparisons** for n2v and **strict DNF
wrapping** (`(and ...)`) for alpha-beta-CROWN, because their parsers differ:

```
n2v:      (assert (or (<= Y_4 a) (>= Y_4 b)))
abcrown:  (assert (or (and (<= Y_4 a)) (and (>= Y_4 b))))
```

### Certified-ε binary search

For properties 1 and 4, `verify.sh` runs a 12-iteration binary search over ε ∈
[0, 0.1] to find the largest perturbation radius for which the property still holds.
The physical noise bound is ε = 0.001; the certified values are typically 10–45×
larger. Pass `--no-search` to skip this and emit only the fixed-ε verdicts (much
faster, especially for alpha-beta-CROWN).

---

## Results

### Classification (LOSO — paper target: 100%)

| Fold | Trials | Accuracy |
|---|---|---|
| 1 | 8 | 75.0% |
| 2 | 7 | 100.0% |
| 3 | 8 | 100.0% |
| 4 | 8 | 100.0% |
| 5 | 8 | 100.0% |
| **Mean** | 39 | **95.0%** |

LOUO mean: 34.4% (paper reports significant variation across subjects; see
`results/results.txt` for per-subject breakdown).

### Formal verification — n2v exact-star, T=10 (primary result)

`unsat` = property provably holds over the entire ε-ball. `sat` where `novice≥L`
(folds 1, 3, 5) is the *correct* verdict — those folds' novice prediction at the
first 10 timesteps already exceeds the expert floor, so the ordering property is
genuinely violated in that window. Folds 2 and 4 certify all four properties.

```
Fold | noise   | mono    | seg     | range   | eps[noise]  | eps[range]
----------------------------------------------------------------------
1    | unsat   | sat     | unsat   | unsat   |   0.009937  |   0.025854
2    | unsat   | unsat   | unsat   | unsat   |   0.011084  |   0.018555
3    | unsat   | sat     | unsat   | unsat   |   0.009448  |   0.008008
4    | unsat   | unsat   | unsat   | unsat   |   0.013574  |   0.045068
5    | unsat   | sat     | unsat   | unsat   |   0.014624  |   0.015454

certified eps[noise]: mean=0.011733  (11.7× physical bound)
certified eps[range]: mean=0.022588  (22.6× physical bound)
```

### Cross-verification — alpha-beta-CROWN, T=10 and T=70

All 20 verdicts (5 folds × 4 properties) match n2v exactly at T=10. At T=70
(where n2v approx-star returns all `unknown`), alpha-beta-CROWN still certifies
noise, segmentation, and range as `unsat` across all folds:

```
T=70 verdicts (abcrown):
Fold | noise   | mono    | seg     | range
-------------------------------------------
1    | unsat   | sat     | unsat   | unsat
2    | unsat   | unsat   | unsat   | unsat
3    | unsat   | sat     | unsat   | unsat
4    | unsat   | sat     | unsat   | unsat
5    | unsat   | sat     | unsat   | unsat
```

Timing: a single abcrown verification at T=20 takes ~18 s, at T=100 ~36 s — all
resolved at "initial CROWN" without branch-and-bound, confirming that abcrown's
bound propagation scales sub-linearly with T for this network.

---

## Key design decisions

**Why SurgicalFCNFlat?** n2v cannot handle the `Slice/Concat/ReduceMean` ONNX ops
from the grouped architecture. `SurgicalFCNFlat` reformulates the same computation
using only standard dense layers (block-diagonal weights), which n2v and abcrown both
support. Equivalence is verified on export (`max |trained(x) - flat(x)| < 1e-6`).

**Why LOSO for verification?** The paper uses LOSO for both tasks. Verifying each
fold on held-out anchors means every certificate is a *generalization* claim — not
in-sample fit. Anchors `D{fold}` (expert) and `B{fold}` (novice) are the trials
that fold model never saw during training.

**Why two verifiers?** n2v exact-star is complete but memory-limited (OOM at T≳20).
alpha-beta-CROWN uses tighter bound propagation (α-CROWN) with GPU acceleration;
it certifies T=70+ properties that n2v cannot reach. Identical verdicts at T=10
from two independent tools strengthen the paper's claims.

**Why an isolated venv for abcrown?** alpha-beta-CROWN requires Python 3.11 /
PyTorch 2.8, conflicting with the project's Python 3.12 / PyTorch 2.12. A separate
venv at `envs/abcrown/` isolates these dependencies. No activation switching is
needed — `_run_abcrown()` calls `envs/abcrown/bin/python` by full path.

---

## File naming convention

Result and property files are suffixed to prevent collisions:

```
regression_results[_T{N}][_approx][_abcrown][_verdicts].txt

regression_results.txt                    → T=10, n2v exact (primary)
regression_results_T20_approx.txt         → T=20, n2v approx-only
regression_results_abcrown_verdicts.txt   → T=10, abcrown, no search
regression_results_T70_abcrown_verdicts.txt → T=70, abcrown, no search
```

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{ismailfawaz2019surgical,
  title={Accurate and interpretable evaluation of surgical skills from kinematic
         data using fully convolutional neural networks},
  author={Ismail Fawaz, Hassan and others},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  year={2019},
  doi={10.1007/s11548-019-02039-4}
}
```
