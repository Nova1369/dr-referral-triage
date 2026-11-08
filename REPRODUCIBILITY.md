# Reproducibility

## Declaration

We declare that every number, table and figure in the report (`report/main.pdf`) was produced by the code in
this repository, at the commits listed in the README and in the report's appendix, from the three public
datasets described in `data/README.md`, without manual editing of any result. Running `bash run_all.sh` in the
environment of `requirements.txt` regenerates all of them. Fixed seeds (42) make every split, draw and model
deterministic on a given machine; across library versions the results agree to the tolerances measured below.
Nothing was tuned on the test photos: model settings are chosen in inner folds, conformal thresholds on
calibration photos only, and new clinics are used only for calibration draws and testing.

Swayam Sahu, Sahil Sihag, Nitesh Kumar - IIT Jodhpur, November 2026.

## Checklist

| Item | Where |
|---|---|
| Code, in run order | `src/`, `run_all.sh` (steps 1-12) |
| Exact environment | `requirements.txt` (pip pins), `environment.yml` (conda), `requirements-min.txt` (looser, also tested) |
| Hardware | CPU only; Linux x86-64 (main run) and Linux ARM VM on a Mac (re-run). No GPU needed |
| Data | public; download links and expected counts in `data/README.md`; not redistributed (licences) |
| Pretrained weights | timm ResNet-50, URL + SHA-256 in `checkpoints/README.md`, `scripts/download_weights.sh` checks the hash |
| Trained model | `checkpoints/lr_aptos_ref3.npz` (55 kB) + `src/export_checkpoint.py check` |
| Randomness | seed 42 everywhere; folds stratified and deterministic (same on every machine); bootstrap with 1,000 group resamples shared by all models |
| Splits / leakage | duplicates and look-alike photos grouped; groups never cross train / calibration / test; `src/check_splits.py` rebuilds every split from the saved fold files and confirms this (run in step 3) |
| Reference outputs | `results/` (aggregate only); compare a re-run with `scripts/compare_results.py` |
| Hyperparameters | chosen in 5 inner folds of each of 10 outer folds; per-fold choices in `results/e1/best_params.csv`, full configs in `results/e1/run_config*.json` |
| Uncertainty | 95 % group-bootstrap CIs, Wilson intervals for subgroups, exact McNemar tests, 200-1,000 draws for C2/C2-IW |
| Commit history | README "Commit history" and the commit table in the report (hashes and timestamps) |

## Second-machine re-run (4 Nov 2026)

The results in `results/` were made with Python 3.13 (scikit-learn 1.9, numpy 2.5, pandas 3.0) on Linux
x86-64. Features were extracted with torch 2.7.1 (CPU) on Python 3.10. Six scripts were then re-run on a
second machine (Linux ARM; Python 3.10, scikit-learn 1.7, numpy 2.2, pandas 2.3) from the same features:

| Script | Result on the second machine |
|---|---|
| `c1_referral.py` | identical outputs (every decision and metric) |
| `c3_prevalence.py` | identical tables |
| `error_analysis.py` | identical tables |
| `e2_new_clinics.py` | LR probabilities differ by at most 0.005 (solver tolerance); tables within 0.1 percentage points |
| `group_robust.py` | H-camera results and all McNemar tests identical; other rates within 1.1 points (one cell: one eye of 37) |
| `label_efficient.py` | identical outputs |

The checkpoint (`export_checkpoint.py check`) reproduces the refitted model's probabilities to 6e-7.

## Known sources of variation

* Logistic-regression solver tolerance across scikit-learn versions (probabilities within 0.005).
* Isotonic-calibrated probabilities can move up to 0.07 for the same reason (step functions); the conformal
  rules use raw probabilities, which is one reason why.
* Feature extraction on another CPU/BLAS or torch version changes features in the last decimals; the
  downstream tables are expected to agree to within the bootstrap intervals, not bit for bit.
* `group_frontier.py` (10 min) was not re-run on the second machine; its default settings reproduce
  `group_robust.py`'s numbers exactly.
