# Three-class diabetic retinopathy referral triage with a sensitivity guarantee

From one colour fundus photo, decide **no referral**, **routine referral** or **urgent referral**, with a
stated guarantee for each class: at most 10 % of referable eyes (grades 2-4) sent home, and at least 90 % of
grade 3-4 eyes referred urgently. Machine Learning course project, M.Tech (Data and Computational Sciences),
IIT Jodhpur. Swayam Sahu, Sahil Sihag, Nitesh Kumar.

**Report:** [`report/main.pdf`](report/main.pdf) (CVPR format, 8 pages including references).

| | Result (APTOS, 3,662 photos, out of fold) |
|---|---|
| Plain most-likely-class decision | 12.8 % of referable eyes sent home, 46.3 % of urgent eyes referred urgently |
| **C1, class-wise conformal rule** | **4.2 %** (95 % CI 3.2-5.2) sent home, **91.2 %** (88.3-93.8) referred urgently |
| C2 at a new clinic | both targets restored from about 50 labelled photos (IDRiD) and 300 (Messidor-2) |
| C3 | guarantee unaffected by prevalence; workload forecast from unlabelled photos to about 10 referrals per 1,000 |

Pipeline: frozen ImageNet ResNet-50 features (2,048-d, 224 px) -> logistic regression (C = 0.003) ->
one conformal threshold per class (C1), refitted per clinic (C2), with prevalence-aware workload (C3).

## Repository layout

```
dr-referral-triage/
├── README.md               this file: setup, reproduction, commit history
├── REPRODUCIBILITY.md      reproducibility declaration and checklist
├── requirements.txt        exact package versions used (pip)
├── requirements-min.txt    looser versions also tested
├── environment.yml         conda environment (uses requirements.txt)
├── run_all.sh              every step, in order (bash run_all.sh [first_step])
├── src/                    all Python code (scripts import each other, keep them together)
│   ├── dr_preprocess.py        step 1: crop, quality, duplicates, groups, folds
│   ├── extract_features.py     step 2: ResNet-50 features
│   ├── run_models.py, summarize_e1.py, check_splits.py   step 3: E1 + leakage check
│   ├── e0_conformal.py         step 4: E0
│   ├── kmeans_check.py         step 5
│   ├── c1_referral.py          step 6: C1, B3-B5, C1-q, C1-g
│   ├── e2_new_clinics.py       step 7: C2/E2, B6, B7, C2-PAC
│   ├── c3_prevalence.py        step 8: C3/E3, B8, B9, BBSE
│   ├── error_analysis.py       step 9
│   ├── extract_variants.py, ablations.py   step 10
│   ├── camera_shortcut.py, camera_fixes.py, group_robust.py, group_frontier.py,
│   │   evidence_maps.py, mirror_fix.py, label_efficient.py   step 11: additional work (report section 5)
│   └── export_checkpoint.py    step 12: export / check the model checkpoint
├── scripts/
│   ├── download_weights.sh     fetch + SHA-256-check the ResNet-50 backbone
│   └── compare_results.py      diff a re-run against results/
├── checkpoints/            lr_aptos_ref3.npz (final model, 55 kB); backbone linked, see checkpoints/README.md
├── data/README.md          where to download the three datasets (none are included)
├── results/                reference outputs: summaries, aggregate tables, figures (no per-image files)
├── docs/PIPELINE_DETAILS.md  long per-step documentation (options, outputs, checks)
└── report/                 CVPR LaTeX source and main.pdf
```

## Setup

```bash
git clone https://github.com/Nova1369/dr-referral-triage.git && cd dr-referral-triage
# either
conda env create -f environment.yml && conda activate dr-triage
# or
python3.13 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
bash scripts/download_weights.sh          # 102 MB backbone into checkpoints/, SHA-256 checked
```

Then download the datasets as described in [`data/README.md`](data/README.md) into `datasets/`.
PyTorch is only needed for feature extraction (steps 2, 10 and the evidence maps); CPU is enough.

## Reproduce

```bash
bash run_all.sh            # all steps 1-12, writes data_512/ and results_rerun/
bash run_all.sh 3          # start at step 3 if data_512/ and the features already exist
python scripts/compare_results.py results results_rerun   # largest difference per table
```

Every step caches its work and can be re-run. Seed 42 throughout. Approximate CPU times: preprocessing 10-30 min
(4 cores), features about 7 min (4 cores), E1 about 75 min (2 cores), C1, C3 and the error analysis seconds,
E2 about 1.5 min (4 cores), camera fixes about 20 min (2 cores), the 11-method comparison about 1.5 min plus
10 min for the sweep, C2-IW about 1 min. Total: about 3 hours on a laptop, no GPU. Single steps, e.g. C1 only:

```bash
python src/c1_referral.py --data data_512 --results results_rerun/e1 --out results_rerun/c1
```

To apply the final model without retraining, see `checkpoints/README.md`.

Report number -> file: every table in the report is printed in `results/<experiment>/summary.md`
(mapping in `results/README.md`).

## Commit history

Run `git log --date=iso` for the full history.

## Licence and data

Code: MIT (`LICENSE`). Datasets are not redistributed; Messidor-2 images: Kindly provided by the Messidor
program partners (see https://www.adcis.net/en/third-party/messidor/). Messidor-2 grades: Krause et al. [16].
