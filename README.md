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

Every commit on `main`, oldest first (dates in IST). The report's commit table lists the same commits up to
the code freeze (no. 37). This table is added by the last commit, so that one is not listed;
`git log --date=iso` shows it.

| # | Commit | Date (IST) | Message |
|---|---|---|---|
| 1 | `eb7ccc49ce4d84e7e33e8c501b3ee47eb2e9ca58` | 2026-10-25 00:39 | Preprocessing: retina crop, quality checks, duplicate groups, folds |
| 2 | `9ed58ff6199a8accbd3527089305b35601192937` | 2026-10-25 02:29 | Frozen ResNet-50 feature extraction |
| 3 | `e3a47c07abe73cd9113afc5d86790c04adef8b36` | 2026-10-25 09:49 | Add requirements.txt |
| 4 | `85607c91ba6dd4500205142a2ce7e07dc6ba08cb` | 2026-10-25 10:25 | E1: nested CV for the course models and baselines |
| 5 | `5b03e5c542ebab8452bd79b3cd288c8425c9a2cb` | 2026-10-25 10:28 | E0: standard conformal prediction at IDRiD and Messidor-2 |
| 6 | `51e23b0020069511b52223a8f24df4785490ce56` | 2026-10-25 11:29 | E1 summary tables; E0 results |
| 7 | `e711fbd545b78bdfb46adc0f14d5c6b9e1b289e8` | 2026-10-25 11:30 | E1 results |
| 8 | `7b233b7f2d713e8266232ae8ac015e029fa2a1dc` | 2026-10-25 11:32 | k-means check |
| 9 | `30ae18382435c138cca571b52567dcf44ff4782d` | 2026-10-25 11:58 | Camera shortcut study |
| 10 | `57e9d73397e05852ae6620b49ed2d082d9a20403` | 2026-10-25 12:14 | run_all.sh: every step in order |
| 11 | `5a2082b6dd32c34f3191abbfd1008e35161395be` | 2026-10-25 12:16 | Pipeline notes |
| 12 | `328064940c44ac1e9fffb64881a947c3d904dc9c` | 2026-11-03 21:31 | C1 class-wise referral rule; error analysis |
| 13 | `91a9b7c2cd2c9836409876d0eddfdcce28fc9503` | 2026-11-04 01:27 | Ablations and robustness checks |
| 14 | `597b1fffd1b4b7f7e553b32d34bf4032bd279673` | 2026-11-04 01:30 | C1 results |
| 15 | `b60cfed70be5be110dbe6e98d69119402fdb1f62` | 2026-11-04 01:36 | C3: prevalence shift and workload forecast |
| 16 | `9821a6f00300ac7e6b9c08c1a069a7b7bc490c66` | 2026-11-04 02:04 | Ablation and robustness results |
| 17 | `437ef23817c6003b8375f0bb8e6790ee0d96a953` | 2026-11-04 02:18 | C2/E2: recalibration at new clinics, C2-PAC |
| 18 | `98eb9b7b442a7b6a2178f332edf574f1a6a0d2f7` | 2026-11-04 02:19 | Update requirements |
| 19 | `e8abf8148e5646cf35223532b1cbb6c57595ad2a` | 2026-11-04 10:26 | Camera fixes tuned under the E1 protocol |
| 20 | `bd70ea68ddb259a2722a99627f9330b62610dcf5` | 2026-11-04 10:29 | E2 results |
| 21 | `9f82faac8071385439726a0c1da7c3cf6ad35566` | 2026-11-04 16:43 | Exact evidence maps, deletion and mirror tests |
| 22 | `6ea7da0aa936bab4917eb851f5a195672b157a1a` | 2026-11-04 18:35 | Feature variants, including mirrored photos |
| 23 | `fb76710b1ffe8e1eb78c89c0bab23175490f2447` | 2026-11-04 20:33 | Mirror-averaged features |
| 24 | `92ed8b00efdf6a899d4b1a9d742c01718b90deb4` | 2026-11-05 01:24 | Camera-group benchmark against strong baselines |
| 25 | `e5f8ea8728984556390a7f4cea88ac626ae26c62` | 2026-11-05 01:33 | Camera-group benchmark results |
| 26 | `200fe582cbf1685873be4131095e1ffdedc3c782` | 2026-11-05 01:34 | C2-IW: label-efficient recalibration |
| 27 | `f951cadde02dc51834256c364108b1e8eaddbc9e` | 2026-11-05 01:56 | Baseline sweeps at matched fairness |
| 28 | `4d1a577322acaedfe73740050970935f0204ccb4` | 2026-11-05 02:01 | C2-IW results |
| 29 | `08984095a720febebf61e347bb8175ec4e1ab67d` | 2026-11-05 02:01 | Update run_all.sh and notes for the new studies |
| 30 | `c4f93ae58603ec8ead5c3c4039c2044ea8323d15` | 2026-11-08 17:09 | Restructure as a repository: src/, docs/, results/ |
| 31 | `45ff75397bc442bb6f500b4d8de4371150eaf455` | 2026-11-08 17:10 | Export the final model as a small checkpoint |
| 32 | `3b44f9fb509a20d2ae5b689eedd6ad595817e27d` | 2026-11-08 17:10 | run_all.sh for the repository layout |
| 33 | `520c2bd6bd8d9f5626ff073d3f8a45fcc6a2d5a3` | 2026-11-08 17:10 | Pin the environment; weight download and result comparison scripts |
| 34 | `8ee80c8ef1f69619baac3032bae459e09ba67728` | 2026-11-08 17:11 | Notes for data, checkpoints and results; MIT licence |
| 35 | `01d5099164576cab238689773ad9fea30a4a6dc2` | 2026-11-08 17:11 | README: setup, reproduction, layout |
| 36 | `9abb423e7894cadf839bc7bbc1e91d56ece353d3` | 2026-11-08 17:12 | Reproducibility statement |
| 37 | `034205e71c87a433283d12f10fcfc5dc2c730ed0` | 2026-11-08 17:12 | Leakage check for the E1 splits |
| 38 | `84df781cde996c0f55f765b0dbb0bb4a34086a2f` | 2026-11-08 17:17 | CVPR report |

## Licence and data

Code: MIT (`LICENSE`). Datasets are not redistributed; Messidor-2 images: Kindly provided by the Messidor
program partners (see https://www.adcis.net/en/third-party/messidor/). Messidor-2 grades: Krause et al. [16].
