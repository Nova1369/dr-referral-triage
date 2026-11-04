# Additional work: camera-group guarantees against strong baselines

Logistic regression on standardised ResNet-50 features (C = 0.003 for every trained method; DFR 0.03), refitted in each of E1's 10 outer folds; C1 (α = 0.1) on each fold's calibration photos. Camera groups for training are recomputed from each fold's training photos only (225 training photos, summed over the folds, fall in a different group than in the fixed analysis grouping). LEACE rows: camera_fixes.py, C tuned in inner folds. In %.

## 1. Every camera group against the 10 % target

| Method | Referable sent home: H (55 eyes) | M (717) | S (715) | Worst group | Urgent referred urgently: worst group | C1 overall: referable sent home / urgent referred urgently | Healthy sent home | Referred per 1,000 | AUROC referable | Same-camera AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| Plain training (ERM) | 43.6 (31–57) | 2.9 | 1.5 | 43.6 | 91.8 | 3.8 / 93.0 | 87.6 | 451 | 0.974 | 0.919 |
| Camera-decorrelated weights (ours) | 9.1 (4–20) | 4.3 | 4.8 | 9.1 | 91.1 | 4.7 / 92.2 | 84.3 | 437 | 0.965 | 0.923 |
| Group-balanced weights | 5.5 (2–15) | 2.8 | 2.5 | 5.5 | 91.8 | 2.8 / 92.8 | 76.7 | 457 | 0.965 | 0.921 |
| DFR: group-balanced subsampling | 0.0 (0–7) | 3.2 | 3.9 | 3.9 | 91.2 | 3.4 / 91.6 | 26.4 | 466 | 0.877 | 0.890 |
| Group DRO | 3.6 (1–12) | 2.5 | 1.4 | 3.6 | 91.4 | 2.0 / 93.4 | 70.7 | 485 | 0.963 | 0.919 |
| JTT, errors x5 | 21.8 (13–34) | 2.6 | 1.4 | 21.8 | 86.7 | 2.8 / 92.2 | 81.4 | 454 | 0.971 | 0.913 |
| JTT, errors x20 | 7.3 (3–17) | 0.8 | 1.4 | 7.3 | 86.7 | 1.3 / 92.4 | 63.5 | 469 | 0.956 | 0.893 |
| Per-group thresholds (fold calibration) | 0.0 (0–7) | 2.2 | 2.0 | 2.2 | 93.6 | 2.0 / 93.9 | 15.8 | 812 | 0.974 | 0.919 |
| Per-group thresholds (cross-fitted) | 7.3 (3–17) | 2.9 | 2.7 | 7.3 | 90.2 | 3.0 / 90.4 | 75.8 | 415 | 0.974 | 0.919 |
| LEACE fitted on healthy eyes | 52.7 (40–65) | 3.9 | 1.5 | 52.7 | 90.7 | 4.6 / 91.4 | 88.1 | 448 | 0.972 | 0.920 |
| LEACE within each grade | 36.4 (25–50) | 6.3 | 3.4 | 36.4 | 91.8 | 6.0 / 92.2 | 85.9 | 439 | 0.963 | 0.901 |

H-camera referable eyes: exact McNemar test on sent home or not, against plain training and against the camera-decorrelated weights (eyes the method saves / eyes it newly sends home):

| Method | Against ERM: saved / lost, p | Against ours: saved / lost, p |
|---|---|---|
| Plain training (ERM) | – | 0 / 19, p = 3.8e-06 |
| Camera-decorrelated weights (ours) | 19 / 0, p = 3.8e-06 | – |
| Group-balanced weights | 21 / 0, p = 9.5e-07 | 2 / 0, p = 0.5 |
| DFR: group-balanced subsampling | 24 / 0, p = 1.2e-07 | 5 / 0, p = 0.062 |
| Group DRO | 22 / 0, p = 4.8e-07 | 3 / 0, p = 0.25 |
| JTT, errors x5 | 13 / 1, p = 0.0018 | 0 / 7, p = 0.016 |
| JTT, errors x20 | 20 / 0, p = 1.9e-06 | 2 / 1, p = 1 |
| Per-group thresholds (fold calibration) | 24 / 0, p = 1.2e-07 | 5 / 0, p = 0.062 |
| Per-group thresholds (cross-fitted) | 20 / 0, p = 1.9e-06 | 1 / 0, p = 1 |
| LEACE fitted on healthy eyes | 0 / 5, p = 0.062 | 0 / 24, p = 1.2e-07 |
| LEACE within each grade | 5 / 1, p = 0.22 | 1 / 16, p = 0.00027 |

## 2. Healthy eyes sent home per camera group (the cost)

| Method | H | M | S |
|---|---|---|---|
| Plain training (ERM) | 98.5 | 63.2 | 10.8 |
| Camera-decorrelated weights (ours) | 92.4 | 65.7 | 35.1 |
| Group-balanced weights | 85.7 | 56.1 | 18.9 |
| DFR: group-balanced subsampling | 20.4 | 42.0 | 37.8 |
| Group DRO | 77.1 | 57.1 | 10.8 |
| JTT, errors x5 | 94.5 | 50.9 | 10.8 |
| JTT, errors x20 | 76.9 | 30.8 | 13.5 |
| Per-group thresholds (fold calibration) | 0.0 | 58.2 | 13.5 |
| Per-group thresholds (cross-fitted) | 82.9 | 60.9 | 13.5 |
| LEACE fitted on healthy eyes | 99.0 | 63.7 | 13.5 |
| LEACE within each grade | 96.8 | 60.6 | 27.0 |

Every healthy APTOS eye (2175): exact McNemar test on sent home or not, each method against the camera-decorrelated weights (eyes only the method sends home / eyes only ours sends home):

| Method | Healthy sent home | Only the method / only ours | p |
|---|---|---|---|
| Plain training (ERM) | 87.6 | 108 / 36 | 1.5e-09 |
| Group-balanced weights | 76.7 | 0 / 165 | 4.3e-50 |
| DFR: group-balanced subsampling | 26.4 | 10 / 1268 | 0 |
| Group DRO | 70.7 | 3 / 299 | 1.1e-84 |
| JTT, errors x5 | 81.4 | 56 / 118 | 3e-06 |
| JTT, errors x20 | 63.5 | 14 / 465 | 4.2e-118 |
| Per-group thresholds (fold calibration) | 15.8 | 4 / 1494 | 0 |
| Per-group thresholds (cross-fitted) | 75.8 | 10 / 194 | 2.3e-45 |
| LEACE fitted on healthy eyes | 88.1 | 121 / 38 | 2.7e-11 |
| LEACE within each grade | 85.9 | 113 / 77 | 0.011 |

## 3. New clinics

Each trained method refitted on all of APTOS. B6: APTOS thresholds (from the method's out-of-fold APTOS predictions) used unchanged; C2: thresholds refitted on k labelled local photos (200 draws, the same for every method). Referable sent home / urgent referred urgently / healthy sent home; last column: share of draws meeting both targets.

| Method | Clinic | AUROC referable | B6 | C2, smaller k | C2, larger k |
|---|---|---|---|---|---|
| Plain training (ERM) | IDRiD | 0.929 | 9.9 / 89.0 / 75.6 | k = 50: 2.5 / 92.8 / 38.5 (both targets 72 %) | k = 100: 3.8 / 91.9 / 46.6 (both targets 70 %) |
| Plain training (ERM) | Messidor-2 | 0.764 | 79.6 / 35.5 / 98.0 | k = 200: 5.2 / 94.0 / 20.9 (both targets 70 %) | k = 300: 6.4 / 92.0 / 26.0 (both targets 62 %) |
| Camera-decorrelated weights (ours) | IDRiD | 0.929 | 20.1 / 88.4 / 91.2 | k = 50: 3.5 / 92.6 / 43.8 (both targets 72 %) | k = 100: 4.9 / 92.0 / 53.7 (both targets 70 %) |
| Camera-decorrelated weights (ours) | Messidor-2 | 0.750 | 56.7 / 41.8 / 87.1 | k = 200: 5.3 / 93.9 / 20.0 (both targets 72 %) | k = 300: 6.6 / 91.7 / 25.2 (both targets 57 %) |
| Group-balanced weights | IDRiD | 0.922 | 12.1 / 89.0 / 78.2 | k = 50: 2.5 / 92.8 / 36.2 (both targets 76 %) | k = 100: 3.8 / 91.7 / 46.9 (both targets 68 %) |
| Group-balanced weights | Messidor-2 | 0.746 | 46.8 / 42.7 / 78.9 | k = 200: 4.5 / 93.4 / 17.8 (both targets 72 %) | k = 300: 5.6 / 91.6 / 22.4 (both targets 58 %) |
| DFR: group-balanced subsampling | IDRiD | 0.901 | 11.8 / 86.5 / 69.4 | k = 50: 3.0 / 93.2 / 42.6 (both targets 78 %) | k = 100: 3.7 / 92.4 / 48.4 (both targets 71 %) |
| DFR: group-balanced subsampling | Messidor-2 | 0.681 | 4.8 / 51.8 / 17.4 | k = 200: 3.6 / 94.2 / 13.2 (both targets 78 %) | k = 300: 4.9 / 92.5 / 17.2 (both targets 69 %) |
| Group DRO | IDRiD | 0.921 | 7.4 / 91.0 / 66.3 | k = 50: 1.9 / 92.7 / 32.5 (both targets 72 %) | k = 100: 3.0 / 91.6 / 42.5 (both targets 67 %) |
| Group DRO | Messidor-2 | 0.739 | 38.3 / 51.8 / 72.2 | k = 200: 3.9 / 93.2 / 15.4 (both targets 71 %) | k = 300: 5.0 / 92.2 / 19.6 (both targets 69 %) |
| JTT, errors x5 | IDRiD | 0.920 | 6.2 / 90.3 / 58.5 | k = 50: 2.0 / 92.5 / 31.3 (both targets 71 %) | k = 100: 3.0 / 91.8 / 40.6 (both targets 71 %) |
| JTT, errors x5 | Messidor-2 | 0.752 | 67.6 / 43.6 / 94.5 | k = 200: 4.6 / 93.6 / 18.2 (both targets 74 %) | k = 300: 5.7 / 91.6 / 22.6 (both targets 63 %) |
| JTT, errors x20 | IDRiD | 0.900 | 3.1 / 90.3 / 36.8 | k = 50: 1.6 / 92.6 / 22.6 (both targets 74 %) | k = 100: 2.4 / 91.7 / 29.8 (both targets 66 %) |
| JTT, errors x20 | Messidor-2 | 0.731 | 42.9 / 43.6 / 74.1 | k = 200: 3.7 / 93.5 / 14.5 (both targets 77 %) | k = 300: 4.6 / 91.9 / 18.2 (both targets 69 %) |

Made 2026-11-04 19:59 by group_robust.py (287 s).
