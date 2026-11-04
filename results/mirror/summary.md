# Additional work: mirror-averaged features

Left and right eyes are mirror images, but ResNet-50's features are not mirror-invariant. Averaging the features of a photo and of its mirror image makes the model give a photo and its mirror image the same answer. APTOS: E1's outer folds, each fold's chosen setting of E1's logistic regression refitted (on the photo's own features it reproduces E1's saved predictions to within 2e-03), C1 at α = 0.1 on the fold's calibration photos, pooled over the 10 held-out folds. In %.

## 1. APTOS

| Features | AUROC referable | Same-camera AUROC referable | AUROC urgent | C1: referable sent home | C1: urgent referred urgently | C1: healthy sent home | Referred per 1,000 | C1's decision changes when the photo is mirrored | ... for referable eyes |
|---|---|---|---|---|---|---|---|---|---|
| Features of the photo (E1) | 0.973 | 0.915 | 0.920 | 4.2 | 91.2 | 87.8 | 451 | 10.9 | 16.7 |
| Mirror-averaged features | 0.974 | 0.920 | 0.924 | 4.4 | 91.0 | 87.7 | 447 | 0.0 | 0.0 |

With the photo's own features, 6.2 % of referable eyes are sent home in at least one of the two orientations, against 4.2 % in the original one.

## 2. New clinics

All-APTOS logistic regression (C = 0.003, no PCA). C2: thresholds refitted on k labelled local photos, 200 draws by whole groups, as in E2. In %.

| Features | IDRiD AUROC referable | Messidor-2 AUROC referable | Messidor-2 AUROC urgent |
|---|---|---|---|
| Features of the photo (E1) | 0.929 | 0.764 | 0.935 |
| Mirror-averaged features | 0.927 | 0.760 | 0.933 |

| Features | Clinic | k | Referable sent home | Urgent referred urgently | Healthy sent home | Uncertain |
|---|---|---|---|---|---|---|
| Features of the photo (E1) | IDRiD | 50 | 2.5 | 92.8 | 38.5 | 15.9 |
| Mirror-averaged features | IDRiD | 50 | 2.5 | 92.7 | 40.9 | 16.4 |
| Features of the photo (E1) | IDRiD | 100 | 3.8 | 91.9 | 46.6 | 15.2 |
| Mirror-averaged features | IDRiD | 100 | 3.3 | 91.6 | 46.9 | 16.8 |
| Features of the photo (E1) | Messidor-2 | 200 | 5.2 | 94.0 | 20.9 | 35.5 |
| Mirror-averaged features | Messidor-2 | 200 | 4.8 | 93.9 | 20.2 | 36.0 |
| Features of the photo (E1) | Messidor-2 | 400 | 6.9 | 91.9 | 27.5 | 45.4 |
| Mirror-averaged features | Messidor-2 | 400 | 6.6 | 91.5 | 27.2 | 44.8 |

Made 2026-11-04 15:03 by mirror_fix.py (205 s).
