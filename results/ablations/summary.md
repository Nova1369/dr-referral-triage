# Ablations and robustness

## 1. Image ablations: resolution and colour normalisation

Logistic regression with E1's most frequent setting (standardised features, no PCA reduction, C = 0.003), refitted in every E1 outer fold; C1 (α = 0.1) fitted on each fold's calibration photos. New clinics: AUROC of the 10 fold models (mean).

| Features | AUROC referable | Same-camera AUROC referable | AUROC urgent | C1: referable sent home | C1: urgent referred urgently | C1: healthy sent home | IDRiD AUROC referable | Messidor-2 AUROC referable | Messidor-2 AUROC urgent |
|---|---|---|---|---|---|---|---|---|---|
| 224 px (E1) | 0.974 | 0.919 | 0.920 | 3.8 | 93.0 | 87.6 | 0.926 | 0.760 | 0.932 |
| 512 px | 0.975 | 0.912 | 0.930 | 4.4 | 91.0 | 87.6 | 0.932 | 0.782 | 0.947 |
| Graham filter, 224 px | 0.970 | 0.908 | 0.914 | 4.0 | 93.2 | 85.8 | 0.924 | 0.747 | 0.923 |

C2 at the new clinics with each feature set (model trained on all of APTOS; 200 draws of k labelled local photos, as in E2), in %:

| Features | Clinic | k | Referable sent home | Urgent referred urgently | Healthy sent home | Uncertain |
|---|---|---|---|---|---|---|
| 224 px (E1) | IDRiD | 50 | 2.5 | 92.8 | 38.5 | 15.9 |
| 224 px (E1) | IDRiD | 100 | 3.8 | 91.9 | 46.6 | 15.2 |
| 224 px (E1) | Messidor-2 | 200 | 5.2 | 94.0 | 20.9 | 35.5 |
| 224 px (E1) | Messidor-2 | 400 | 6.9 | 91.9 | 27.5 | 45.4 |
| 512 px | IDRiD | 50 | 3.3 | 93.2 | 42.2 | 17.7 |
| 512 px | IDRiD | 100 | 4.9 | 92.4 | 54.5 | 15.8 |
| 512 px | Messidor-2 | 200 | 5.3 | 94.0 | 21.2 | 36.5 |
| 512 px | Messidor-2 | 400 | 7.3 | 90.6 | 29.0 | 47.3 |
| Graham filter, 224 px | IDRiD | 50 | 2.9 | 93.6 | 36.3 | 24.1 |
| Graham filter, 224 px | IDRiD | 100 | 3.7 | 91.5 | 44.2 | 22.6 |
| Graham filter, 224 px | Messidor-2 | 200 | 4.9 | 93.9 | 20.0 | 33.3 |
| Graham filter, 224 px | Messidor-2 | 400 | 7.0 | 91.9 | 26.7 | 44.2 |

## 2. PCA size (logistic regression, E1's inner folds)

Best inner-fold score over C for each number of components, mean over the 10 outer folds (score: mean of the referable and urgent AUROCs).

| PCA components | Score | AUROC referable | AUROC urgent |
|---|---|---|---|
| 256 | 0.9433 | 0.9700 | 0.9165 |
| 512 | 0.9439 | 0.9704 | 0.9174 |
| 1024 | 0.9443 | 0.9708 | 0.9177 |
| 2048 | 0.9443 | 0.9706 | 0.9180 |

## 3. Robustness: worse photos at the clinic

E1's logistic regression (each fold's chosen setting; its predictions on the original photos match E1's saved ones to within 2e-03). Only the test photos (and, for recalibration, the calibration photos) are altered; pooled over the 10 held-out folds, in %.

| Test photos | C1 thresholds from | AUROC referable | AUROC urgent | Referable sent home | Urgent referred urgently | Healthy sent home | Uncertain |
|---|---|---|---|---|---|---|---|
| original photos | original calibration | 0.973 | 0.920 | 4.2 | 91.2 | 87.8 | 1.0 |
| blurred (radius 3 px) | original calibration | 0.918 | 0.872 | 8.2 | 96.5 | 73.5 | 0.5 |
| blurred (radius 3 px) | recalibrated on altered photos | 0.918 | 0.872 | 3.6 | 91.4 | 49.1 | 23.0 |
| half brightness | original calibration | 0.969 | 0.912 | 6.7 | 87.1 | 88.6 | 2.0 |
| half brightness | recalibrated on altered photos | 0.969 | 0.912 | 5.0 | 91.8 | 86.4 | 2.3 |
| JPEG quality 20 | original calibration | 0.945 | 0.851 | 2.9 | 93.0 | 69.7 | 1.1 |
| JPEG quality 20 | recalibrated on altered photos | 0.945 | 0.851 | 1.7 | 89.1 | 65.0 | 10.6 |

Made 2026-11-03 20:34 by ablations.py (178 s).
