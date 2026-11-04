# Additional work: fixing the camera shortcut, tuned under the E1 protocol

Logistic regression on standardised ResNet-50 features; C tuned in 5 grouped inner folds (mean of the two AUROCs), E1's outer folds and calibration photos. C1 (one conformal threshold per class, α = 0.1) is fitted on each fold's calibration photos with the raw probabilities.

- **Weights:** w = P(class) / P(class | camera group): in the weighted training data the camera says nothing about the class.
- **Erasure fitted on healthy eyes (LEACE):** the standardised features are moved so that every camera group has the same mean of every linear feature. The eraser is fitted on the grade-0 training eyes only, so it removes camera style and keeps disease.
- **Erasure within each grade (LEACE):** the same kind of eraser, estimated within each DR grade on all training eyes, so the camera groups have the same means inside every grade, sick eyes included; the differences between grades are kept.

Camera probes (H against M cameras, AUROC; 0.5 = the camera cannot be read; 5 grouped folds, eraser fitted on the training part):

| Eyes | Eyes in the probe (H) | No erasure | Erasure fitted on healthy eyes | Erasure within each grade |
|---|---|---|---|---|
| healthy (grade 0) | 1803 (1531) | 1.000 | 0.500 | 0.500 |
| referable (grades 2-4) | 772 (55) | 0.996 | 0.998 | 0.625 |

C chosen per fold: Camera-decorrelated weights: 0.003, 0.01, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003; Camera erasure fitted on healthy eyes (LEACE): 0.001, 0.003, 0.003, 0.003, 0.001, 0.003, 0.001, 0.001, 0.001, 0.001; Camera erasure within each grade (LEACE): 0.001, 0.003, 0.003, 0.003, 0.001, 0.003, 0.003, 0.001, 0.001, 0.001

## 1. APTOS, out-of-fold

| Model | AUROC referable | Same-camera AUROC referable | AUROC urgent | C1: referable sent home | C1: urgent referred urgently | C1: healthy sent home | C1: referred per 1,000 |
|---|---|---|---|---|---|---|---|
| No fix (E1) | 0.973 | 0.915 | 0.920 | 4.2 | 91.2 | 87.8 | 451 |
| Camera-decorrelated weights | 0.965 | 0.922 | 0.915 | 4.7 | 92.0 | 83.8 | 442 |
| Camera erasure fitted on healthy eyes (LEACE) | 0.972 | 0.920 | 0.919 | 4.6 | 91.4 | 88.1 | 448 |
| Camera erasure within each grade (LEACE) | 0.963 | 0.901 | 0.914 | 6.0 | 92.2 | 85.9 | 439 |

## 2. By camera group (APTOS)

Most likely class (as in the camera study) and the C1 rule, in %.

| Model | Camera group | Referable eyes | Most likely class: referable sent home | Most likely class: healthy referred | C1: referable sent home | C1: urgent referred urgently | C1: healthy sent home |
|---|---|---|---|---|---|---|---|
| No fix (E1) | H | 55.0 | 63.6 | 0.3 | 47.3 | 93.3 | 98.8 |
| No fix (E1) | M | 717.0 | 15.6 | 19.4 | 3.8 | 91.1 | 63.0 |
| No fix (E1) | S | 715.0 | 6.2 | 67.6 | 1.4 | 91.2 | 10.8 |
| Camera-decorrelated weights | H | 55.0 | 27.3 | 1.2 | 9.1 | 93.3 | 92.4 |
| Camera-decorrelated weights | M | 717.0 | 21.1 | 13.9 | 4.5 | 90.7 | 64.0 |
| Camera-decorrelated weights | S | 715.0 | 20.1 | 16.2 | 4.6 | 93.8 | 32.4 |
| Camera erasure fitted on healthy eyes (LEACE) | H | 55.0 | 67.3 | 0.2 | 52.7 | 93.3 | 99.0 |
| Camera erasure fitted on healthy eyes (LEACE) | M | 717.0 | 14.9 | 19.3 | 3.9 | 90.7 | 63.7 |
| Camera erasure fitted on healthy eyes (LEACE) | S | 715.0 | 8.0 | 67.6 | 1.5 | 92.2 | 13.5 |
| Camera erasure within each grade (LEACE) | H | 55.0 | 50.9 | 0.6 | 36.4 | 93.3 | 96.8 |
| Camera erasure within each grade (LEACE) | M | 717.0 | 17.3 | 21.5 | 6.3 | 91.8 | 60.6 |
| Camera erasure within each grade (LEACE) | S | 715.0 | 12.9 | 51.4 | 3.4 | 92.7 | 27.0 |

## 3. The new clinics (the 10 fold models, APTOS thresholds, no recalibration)

| Model | IDRiD AUROC referable | IDRiD: referable sent home / urgent referred urgently / healthy sent home | Messidor-2 AUROC referable | Messidor-2: referable sent home / urgent referred urgently / healthy sent home |
|---|---|---|---|---|
| No fix (E1) | 0.925 | 9.6 / 89.6 / 72.2 | 0.756 | 79.7 / 37.1 / 97.8 |
| Camera-decorrelated weights | 0.924 | 17.6 / 90.4 / 86.6 | 0.746 | 53.9 / 45.5 / 85.0 |
| Camera erasure fitted on healthy eyes (LEACE) | 0.929 | 10.0 / 89.2 / 75.4 | 0.747 | 78.8 / 36.9 / 97.3 |
| Camera erasure within each grade (LEACE) | 0.915 | 16.4 / 87.7 / 84.4 | 0.763 | 68.8 / 54.5 / 95.2 |

## 4. Why not one threshold per camera group (Mondrian by camera)?

Calibration photos per class and camera group, mean over the 10 folds. A class needs at least 9 calibration photos for a finite threshold.

| Camera group | No referral | Routine | Urgent |
|---|---|---|---|
| H | 211.1 | 5.0 | 2.2 |
| M | 78.6 | 57.1 | 39.5 |
| S | 3.9 | 72.9 | 24.8 |

The mostly-healthy cameras have about 5 routine and 2 urgent calibration photos per fold, and the mostly-sick cameras about 4 healthy ones: per-camera thresholds would put those classes in every set for those cameras and refer everyone. APTOS does not have enough sick eyes from the H cameras (55 in all) to give those cameras their own guarantee, so the fix has to be in the model, not in the calibration.

## 5. Why the healthy-eye erasure does not help

Median P(referable) that each model gives the grade-2 (routine) eyes of each camera group, out of fold:

| Model | H cameras (40 eyes) | M cameras (437 eyes) | S cameras (522 eyes) |
|---|---|---|---|
| No fix (E1) | 0.15 | 0.86 | 0.94 |
| Camera-decorrelated weights | 0.64 | 0.82 | 0.87 |
| Camera erasure fitted on healthy eyes (LEACE) | 0.17 | 0.86 | 0.92 |
| Camera erasure within each grade (LEACE) | 0.33 | 0.87 | 0.93 |

Decorrelating weights (mean 1 over APTOS) and APTOS eyes per camera group and class:

| Camera group | No referral | Routine | Urgent |
|---|---|---|---|
| H | 0.61 (1557) | 10.99 (40) | 14.32 (15) |
| M | 1.33 (581) | 0.81 (437) | 0.62 (280) |
| S | 12.07 (37) | 0.39 (522) | 0.52 (193) |

The plain model treats grade-2 eyes from the mostly-healthy cameras as healthy (median P(referable) 0.15, against 0.86 on M cameras): in its training data almost every H-camera photo is healthy. The eraser fitted on healthy eyes removes the camera only where it was estimated. Afterwards the camera can no longer be read from healthy eyes (probe AUROC 0.50), but it can still be read from referable eyes (0.998, against 0.996 before), so the model can still learn that sick-looking H-camera photos are rare. Erasing within each grade brings the referable-eye probe to 0.625. The weights change the rarity itself: the routine and urgent H-camera eyes count 11 and 14 times the average photo.

Made 2026-11-03 21:09 by camera_fixes.py (1276 s).
