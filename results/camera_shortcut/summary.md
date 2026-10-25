# Additional study: the APTOS camera shortcut

APTOS photos come from several cameras (identified here by photo size) that saw very different patients. This study asks whether the models learn the camera instead of the retina, and whether that explains who is missed at IDRiD and Messidor-2.

## 1. Cameras and how much the features know about them

APTOS cameras grouped by their share of referable eyes:

| Group | Cameras (photo sizes) | Photos | Referable photos |
|---|---|---|---|
| mostly-healthy cameras (≤ 10 % referable) | 2048x1536, 819x614, 1050x1050 | 1,612 | 55 (3%) |
| mixed cameras | 1476x1117, 2588x1958, 2416x1736, 4288x2848, 1844x1226 | 1,298 | 717 (55%) |
| mostly-sick cameras (≥ 80 % referable) | 2896x1944, 3388x2588, 3216x2136, 2144x1424, 1504x1000, 2146x1764, 474x358, 1467x1110, 640x480 | 752 | 715 (95%) |

Probes trained and tested on **grade-0 eyes only** (no diabetic retinopathy at all), so disease cannot help them; out-of-fold (APTOS outer folds; grouped 5-fold for the dataset probe):

| What is predicted | From | Eyes | Classes | Result | Chance |
|---|---|---|---|---|---|
| Camera (photo size) | ResNet-50 features | APTOS grade 0 (1,793) | 5 | balanced accuracy 91.1% | 20% |
| Camera (photo size) | quality columns only | APTOS grade 0 (1,793) | 5 | balanced accuracy 96.0% | 20% |
| Mostly-healthy camera or not (score h) | ResNet-50 features | APTOS grade 0 (1,805) | 2 | AUROC 1.000 | 0.5 |
| Dataset (clinic) | ResNet-50 features | grade 0, all datasets (2,990) | 3 | balanced accuracy 98.4% | 33% |

## 2. Who is missed

### 2a. APTOS, by camera group

E1 out-of-fold predictions, most likely class (no tuned threshold, no conformal step). 95 % CIs from 1000 group-bootstrap resamples. In %.

| Model | Camera group | Referable eyes missed | Grade 2 missed | Grade 3–4 missed | Urgent eyes called urgent | Healthy eyes referred |
|---|---|---|---|---|---|---|
| Logistic regression | H (referable 55, urgent 15, healthy 1557) | 63.6 (50.0–78.3) | 80.0 (65.8–91.7) | 20.0 (0.0–50.0) | 40.0 (7.1–72.2) | 0.3 (0.1–0.5) |
| Logistic regression | M (referable 717, urgent 280, healthy 581) | 15.6 (12.8–18.4) | 19.5 (15.8–23.2) | 9.6 (6.3–13.7) | 53.6 (47.9–59.6) | 19.4 (16.2–22.7) |
| Logistic regression | S (referable 715, urgent 193, healthy 37) | 6.2 (4.2–8.1) | 6.3 (4.0–8.8) | 5.7 (2.5–9.8) | 36.3 (29.0–43.6) | 67.6 (51.6–82.1) |
| SVM (RBF) | H (referable 55, urgent 15, healthy 1557) | 61.8 (47.5–75.6) | 75.0 (60.5–87.9) | 26.7 (0.0–57.1) | 33.3 (6.7–66.7) | 0.4 (0.1–0.8) |
| SVM (RBF) | M (referable 717, urgent 280, healthy 581) | 15.3 (12.7–18.2) | 18.8 (15.1–22.5) | 10.0 (6.6–13.9) | 51.4 (45.3–57.3) | 18.1 (15.0–21.5) |
| SVM (RBF) | S (referable 715, urgent 193, healthy 37) | 5.9 (4.0–7.8) | 6.5 (4.4–9.0) | 4.1 (1.5–7.6) | 34.2 (27.2–41.4) | 64.9 (48.6–81.8) |
| B2a Image size only | H (referable 55, urgent 15, healthy 1557) | 100.0 (100.0–100.0) | 100.0 (100.0–100.0) | 100.0 (100.0–100.0) | 0.0 (0.0–0.0) | 0.0 (0.0–0.0) |
| B2a Image size only | M (referable 717, urgent 280, healthy 581) | 61.1 (57.4–64.7) | 61.6 (57.0–65.9) | 60.4 (54.6–66.5) | 5.0 (2.4–7.7) | 27.5 (23.6–31.3) |
| B2a Image size only | S (referable 715, urgent 193, healthy 37) | 0.4 (0.0–1.0) | 0.4 (0.0–1.0) | 0.5 (0.0–1.7) | 14.0 (8.6–20.1) | 100.0 (100.0–100.0) |

Logistic regression misses 64 % of the referable eyes photographed with the mostly-healthy cameras, against 16 % on the mixed cameras (difference +48 points, 95 % CI +33 to +63). Inside the 1050×1050 camera alone the referable AUROC is 0.955: the eyes are ranked well, but the decision is pulled towards "healthy" for that camera.

### 2b. Camera style at the other clinics

h = probability that a photo comes from one of APTOS's mostly-healthy cameras, learnt from APTOS grade-0 eyes only (out-of-fold for APTOS). Predictions: logistic regression (APTOS: out-of-fold; IDRiD and Messidor-2: the 10 E1 fold models averaged), most likely class.

| Dataset | Photos | Median h (IQR) | Photos with h > 0.5 | AUROC referable | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| APTOS | 3,662 | 0.28 (0.02–1.00) | 47% | 0.973 | 87% | 93% |
| IDRiD | 516 | 0.60 (0.33–0.81) | 60% | 0.931 | 73% | 95% |
| Messidor-2 | 1,744 | 1.00 (1.00–1.00) | 100% | 0.764 | 10% | 100% |

The same model on APTOS's camera groups and on the two other clinics. "E0 coverage": share of referable eyes whose true class is in the standard conformal set (90 % target, threshold from APTOS), mean over the 10 folds.

| Photos from | Photos | Median h | Referable eyes | Mean P(referable) of referable eyes | Referable eyes missed | E0 coverage of referable eyes | Healthy eyes referred |
|---|---|---|---|---|---|---|---|
| APTOS, H cameras | 1,612 | 1.00 | 55 | 0.43 | 64% | 45% | 0% |
| APTOS, M cameras | 1,298 | 0.01 | 717 | 0.82 | 16% | 81% | 19% |
| APTOS, S cameras | 752 | 0.08 | 715 | 0.88 | 6% | 85% | 68% |
| IDRiD | 516 | 0.60 | 323 | 0.74 | 27% | 68% | 5% |
| Messidor-2 | 1,744 | 1.00 | 457 | 0.18 | 90% | 8% | 0% |

### 2c. Within each dataset

Referable eyes only, split into thirds of h within each dataset. "Missed": called no referral.

| Dataset | Third of h (range) | Referable eyes | Missed, % (95 % CI) | Grade-2 eyes missed | Mean P(referable) | E0 coverage |
|---|---|---|---|---|---|---|
| APTOS | 1 (0.00–0.02) | 496 | 13.1 (10.0–16.4) | 15 % of 336 | 0.83 | 85 % |
| APTOS | 2 (0.02–0.11) | 495 | 10.7 (8.2–13.6) | 13 % of 349 | 0.84 | 84 % |
| APTOS | 3 (0.11–1.00) | 496 | 14.7 (11.5–17.9) | 18 % of 314 | 0.83 | 76 % |
| IDRiD | 1 (0.01–0.50) | 108 | 33.3 (25.0–42.2) | 45 % of 64 | 0.69 | 62 % |
| IDRiD | 2 (0.50–0.80) | 107 | 29.9 (21.5–39.3) | 47 % of 59 | 0.72 | 65 % |
| IDRiD | 3 (0.80–0.99) | 108 | 17.6 (11.0–25.9) | 29 % of 45 | 0.81 | 76 % |
| Messidor-2 | 1 (0.93–1.00) | 152 | 86.8 (80.1–92.2) | 100 % of 106 | 0.20 | 6 % |
| Messidor-2 | 2 (1.00–1.00) | 152 | 94.1 (89.9–98.0) | 98 % of 127 | 0.14 | 6 % |
| Messidor-2 | 3 (1.00–1.00) | 153 | 88.9 (82.6–94.2) | 97 % of 114 | 0.19 | 11 % |

- APTOS: Spearman correlation between h and P(referable) among referable eyes +0.01 (95 % CI -0.05 to +0.06).
- IDRiD: Spearman correlation between h and P(referable) among referable eyes +0.22 (95 % CI +0.12 to +0.32).
- Messidor-2: Spearman correlation between h and P(referable) among referable eyes -0.01 (95 % CI -0.12 to +0.10).

Within a dataset this test is weak: Messidor-2 has almost no spread in h (nearly every photo looks like a mostly-healthy APTOS camera), and IDRiD uses a single camera. Part 3 therefore tests the mechanism by changing the training data instead.

## 3. First look at two fixes

Same model everywhere: standardised ResNet-50 features, logistic regression, C = 0.003 (one fixed setting, not tuned per fold).

- **Camera-decorrelated weights:** all APTOS photos, each weighted by P(class) / P(class | camera group), so that in the weighted training data the camera says nothing about the class. The overall class mix is unchanged.
- **Mixed cameras only:** the 1,298 photos from the mixed cameras, where the camera says little about the class.
- **Random subsets** (control for the previous row): 20 random sets of APTOS photos with the same size and class mix (581 / 437 / 280) from all cameras; mean (lowest–highest).

### 3a. Tested on the other clinics

| Training set | Photos | IDRiD AUROC referable | IDRiD sensitivity | Messidor-2 AUROC referable | Messidor-2 AUROC urgent | Messidor-2 sensitivity | Messidor-2 specificity |
|---|---|---|---|---|---|---|---|
| All of APTOS | 3,662 | 0.929 | 75% | 0.764 | 0.935 | 12% | 99% |
| All of APTOS, camera-decorrelated weights | 3,662 | 0.929 | 61% | 0.750 | 0.912 | 25% | 97% |
| Mixed cameras only | 1,298 | 0.913 | 68% | 0.632 | 0.872 | 23% | 92% |
| Random APTOS subsets, same size and class mix as the mixed cameras | 1,298 | 0.918 (0.901–0.929) | 82% (74%–89%) | 0.745 (0.702–0.769) | 0.916 (0.892–0.931) | 11% (6%–14%) | 99% (99%–100%) |

Sensitivity and specificity use the most likely class, so they also depend on the class mix of the training set: compare the mixed-camera row with the random subsets, and the weighted row with all of APTOS.

### 3b. APTOS, 10-fold cross-validation (E1 outer folds)

| Training | AUROC referable | Same-camera AUROC referable | AUROC urgent | Referable missed: H / M / S cameras | Healthy referred: H / M / S cameras |
|---|---|---|---|---|---|
| unweighted | 0.975 | 0.922 | 0.921 | 58% / 15% / 5% | 0% / 19% / 73% |
| camera-decorrelated weights | 0.966 | 0.925 | 0.917 | 27% / 20% / 21% | 1% / 13% / 22% |

Made 2026-10-25 06:28 by camera_shortcut.py (133 s).
