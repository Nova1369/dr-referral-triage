# E1: baselines and course models (APTOS, 10-fold cross-validation)

3662 APTOS images, each predicted once by the model of the outer fold that held it out. 95 % CIs: 1000 group-bootstrap resamples. Classes: no referral (grade 0–1, 59.4%), routine (grade 2, 27.3%), urgent (grade 3–4, 13.3%).

## Ranking and agreement

| Model | AUROC referable | AUROC urgent | QWK | Accuracy |
|---|---|---|---|---|
| B0 Refer none | – | – | 0.000 (0.000–0.000) | 0.594 (0.578–0.610) |
| B0 Refer all (urgent) | – | – | 0.000 (0.000–0.000) | 0.133 (0.122–0.144) |
| B1 Most likely class | 0.500 (0.481–0.517) | 0.498 (0.470–0.526) | 0.000 (0.000–0.000) | 0.594 (0.578–0.610) |
| B2 Image quality only | 0.930 (0.922–0.939) | 0.798 (0.780–0.815) | 0.540 (0.510–0.571) | 0.736 (0.722–0.751) |
| B2a Image size only | 0.920 (0.911–0.929) | 0.774 (0.755–0.795) | 0.469 (0.439–0.500) | 0.732 (0.718–0.747) |
| B2b Quality, without size | 0.930 (0.921–0.938) | 0.798 (0.780–0.815) | 0.539 (0.511–0.570) | 0.738 (0.724–0.752) |
| Logistic regression, 5 grades | 0.974 (0.970–0.979) | 0.920 (0.910–0.931) | 0.770 (0.751–0.790) | 0.823 (0.810–0.836) |
| Logistic regression | 0.973 (0.968–0.977) | 0.920 (0.909–0.931) | 0.766 (0.746–0.785) | 0.821 (0.809–0.835) |
| SVM (RBF) | 0.973 (0.968–0.977) | 0.919 (0.908–0.930) | 0.766 (0.747–0.786) | 0.822 (0.810–0.835) |
| LDA | 0.964 (0.958–0.969) | 0.913 (0.901–0.924) | 0.772 (0.752–0.791) | 0.819 (0.807–0.832) |
| MLP | 0.963 (0.957–0.968) | 0.895 (0.879–0.910) | 0.743 (0.722–0.765) | 0.804 (0.791–0.818) |
| Gradient boosting | 0.958 (0.952–0.964) | 0.892 (0.878–0.905) | 0.715 (0.692–0.737) | 0.792 (0.780–0.805) |
| Random forest | 0.949 (0.942–0.956) | 0.886 (0.871–0.900) | 0.693 (0.669–0.715) | 0.784 (0.771–0.797) |
| AdaBoost | 0.950 (0.943–0.956) | 0.883 (0.869–0.897) | 0.699 (0.676–0.720) | 0.772 (0.757–0.786) |
| k-NN | 0.950 (0.944–0.957) | 0.875 (0.856–0.891) | 0.638 (0.616–0.659) | 0.759 (0.745–0.774) |
| QDA | 0.921 (0.913–0.931) | 0.839 (0.819–0.859) | 0.731 (0.711–0.751) | 0.770 (0.755–0.784) |
| Naive Bayes | 0.901 (0.891–0.912) | 0.845 (0.827–0.862) | 0.570 (0.542–0.599) | 0.708 (0.693–0.724) |
| Decision tree | 0.913 (0.904–0.923) | 0.823 (0.807–0.839) | 0.585 (0.556–0.612) | 0.726 (0.711–0.741) |

## Decisions at the most likely class, and calibration

Sensitivity and specificity in %. These use the plain argmax (no tuned threshold, no conformal step): they show what the models do before C1.

| Model | Sens. referable | Spec. referable | Sens. urgent | Referred per 1,000 | Brier raw → calibrated | ECE referable raw → calibrated |
|---|---|---|---|---|---|---|
| B0 Refer none | 0.0 | 100.0 | 0.0 | 0 | – | – |
| B0 Refer all (urgent) | 100.0 | 0.0 | 100.0 | 1000 | – | – |
| B1 Most likely class | 0.0 | 100.0 | 0.0 | 0 | 0.555 → 0.555 | 0.000 → 0.000 |
| B2 Image quality only | 74.8 | 89.8 | 15.6 | 365 | 0.339 → 0.331 | 0.020 → 0.017 |
| B2a Image size only | 66.6 | 90.9 | 8.4 | 324 | 0.340 → 0.341 | 0.021 → 0.013 |
| B2b Quality, without size | 74.3 | 89.4 | 17.4 | 365 | 0.337 → 0.334 | 0.016 → 0.012 |
| Logistic regression, 5 grades | 88.1 | 93.1 | 44.9 | 399 | 0.240 → 0.241 | 0.015 → 0.010 |
| Logistic regression | 87.2 | 93.5 | 46.3 | 393 | 0.244 → 0.243 | 0.015 → 0.009 |
| SVM (RBF) | 87.5 | 93.7 | 44.1 | 392 | 0.241 → 0.242 | 0.018 → 0.008 |
| LDA | 89.2 | 91.5 | 51.6 | 413 | 0.267 → 0.262 | 0.011 → 0.014 |
| MLP | 85.1 | 92.9 | 45.1 | 388 | 0.272 → 0.268 | 0.020 → 0.013 |
| Gradient boosting | 85.3 | 91.1 | 33.8 | 400 | 0.281 → 0.280 | 0.015 → 0.013 |
| Random forest | 85.0 | 90.7 | 22.7 | 401 | 0.308 → 0.295 | 0.069 → 0.013 |
| AdaBoost | 87.2 | 88.5 | 31.6 | 422 | 0.600 → 0.309 | 0.271 → 0.024 |
| k-NN | 85.4 | 89.1 | 2.3 | 411 | 0.321 → 0.305 | 0.065 → 0.010 |
| QDA | 93.1 | 85.0 | 50.2 | 467 | 0.421 → 0.305 | 0.109 → 0.024 |
| Naive Bayes | 72.1 | 85.7 | 35.5 | 377 | 0.380 → 0.361 | 0.068 → 0.015 |
| Decision tree | 75.9 | 87.8 | 21.1 | 381 | 0.361 → 0.357 | 0.028 → 0.017 |

## Confusion matrix: Logistic regression (most likely class)

Rows: true class; columns: predicted class.

| True \ predicted | No referral | Routine | Urgent | Total |
|---|---|---|---|---|
| No referral | 2033 | 129 | 13 | 2175 |
| Routine | 150 | 748 | 101 | 999 |
| Urgent | 41 | 221 | 226 | 488 |

## Confusion matrix: Logistic regression, 5 grades (most likely class)

Rows: true class; columns: predicted class.

| True \ predicted | No referral | Routine | Urgent | Total |
|---|---|---|---|---|
| No referral | 2025 | 139 | 11 | 2175 |
| Routine | 140 | 768 | 91 | 999 |
| Urgent | 37 | 232 | 219 | 488 |

## Shortcut check: AUROC within the same camera

APTOS photo size identifies the camera, and the cameras differ hugely in how sick their patients are (see results/kmeans). Here a referable eye is only compared with a non-referable eye taken with the same camera (per-camera AUROCs weighted by their number of pairs), so the camera alone cannot help. A model that only knows the camera scores 0.5.

| Model | AUROC referable, all pairs | Same camera | AUROC urgent, all pairs | Same camera |
|---|---|---|---|---|
| B1 Most likely class | 0.500 | 0.518 (0.490–0.547) | 0.498 | 0.502 (0.470–0.539) |
| B2 Image quality only | 0.930 | 0.673 (0.639–0.705) | 0.798 | 0.622 (0.584–0.660) |
| B2a Image size only | 0.920 | 0.545 (0.514–0.581) | 0.774 | 0.557 (0.520–0.593) |
| B2b Quality, without size | 0.930 | 0.673 (0.639–0.707) | 0.798 | 0.621 (0.584–0.659) |
| Logistic regression, 5 grades | 0.974 | 0.913 (0.895–0.929) | 0.920 | 0.858 (0.832–0.880) |
| Logistic regression | 0.973 | 0.915 (0.898–0.931) | 0.920 | 0.858 (0.833–0.880) |
| SVM (RBF) | 0.973 | 0.914 (0.897–0.930) | 0.919 | 0.860 (0.835–0.882) |
| LDA | 0.964 | 0.896 (0.877–0.915) | 0.913 | 0.847 (0.820–0.871) |
| MLP | 0.963 | 0.884 (0.865–0.902) | 0.895 | 0.825 (0.799–0.848) |
| Gradient boosting | 0.958 | 0.874 (0.851–0.894) | 0.892 | 0.811 (0.782–0.837) |
| Random forest | 0.949 | 0.852 (0.830–0.875) | 0.886 | 0.805 (0.774–0.830) |
| AdaBoost | 0.950 | 0.861 (0.838–0.881) | 0.883 | 0.800 (0.771–0.826) |
| k-NN | 0.950 | 0.855 (0.835–0.876) | 0.875 | 0.783 (0.751–0.810) |
| QDA | 0.921 | 0.752 (0.727–0.779) | 0.839 | 0.761 (0.730–0.790) |
| Naive Bayes | 0.901 | 0.777 (0.750–0.803) | 0.845 | 0.781 (0.750–0.807) |
| Decision tree | 0.913 | 0.761 (0.733–0.788) | 0.823 | 0.698 (0.661–0.730) |

## Difference to logistic regression (reference model)

AUROC difference with its paired 95 % CI; a CI that excludes 0 marks a real difference.

| Model | Δ AUROC referable | Δ AUROC urgent |
|---|---|---|
| B1 Most likely class | -0.473 (-0.492 to -0.456) | -0.422 (-0.451 to -0.389) |
| B2 Image quality only | -0.042 (-0.050 to -0.035) | -0.122 (-0.141 to -0.102) |
| B2a Image size only | -0.053 (-0.060 to -0.044) | -0.145 (-0.166 to -0.124) |
| B2b Quality, without size | -0.043 (-0.051 to -0.036) | -0.122 (-0.140 to -0.103) |
| Logistic regression, 5 grades | +0.002 (+0.000 to +0.003) | +0.000 (-0.003 to +0.004) |
| SVM (RBF) | -0.000 (-0.002 to +0.001) | -0.001 (-0.005 to +0.003) |
| LDA | -0.009 (-0.012 to -0.007) | -0.007 (-0.014 to -0.002) |
| MLP | -0.010 (-0.013 to -0.006) | -0.025 (-0.035 to -0.015) |
| Gradient boosting | -0.015 (-0.018 to -0.011) | -0.028 (-0.037 to -0.020) |
| Random forest | -0.024 (-0.028 to -0.019) | -0.034 (-0.045 to -0.025) |
| AdaBoost | -0.023 (-0.027 to -0.019) | -0.037 (-0.048 to -0.027) |
| k-NN | -0.023 (-0.027 to -0.018) | -0.045 (-0.060 to -0.032) |
| QDA | -0.051 (-0.059 to -0.044) | -0.081 (-0.099 to -0.063) |
| Naive Bayes | -0.071 (-0.080 to -0.063) | -0.075 (-0.090 to -0.060) |
| Decision tree | -0.059 (-0.067 to -0.052) | -0.096 (-0.112 to -0.081) |

## Chosen settings

Most frequent setting over the 10 outer folds, and how often a chosen value sat at the edge of its grid (a sign the grid could be widened).

| Model | Most frequent setting | Folds | At grid edge |
|---|---|---|---|
| B1 Most likely class | `{}` | 10/10 | 0/10 |
| B2 Image quality only | `{"lr": 0.05, "leaves": 63}` | 3/10 | 5/10 (leaves=63 leaves=7) |
| B2a Image size only | `{"lr": 0.05, "leaves": 7}` | 4/10 | 6/10 (leaves=7) |
| B2b Quality, without size | `{"lr": 0.05, "leaves": 15}` | 4/10 | 4/10 (leaves=7) |
| Logistic regression, 5 grades | `{"pca": 2048, "C": 0.003}` | 3/10 | 2/10 (pca=256) |
| Logistic regression | `{"pca": 2048, "C": 0.003}` | 4/10 | 0/10 |
| SVM (RBF) | `{"pca": 512, "C": 8.0, "gamma": 0.125}` | 7/10 | 8/10 (gamma=0.125) |
| LDA | `{"pca": 128}` | 7/10 | 0/10 |
| MLP | `{"pca": 128, "hidden": [512, 128], "alpha": 0.001}` | 2/10 | 8/10 (alpha=0.0001 alpha=0.01 pca=64) |
| Gradient boosting | `{"pca": 128, "lr": 0.05, "leaves": 7}` | 4/10 | 0/10 |
| Random forest | `{"pca": 64, "max_features": 0.2}` | 5/10 | 1/10 (pca=32) |
| AdaBoost | `{"pca": 128, "depth": 3, "lr": 0.5}` | 4/10 | 0/10 |
| k-NN | `{"pca": 128, "k": 100}` | 6/10 | 2/10 (pca=64) |
| QDA | `{"pca": 32, "reg": 0.1}` | 4/10 | 7/10 (pca=128 pca=32 reg=0.1) |
| Naive Bayes | `{"pca": 16}` | 9/10 | 1/10 (pca=8) |
| Decision tree | `{"pca": 16, "depth": 8, "leaf": 60}` | 5/10 | 10/10 (depth=12 leaf=100 pca=16 pca=64) |

## Protocol

- Outer folds: the `fold` column of `manifest.csv` (10 folds, stratified by grade, groups never split).
- In each outer fold, 15 % of the training images (by group) are kept for the conformal step (C1); the models never see them. Hyper-parameters are tuned by 5-fold grouped cross-validation on the other 85 %, choosing the best mean of the two AUROCs; the best setting is refitted on all 85 %.
- ResNet-50 features: standardised and reduced with PCA inside every training split.
- B2 uses only the columns of `manifest.csv` that describe the photo, not the retina: B2a the photo and retina size, B2b brightness, contrast, colour and sharpness.
- Calibration: one-vs-rest isotonic regression fitted on the inner out-of-fold predictions.
- Copies of the same photo with different grades are all kept (each copy with its own grade); they always fall in the same fold.
- Made 2026-10-25 06:00 by summarize_e1.py.
