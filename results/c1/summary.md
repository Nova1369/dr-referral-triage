# C1: a referral rule with one conformal threshold per class

Every rule below is fitted, in each E1 outer fold, on that fold's calibration images (about 495 APTOS photos the model never trained on: 294 / 135 / 66 per class) and applied to the held-out fold, so each of the 3,662 APTOS photos is decided once by a rule that never saw it. Model inputs: the raw probabilities saved by E1. 95 % CIs: 1,000 bootstrap resamples of whole groups (the E1 resamples). Targets: at most 10 % of referable eyes sent home, at least 90 % of urgent (grade 3–4) eyes referred urgently.

Rules: **argmax** the most likely class (as in E1). **B3** two thresholds tuned on the calibration images so that 90 % of their referable eyes clear the first (P(referable)) and 90 % of their urgent eyes the second (P(urgent)). **B4** standard conformal prediction, one threshold for all classes (E0). **B5** uncertain if the top class probability is below 0.6, else the most likely class. **C1** one conformal threshold per class; the set gives the decision: contains urgent → urgent; only routine → routine; only no referral → no referral; anything else → uncertain (a grader decides). **C1-g** (additional variant) uses C1's sets but sends a set holding urgent and another class to a grader instead of referring it urgently; its urgent guarantee counts urgent eyes referred urgently or sent to a grader. **C1-q** as C1 with one threshold per class and sharpness tertile. B4, C1 and C1-q turn their sets into decisions with the same table.

## 1. The rules compared (Logistic regression, α = 0.1)

In %, pooled over the 10 held-out folds, with 95 % CIs. "Uncertain": share of all eyes sent to a grader. Referred: routine and urgent decisions per 1,000 eyes screened (APTOS prevalence, 40.6 % referable). The most likely class is the decision of E1.

| Rule | Referable eyes sent home (≤ 10) | Urgent eyes referred urgently (≥ 90) | Urgent eyes referred urgently or to a grader | Routine eyes sent home | Urgent eyes sent home | Healthy eyes sent home | Uncertain | Referred per 1,000 | Urgent referrals per 1,000 | Urgent referrals that are urgent |
|---|---|---|---|---|---|---|---|---|---|---|
| Most likely class | 12.8 (11.1–14.6) | 46.3 (41.9–51.3) | 46.3 | 15.0 | 8.4 | 93.5 | 0.0 | 393 | 93 | 66 |
| B3 tuned thresholds | 7.9 (6.4–9.2) | 88.5 (85.4–91.6) | 88.5 | 10.0 | 3.5 | 90.6 | 0.0 | 430 | 310 | 38 |
| B4 standard conformal | 5.9 (4.6–7.1) | 67.0 (62.7–71.4) | 72.7 | 6.8 | 3.9 | 89.1 | 9.5 | 353 | 164 | 54 |
| B5 uncertain if top probability < 0.6 | 6.8 (5.5–8.0) | 34.4 (30.0–38.9) | 66.6 | 8.2 | 3.9 | 90.1 | 15.8 | 279 | 62 | 74 |
| C1 class-wise conformal | 4.2 (3.2–5.2) | 91.2 (88.3–93.8) | 91.6 | 5.2 | 2.3 | 87.8 | 1.0 | 451 | 336 | 36 |
| C1-g as C1, sets with urgent and another class to a grader | 4.2 (3.2–5.2) | 29.5 (25.0–33.7) | 91.6 | 5.2 | 2.3 | 87.8 | 28.8 | 173 | 58 | 68 |
| C1-q class-wise conformal by sharpness tertile | 2.7 (1.9–3.5) | 93.0 (90.5–95.5) | 94.5 | 3.2 | 1.6 | 70.5 | 4.7 | 523 | 448 | 28 |

Fold to fold (lowest–highest of the 10 held-out folds, about 150 referable and 49 urgent eyes each):

| Rule | Referable eyes sent home | Urgent eyes referred urgently |
|---|---|---|
| Most likely class | 10–19 | 35–59 |
| B3 tuned thresholds | 5–12 | 79–94 |
| B4 standard conformal | 2–10 | 52–74 |
| B5 uncertain if top probability < 0.6 | 2–9 | 27–43 |
| C1 class-wise conformal | 2–9 | 83–96 |
| C1-g as C1, sets with urgent and another class to a grader | 2–9 | 16–49 |
| C1-q class-wise conformal by sharpness tertile | 1–5 | 88–100 |

## 2. Coverage of each class against conformal theory

Share of eyes whose true class is in their set. "Expected": the exact average coverage that conformal theory gives for continuous scores, ⌈(n + 1)(1 − α)⌉ / (n + 1), averaged over the folds (n: calibration images of that class for C1, all calibration images for B4). B4 promises only the average over all eyes.

| Rule | Class | Calibration images per fold | Expected | Observed (95 % CI) | Lowest–highest fold |
|---|---|---|---|---|---|
| B4 standard conformal | no referral | 495 (all classes) | 90.1 (all eyes) | 96.3 (95.4–97.1) | 94–98 |
| B4 standard conformal | routine | 495 (all classes) | 90.1 (all eyes) | 88.9 (86.9–90.9) | 80–95 |
| B4 standard conformal | urgent | 495 (all classes) | 90.1 (all eyes) | 67.0 (62.7–71.4) | 52–74 |
| C1 class-wise conformal | no referral | 294 | 90.2 | 89.7 (88.5–91.1) | 87–92 |
| C1 class-wise conformal | routine | 135 | 90.4 | 89.3 (87.4–91.3) | 83–98 |
| C1 class-wise conformal | urgent | 66 | 91.1 | 91.2 (88.3–93.8) | 83–96 |
| C1-q class-wise conformal by sharpness tertile | no referral | about a third of C1's | ≥ 90 (per tertile) | 89.9 (88.7–91.2) | 85–94 |
| C1-q class-wise conformal by sharpness tertile | routine | about a third of C1's | ≥ 90 (per tertile) | 91.3 (89.3–93.1) | 83–98 |
| C1-q class-wise conformal by sharpness tertile | urgent | about a third of C1's | ≥ 90 (per tertile) | 93.0 (90.5–95.5) | 88–100 |
| B3 tuned thresholds | urgent (P(urgent) at or above its threshold) | 66 | 89.6 | 88.5 (85.4–91.6) | 79–94 |
| B4 standard conformal | mean set size | | | 1.22 classes; empty sets 0.0 % | |
| C1 class-wise conformal | mean set size | | | 1.29 classes; empty sets 0.1 % | |
| C1-q class-wise conformal by sharpness tertile | mean set size | | | 1.46 classes; empty sets 0.7 % | |

B3 and C1 set their urgent threshold on the same calibration images and differ by one image: C1 takes the ⌈(n + 1)(1 − α)⌉-th smallest score, B3 the plain empirical 90 % point, whose expected coverage (n − ⌊nα⌋) / (n + 1) is 89.6 % here, just under the target. With fewer calibration images (a new clinic, C2) the gap widens: for n = 20, B3 expects 85.7 % and C1 90.5 %.

## 3. What happens to each DR grade

Share of the eyes of each grade given each decision, in % (Logistic regression, α = 0.1, all 3,662 APTOS eyes).

**B3 tuned thresholds**

| Grade | Eyes | No referral | Uncertain | Routine | Urgent |
|---|---|---|---|---|---|
| 0 | 1805 | 98.6 | 0.0 | 0.6 | 0.8 |
| 1 | 370 | 51.4 | 0.0 | 18.9 | 29.7 |
| 2 | 999 | 10.0 | 0.0 | 32.2 | 57.8 |
| 3 | 193 | 2.6 | 0.0 | 6.2 | 91.2 |
| 4 | 295 | 4.1 | 0.0 | 9.2 | 86.8 |

**C1 class-wise conformal**

| Grade | Eyes | No referral | Uncertain | Routine | Urgent |
|---|---|---|---|---|---|
| 0 | 1805 | 97.8 | 0.2 | 0.8 | 1.2 |
| 1 | 370 | 38.6 | 4.3 | 23.0 | 34.1 |
| 2 | 999 | 5.2 | 1.5 | 29.2 | 64.1 |
| 3 | 193 | 2.1 | 0.5 | 5.7 | 91.7 |
| 4 | 295 | 2.4 | 0.3 | 6.4 | 90.8 |

## 4. A stricter target: α = 0.05 (95 % targets)

Same rules with 95 % in place of 90 % (B5 has no target and is unchanged). Targets: at most 5 % of referable eyes sent home, at least 95 % of urgent eyes referred urgently.

| Rule | Referable eyes sent home | Urgent eyes referred urgently | Healthy eyes sent home | Uncertain | Referred per 1,000 | Urgent referrals per 1,000 |
|---|---|---|---|---|---|---|
| B3 tuned thresholds | 3.8 (2.8–4.8) | 95.1 (92.9–97.1) | 86.7 | 0.0 | 470 | 394 |
| B4 standard conformal | 3.0 (2.2–4.0) | 82.2 (78.8–85.8) | 84.3 | 14.8 | 339 | 247 |
| C1 class-wise conformal | 2.4 (1.6–3.2) | 96.5 (94.6–98.2) | 82.5 | 4.7 | 453 | 421 |
| C1-q class-wise conformal by sharpness tertile | 1.0 (0.5–1.6) | 96.7 (95.0–98.3) | 41.0 | 4.0 | 712 | 700 |

## 5. C1 with every model (α = 0.1)

The guarantee does not depend on the model; what the model changes is how many eyes can safely be sent home. Sorted by the referable AUROC of E1. Last column: healthy eyes sent home when the calibrated probabilities are used instead (the targets hold there too).

| Model | AUROC referable (E1) | Referable eyes sent home | Urgent eyes referred urgently | Healthy eyes sent home | Uncertain | Referred per 1,000 | Sent home per 1,000 | Healthy eyes sent home, calibrated |
|---|---|---|---|---|---|---|---|---|
| Logistic regression, 5 grades | 0.974 | 4.3 | 91.4 | 86.7 | 1.6 | 451 | 532 | 86.5 |
| Logistic regression | 0.973 | 4.2 | 91.2 | 87.8 | 1.0 | 451 | 539 | 87.9 |
| SVM (RBF) | 0.973 | 4.8 | 92.0 | 87.8 | 1.7 | 442 | 541 | 87.4 |
| LDA | 0.964 | 3.2 | 92.6 | 81.5 | 3.2 | 471 | 497 | 83.3 |
| MLP | 0.963 | 3.8 | 92.4 | 81.9 | 2.3 | 475 | 502 | 83.8 |
| Gradient boosting | 0.958 | 4.0 | 93.2 | 81.5 | 2.7 | 473 | 500 | 80.8 |
| k-NN | 0.950 | 5.7 | 91.4 | 80.5 | 4.2 | 457 | 501 | 80.8 |
| AdaBoost | 0.950 | 2.9 | 94.3 | 72.3 | 9.7 | 462 | 441 | 74.6 |
| Random forest | 0.949 | 5.1 | 92.2 | 79.6 | 2.8 | 478 | 494 | 80.1 |
| B2 Image quality only | 0.930 | 3.8 | 90.4 | 74.9 | 1.9 | 520 | 461 | 75.4 |
| B2b Quality, without size | 0.930 | 4.2 | 90.8 | 74.8 | 2.0 | 519 | 461 | 75.4 |
| QDA | 0.921 | 2.4 | 97.7 | 41.1 | 1.6 | 730 | 254 | 80.5 |
| B2a Image size only | 0.920 | 5.5 | 92.4 | 72.8 | 0.6 | 540 | 455 | 73.1 |
| Decision tree | 0.913 | 4.7 | 92.4 | 72.0 | 2.7 | 526 | 447 | 73.2 |
| Naive Bayes | 0.901 | 3.8 | 93.0 | 71.5 | 3.1 | 529 | 440 | 72.2 |
| B1 Most likely class | 0.500 | 0.0 | 100.0 | 0.0 | 0.0 | 1000 | 0 | 0.0 |

Ties in the score cost efficiency, not validity: QDA's raw probabilities are pinned at 0 or 1 for most photos (77 % have a top probability above 0.99), so many scores tie and C1 has to refer more eyes; its calibrated probabilities are spread out and work better. For logistic regression both kinds agree.

## 6. Subgroups: camera group and image sharpness (Logistic regression, α = 0.1)

Camera groups as in the camera study (APTOS photo sizes with ≤ 10 % referable eyes: H; ≥ 80 %: S; the rest: M). Sharpness tertiles: thirds of the variance of the Laplacian, with edges set on each fold's calibration images. Small groups give rough numbers: the counts are shown.

| Subgroup | Eyes (referable / urgent) | C1: referable sent home | C1: urgent referred urgently | C1: healthy sent home | C1-q: referable sent home | C1-q: urgent referred urgently | C1-q: healthy sent home |
|---|---|---|---|---|---|---|---|
| camera group H | 1612 (55 / 15) | 47.3 | 93.3 | 98.8 | 18.2 | 100.0 | 78.6 |
| camera group M | 1298 (717 / 280) | 3.8 | 91.1 | 63.0 | 3.1 | 93.2 | 52.8 |
| camera group S | 752 (715 / 193) | 1.4 | 91.2 | 10.8 | 1.1 | 92.2 | 8.1 |
| sharpness: least sharp third | 1237 (738 / 229) | 3.4 | 89.5 | 69.1 | 3.4 | 90.0 | 67.9 |
| sharpness: middle third | 1233 (489 / 172) | 2.7 | 91.9 | 88.4 | 2.0 | 94.2 | 85.8 |
| sharpness: sharpest third | 1192 (260 / 87) | 9.6 | 94.3 | 97.2 | 1.9 | 98.9 | 59.8 |

C1-q: smallest calibration cell (class × tertile) 6 images; cells with too few images for a finite threshold: 3 of 90 (that class is then in every set of its tertile).

## 7. The same thresholds at new clinics, without recalibration (Logistic regression, α = 0.1)

Each fold's APTOS thresholds applied unchanged to IDRiD and Messidor-2 (B6 in the proposal). Mean over the 10 fold models, lowest–highest in brackets. This is the starting point of C2, which refits the thresholds on a few labelled local images.

| Test set | Rule | Referable eyes sent home | Urgent eyes referred urgently | Healthy eyes sent home | Uncertain | Coverage: no referral / routine / urgent |
|---|---|---|---|---|---|---|
| APTOS, held-out folds | B3 tuned thresholds | 7.9 (5–12) | 88.5 (79–94) | 90.6 (86–94) | 0.0 (0–0) | – |
| APTOS, held-out folds | B4 standard conformal | 5.9 (2–10) | 67.0 (52–74) | 89.1 (86–93) | 9.5 (6–14) | 96 / 89 / 67 |
| APTOS, held-out folds | C1 class-wise conformal | 4.2 (2–9) | 91.2 (83–96) | 87.8 (86–90) | 1.0 (0–2) | 90 / 89 / 91 |
| IDRiD, all 516 | B3 tuned thresholds | 16.6 (10–23) | 87.3 (82–94) | 84.2 (75–93) | 0.0 (0–0) | – |
| IDRiD, all 516 | B4 standard conformal | 14.0 (11–19) | 61.0 (54–69) | 80.6 (73–85) | 19.9 (14–26) | 98 / 74 / 61 |
| IDRiD, all 516 | C1 class-wise conformal | 9.6 (4–16) | 89.6 (85–97) | 72.2 (59–84) | 3.9 (1–8) | 82 / 75 / 90 |
| IDRiD, official test split (103) | B3 tuned thresholds | 22.0 (14–30) | 78.1 (69–88) | 81.0 (67–92) | 0.0 (0–0) | – |
| IDRiD, official test split (103) | B4 standard conformal | 17.2 (14–22) | 45.6 (34–59) | 73.1 (64–79) | 21.9 (16–28) | 97 / 72 / 46 |
| IDRiD, official test split (103) | C1 class-wise conformal | 12.3 (3–19) | 82.2 (75–97) | 66.7 (49–77) | 4.2 (0–11) | 76 / 72 / 82 |
| Messidor-2 (1,744) | B3 tuned thresholds | 85.6 (79–93) | 32.5 (20–52) | 98.9 (97–100) | 0.0 (0–0) | – |
| Messidor-2 (1,744) | B4 standard conformal | 83.6 (77–91) | 15.5 (12–27) | 98.5 (97–100) | 3.3 (1–5) | 100 / 5 / 15 |
| Messidor-2 (1,744) | C1 class-wise conformal | 79.7 (73–88) | 37.1 (26–53) | 97.8 (96–100) | 1.2 (0–3) | 99 / 6 / 37 |

## 8. Robustness: calibrated instead of raw probabilities (Logistic regression, α = 0.1)

Same folds, calibration images and rules; only the input probabilities change.

| Rule | Referable eyes sent home: raw / calibrated | Urgent eyes referred urgently: raw / calibrated | Healthy eyes sent home: raw / calibrated | Uncertain: raw / calibrated | Referred per 1,000: raw / calibrated |
|---|---|---|---|---|---|
| Most likely class | 12.8 / 10.3 | 46.3 / 42.8 | 93.5 / 92.2 | 0.0 / 0.0 | 393 / 410 |
| B3 tuned thresholds | 7.9 / 8.3 | 88.5 / 89.5 | 90.6 / 90.5 | 0.0 / 0.0 | 430 / 428 |
| B4 standard conformal | 5.9 / 5.2 | 67.0 / 66.8 | 89.1 / 88.5 | 9.5 / 7.9 | 353 / 374 |
| B5 uncertain if top probability < 0.6 | 6.8 / 5.9 | 34.4 / 31.8 | 90.1 / 89.3 | 15.8 / 15.9 | 279 / 286 |
| C1 class-wise conformal | 4.2 / 4.4 | 91.2 / 91.4 | 87.8 / 87.9 | 1.0 / 1.2 | 451 / 448 |

## Notes

- The guarantee is an average over calibration draws and eyes: single folds vary (section 1, fold range), and with about 66 urgent calibration images per fold the realised coverage of one draw has a standard deviation of about 3.5 points.
- C1's per-class guarantee covers routine and urgent eyes separately; B3's tuned threshold targets the pooled referable eyes, so its guarantee depends on the mix of routine and urgent eyes it was tuned on.
- B4, C1 and C1-q use the decision table of the proposal, so an eye whose set contains urgent is referred urgently even if other classes are in its set.
- Input probabilities: raw (section 8 repeats the main comparison with the other kind).
- Made 2026-11-03 20:00 by c1_referral.py (43 s).
