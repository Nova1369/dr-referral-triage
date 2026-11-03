# C3 and E3: prevalence and workload

Part A uses the out-of-fold C1 decisions on APTOS (lr, α = 0.1). In each of 200 repeats APTOS is split by group into two halves. Half A gives every class's decision rates; half B becomes a clinic with 5 %, 12.5 % or 25 % referable eyes (all its non-referable eyes, and referable eyes drawn by grade so that APTOS's mix of grades 2, 3 and 4 is kept). Only the prevalence changes, so this is pure label shift. Means over the repeats.

## 1. The guarantee when only the prevalence changes

Referable eyes sent home and urgent eyes referred urgently, in %. Each class's rate does not depend on how common the class is, so C1's per-class guarantee holds at every prevalence [11]. Last column: the share of all eyes whose true class is in B4's set (its marginal coverage), from B4's per-class coverage on APTOS (96.3 / 88.9 / 67.0 %) weighted by the clinic's class shares.

| Referable eyes in the clinic | C1: referable sent home | C1: urgent referred urgently | B4: referable sent home | B4: urgent referred urgently | B4: all eyes covered |
|---|---|---|---|---|---|
| 5 % | 4.1 | 91.2 | 5.6 | 67.5 | 95.5 |
| 12.5 % | 4.4 | 91.2 | 6.0 | 67.3 | 94.5 |
| 25 % | 4.3 | 91.1 | 5.9 | 66.8 | 92.6 |
| APTOS (40.6 %) | 4.3 | 91.1 | 5.9 | 66.9 | 90.4 |

B4's marginal guarantee (90 % of all eyes covered) looks better as sick eyes get rarer, because healthy eyes dominate the average; its urgent eyes stay at about two in three throughout. A marginal guarantee cannot protect a rare class.

## 2. The workload changes a lot

Observed in half B (C1 rule), per 1,000 eyes screened.

| Referable eyes in the clinic | Referred | Urgent referrals | Uncertain | Referable eyes sent home | PPV of a referral (%) |
|---|---|---|---|---|---|
| 5 % | 154 | 101 | 10 | 2.0 | 30.7 |
| 12.5 % | 217 | 150 | 10 | 5.5 | 54.6 |
| 25 % | 321 | 233 | 10 | 10.8 | 73.7 |
| APTOS (40.6 %) | 451 | 336 | 10 | 17.5 | 85.2 |

## 3. Forecasting the workload from unlabelled photos

Forecast minus observed, mean over repeats (mean absolute error in brackets), for the C1 rule. B8 assumes APTOS's shares; EM and BBSE estimate the shares from the clinic's unlabelled photos; B9 knows them.

**Referred per 1,000**

| Referable eyes in the clinic | Observed | B8 APTOS shares (no correction) | EM from unlabelled photos [12] | BBSE from unlabelled photos (additional) | B9 true shares (oracle) |
|---|---|---|---|---|---|
| 5 % | 154.5 | +297.2 (297.2) | +3.8 (10.5) | +0.5 (6.6) | +0.7 (11.7) |
| 12.5 % | 216.8 | +234.8 (234.8) | +3.7 (9.0) | -0.2 (6.4) | +0.9 (10.9) |
| 25 % | 320.8 | +130.8 (130.8) | +3.5 (7.5) | -0.1 (5.9) | +1.1 (9.6) |
| APTOS (40.6 %) | 451.1 | +0.5 (13.8) | +3.0 (5.8) | -0.0 (5.7) | +1.1 (8.1) |

**Urgent referrals per 1,000**

| Referable eyes in the clinic | Observed | B8 APTOS shares (no correction) | EM from unlabelled photos [12] | BBSE from unlabelled photos (additional) | B9 true shares (oracle) |
|---|---|---|---|---|---|
| 5 % | 100.5 | +236.3 (236.3) | +2.6 (8.6) | +0.6 (7.5) | +0.5 (9.2) |
| 12.5 % | 149.7 | +187.1 (187.1) | +2.5 (7.6) | +0.7 (7.6) | +1.1 (8.4) |
| 25 % | 232.7 | +104.0 (104.0) | +1.4 (7.5) | +0.3 (8.4) | +0.9 (9.0) |
| APTOS (40.6 %) | 336.1 | +0.7 (12.7) | +0.9 (8.0) | +0.7 (9.4) | +1.3 (9.5) |

**Uncertain per 1,000**

| Referable eyes in the clinic | Observed | B8 APTOS shares (no correction) | EM from unlabelled photos [12] | BBSE from unlabelled photos (additional) | B9 true shares (oracle) |
|---|---|---|---|---|---|
| 5 % | 9.5 | +0.3 (2.8) | -0.5 (3.2) | -0.5 (3.2) | -0.5 (3.2) |
| 12.5 % | 9.6 | +0.2 (2.8) | -0.4 (3.1) | -0.4 (3.1) | -0.4 (3.1) |
| 25 % | 10.0 | -0.2 (2.8) | -0.5 (2.9) | -0.5 (2.9) | -0.5 (2.9) |
| APTOS (40.6 %) | 10.4 | -0.5 (2.7) | -0.4 (2.7) | -0.5 (2.7) | -0.5 (2.7) |

**Referable eyes sent home per 1,000**

| Referable eyes in the clinic | Observed | B8 APTOS shares (no correction) | EM from unlabelled photos [12] | BBSE from unlabelled photos (additional) | B9 true shares (oracle) |
|---|---|---|---|---|---|
| 5 % | 2.0 | +14.9 (14.9) | +0.3 (1.2) | +0.0 (1.3) | +0.1 (1.1) |
| 12.5 % | 5.5 | +11.5 (11.5) | -0.0 (1.8) | -0.3 (2.0) | -0.3 (1.7) |
| 25 % | 10.8 | +6.2 (6.4) | -0.0 (2.8) | -0.4 (3.0) | -0.3 (2.7) |
| APTOS (40.6 %) | 17.5 | -0.5 (3.6) | -0.1 (3.7) | -0.5 (3.9) | -0.5 (3.6) |

**PPV of a referral (%)**

| Referable eyes in the clinic | Observed | B8 APTOS shares (no correction) | EM from unlabelled photos [12] | BBSE from unlabelled photos (additional) | B9 true shares (oracle) |
|---|---|---|---|---|---|
| 5 % | 30.7 | +54.3 (54.3) | +1.5 (3.9) | -0.6 (7.3) | -0.2 (2.4) |
| 12.5 % | 54.6 | +30.5 (30.5) | +0.6 (3.5) | -0.6 (5.0) | -0.1 (2.7) |
| 25 % | 73.7 | +11.4 (11.4) | +0.2 (2.5) | -0.3 (3.0) | -0.1 (2.1) |
| APTOS (40.6 %) | 85.2 | -0.1 (1.6) | +0.1 (1.5) | -0.2 (1.8) | -0.1 (1.4) |

## 4. How well are the class shares estimated? (APTOS resampled)

Estimated share of referable eyes (routine + urgent), mean (5th–95th percentile over repeats), in %.

| Referable eyes in the clinic | EM | BBSE | B8 |
|---|---|---|---|
| 5 % | 5.4 (4.5–6.3) | 5.0 (2.7–7.4) | 40.6 (39.1–41.8) |
| 12.5 % | 12.8 (11.8–14.1) | 12.4 (10.3–14.9) | 40.6 (39.1–41.8) |
| 25 % | 25.3 (24.3–26.3) | 24.9 (22.7–27.0) | 40.6 (39.1–41.8) |
| APTOS (40.6 %) | 40.9 (39.2–42.5) | 40.5 (38.0–43.0) | 40.6 (39.1–41.8) |

## 5. Each eye's risk, adjusted for the clinic's prevalence

Calibration error (ECE, 10 bins) of P(referable) and its mean, against the true share of referable eyes. The adjusted risk is p(y|x) × π_y / π_APTOS,y, renormalised.

| Referable eyes in the clinic | ECE unadjusted | ECE adjusted with EM shares | ECE adjusted with true shares | Mean risk unadjusted / EM-adjusted (%) |
|---|---|---|---|---|
| 5 % | 0.092 | 0.012 | 0.010 | 14.2 / 5.4 |
| 12.5 % | 0.073 | 0.015 | 0.014 | 19.8 / 12.8 |
| 25 % | 0.041 | 0.015 | 0.014 | 29.1 / 25.3 |
| APTOS (40.6 %) | 0.014 | 0.014 | 0.014 | 40.8 / 40.9 |

## 6. The real clinics: estimating the class shares

No referral / routine / urgent, in %. Unlabelled-photo estimates use the all-APTOS model of E2. "k labelled local photos": the shares among the k photos C2 is refitted on (mean, 5th–95th percentile of the urgent share over 200 draws).

| Clinic | Truth | B8 APTOS | Most likely class | Mean probability | EM | BBSE | k labelled local photos |
|---|---|---|---|---|---|---|---|
| IDRiD | 37.4 / 32.6 / 30.0 | 59.4 / 27.3 / 13.3 | 47.3 / 39.3 / 13.4 | 43.4 / 35.2 / 21.4 | 30.7 / 40.7 / 28.6 | 45.6 / 30.4 / 24.0 | k = 50: urgent 29.8 (21.9–42.0); k = 100: urgent 30.0 (23.9–37.0); k = 200: urgent 30.0 (25.5–34.5) |
| Messidor-2 | 73.8 / 19.9 / 6.3 | 59.4 / 27.3 / 13.3 | 96.2 / 3.2 / 0.6 | 92.2 / 6.3 / 1.4 | 99.5 / 0.0 / 0.5 | 98.8 / 0.0 / 1.2 | k = 100: urgent 6.4 (2.0–12.0); k = 200: urgent 6.4 (3.5–10.5); k = 400: urgent 6.3 (4.0–8.8) |

EM and BBSE assume that only the class shares differ from APTOS. At Messidor-2 the photos themselves differ (camera, population, grading), so the model's probabilities are off and so are these estimates. The labelled photos that C2 needs anyway give an unbiased estimate.

## 7. The real clinics: forecasting C2's workload

C2 refitted on k labelled local photos; its decision rates per class are measured on those same k photos; the forecast for the clinic's other photos uses the class shares from the k photos (local), from EM on the unlabelled photos, from APTOS (B8) or the truth (B9). Forecast minus observed, mean over draws (mean absolute error).

| Clinic | k | Measure | Observed | local | EM | B8 | B9 |
|---|---|---|---|---|---|---|---|
| IDRiD | 50 | Referred per 1,000 | 695.9 | -0.1 (57.3) | +30.7 (49.9) | -137.8 (142.4) | -0.6 (44.1) |
| IDRiD | 50 | Urgent referrals per 1,000 | 640.9 | +16.4 (56.9) | +38.5 (53.6) | -115.6 (121.1) | +16.0 (47.8) |
| IDRiD | 50 | PPV of a referral (%) | 82.5 | +0.9 (5.2) | +4.5 (5.4) | -14.0 (15.0) | +1.0 (4.1) |
| IDRiD | 100 | Referred per 1,000 | 656.7 | -1.1 (44.7) | +33.5 (40.1) | -155.5 (155.7) | -2.5 (32.4) |
| IDRiD | 100 | Urgent referrals per 1,000 | 580.6 | +7.5 (43.5) | +32.0 (41.1) | -135.5 (135.9) | +6.5 (34.7) |
| IDRiD | 100 | PPV of a referral (%) | 85.5 | +0.8 (4.0) | +3.8 (4.4) | -13.3 (13.5) | +0.9 (3.2) |
| IDRiD | 200 | Referred per 1,000 | 629.4 | -2.8 (34.2) | +32.7 (34.5) | -162.6 (162.6) | -3.4 (23.5) |
| IDRiD | 200 | Urgent referrals per 1,000 | 535.8 | +4.1 (35.0) | +28.0 (32.5) | -142.0 (142.0) | +4.1 (26.1) |
| IDRiD | 200 | PPV of a referral (%) | 87.6 | +0.2 (2.9) | +2.9 (3.4) | -13.5 (13.5) | +0.2 (2.4) |
| Messidor-2 | 100 | Referred per 1,000 | 826.7 | +7.1 (14.3) | -28.5 (28.5) | +15.6 (22.6) | -3.4 (14.0) |
| Messidor-2 | 100 | Urgent referrals per 1,000 | 817.7 | +8.8 (14.0) | -25.8 (25.8) | +16.5 (23.8) | -2.3 (13.1) |
| Messidor-2 | 100 | PPV of a referral (%) | 32.2 | +1.4 (6.4) | -31.2 (31.2) | +15.6 (15.6) | +0.7 (1.8) |
| Messidor-2 | 200 | Referred per 1,000 | 447.8 | +6.9 (27.9) | -80.5 (81.1) | +66.7 (68.2) | +4.2 (25.5) |
| Messidor-2 | 200 | Urgent referrals per 1,000 | 424.5 | +8.3 (27.2) | -76.7 (77.8) | +67.9 (69.0) | +5.4 (24.9) |
| Messidor-2 | 200 | PPV of a referral (%) | 45.0 | +0.3 (5.6) | -43.1 (43.1) | +17.2 (17.2) | +0.5 (3.6) |
| Messidor-2 | 400 | Referred per 1,000 | 328.7 | +2.1 (25.9) | -94.9 (94.9) | +73.8 (73.9) | +2.2 (23.1) |
| Messidor-2 | 400 | Urgent referrals per 1,000 | 298.6 | +2.2 (25.4) | -91.3 (91.3) | +73.7 (73.7) | +2.3 (22.5) |
| Messidor-2 | 400 | PPV of a referral (%) | 49.7 | -0.1 (4.6) | -47.5 (47.5) | +17.3 (17.3) | +0.2 (2.8) |

## Notes

- Referrals per 1,000 can also simply be counted once a clinic has run the rule on its photos; the forecast matters before that (planning capacity) and for the quantities that need labels: the PPV and the referable eyes sent home.
- Made 2026-11-03 20:00 by c3_prevalence.py (14 s).
