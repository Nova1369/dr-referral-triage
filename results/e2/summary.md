# C2 and E2: refitting the three thresholds at a new clinic

Model: logistic regression trained on all 3,662 APTOS photos (standardised ResNet-50 features, C = 0.003); the conformal scores use its raw probabilities (section 7 explains why). It never sees an IDRiD or Messidor-2 photo. For every k, 200 times: k labelled photos of the clinic are drawn by whole groups, the rule is refitted on them and applied to the clinic's other photos. Numbers are means over the draws, with the 5th–95th percentile of the draws in brackets: the spread a single clinic should expect. Targets: at most 10 % of referable eyes sent home, at least 90 % of urgent eyes referred urgently.

B6 uses C1's thresholds from APTOS (the out-of-fold raw probabilities of all APTOS photos): q = 0.369 / 0.743 / 0.898, i.e. a class is in the set when its probability is at least 0.631 / 0.257 / 0.102.

## 1. The model at the new clinics

| Clinic | Photos | No referral / routine / urgent | AUROC referable | AUROC urgent |
|---|---|---|---|---|
| IDRiD | 516 | 37.4 / 32.6 / 30.0 % | 0.929 | 0.880 |
| Messidor-2 | 1,744 | 73.8 / 19.9 / 6.3 % | 0.764 | 0.935 |

## 2. C2 against no recalibration (B6)

Referable eyes sent home, urgent eyes referred urgently, healthy eyes sent home and uncertain in %; referred per 1,000 at the clinic's own prevalence. "Urgent threshold finite": share of draws with at least 9 urgent photos among the k, so that urgency is not simply assumed for everyone. With fewer, the targets are met only because every eye is referred (healthy eyes sent home: 0).

### IDRiD, all 516 photos (random split)

| k | Rule | Referable eyes sent home | Urgent eyes referred urgently | Healthy eyes sent home | Uncertain | Referred per 1,000 | Urgent threshold finite | Both targets met (share of draws) |
|---|---|---|---|---|---|---|---|---|
| none | B6 | 10.6 (7–14) | 87.2 (82–92) | 75.5 (70–81) | 2.3 | 629 | – | 6 % |
| 10 | C2 | 0.0 (0–0) | 100.0 (100–100) | 0.0 (0–0) | 0.0 | 1000 | 0 % | 100 % |
| 20 | C2 | 0.0 (0–0) | 98.5 (87–100) | 0.0 (0–0) | 5.8 | 942 | 12 % | 94 % |
| 30 | C2 | 1.0 (0–6) | 95.0 (80–100) | 13.1 (0–57) | 13.4 | 810 | 55 % | 77 % |
| 40 | C2 | 1.9 (0–7) | 93.4 (79–100) | 30.5 (0–60) | 15.4 | 720 | 88 % | 75 % |
| 50 | C2 | 2.1 (0–7) | 93.6 (81–100) | 36.8 (17–59) | 14.4 | 705 | 99 % | 76 % |
| 75 | C2 | 3.1 (0–7) | 91.9 (81–99) | 43.0 (28–60) | 16.3 | 657 | 100 % | 66 % |
| 100 | C2 | 3.5 (0–8) | 90.9 (78–98) | 45.1 (29–60) | 16.4 | 644 | 100 % | 64 % |
| 150 | C2 | 4.1 (1–9) | 91.1 (81–98) | 48.4 (36–60) | 15.2 | 641 | 100 % | 60 % |
| 200 | C2 | 4.7 (1–10) | 90.6 (81–98) | 50.2 (37–61) | 15.2 | 631 | 100 % | 62 % |
| 300 | C2 | 4.8 (2–8) | 90.7 (82–98) | 51.0 (42–60) | 15.5 | 625 | 100 % | 60 % |

### IDRiD official split (k from the 413 training photos, tested on the 103 test photos)

| k | Rule | Referable eyes sent home | Urgent eyes referred urgently | Healthy eyes sent home | Uncertain | Referred per 1,000 | Urgent threshold finite | Both targets met (share of draws) |
|---|---|---|---|---|---|---|---|---|
| none | B6 | 14.2 (14–14) | 76.7 (75–77) | 74.4 (74–74) | 3.9 | 589 | – | 0 % |
| 10 | C2 | 0.0 (0–0) | 100.0 (100–100) | 0.0 (0–0) | 0.0 | 1000 | 0 % | 100 % |
| 20 | C2 | 0.0 (0–0) | 97.6 (75–100) | 0.0 (0–0) | 6.3 | 937 | 14 % | 92 % |
| 30 | C2 | 1.0 (0–6) | 91.6 (69–100) | 9.2 (0–46) | 15.0 | 809 | 55 % | 73 % |
| 40 | C2 | 2.6 (0–11) | 87.7 (72–100) | 24.1 (0–57) | 19.1 | 701 | 92 % | 59 % |
| 50 | C2 | 3.3 (0–11) | 86.6 (72–97) | 29.2 (3–59) | 18.7 | 682 | 100 % | 56 % |
| 75 | C2 | 4.5 (0–11) | 83.7 (72–97) | 35.1 (10–59) | 20.2 | 637 | 100 % | 40 % |
| 100 | C2 | 5.8 (2–11) | 81.7 (72–97) | 39.7 (15–62) | 20.5 | 609 | 100 % | 28 % |
| 150 | C2 | 6.5 (2–11) | 79.4 (72–97) | 42.7 (26–56) | 21.7 | 580 | 100 % | 14 % |
| 200 | C2 | 6.8 (3–11) | 78.8 (72–91) | 44.7 (36–54) | 21.4 | 573 | 100 % | 10 % |
| 300 | C2 | 7.2 (6–8) | 78.0 (74–81) | 45.9 (38–51) | 21.6 | 564 | 100 % | 0 % |

On the official split the urgent target is missed for k ≥ 40. Here the k photos come from IDRiD's training split and the test is always the same 103 photos, so nothing averages the test set out. Its 32 urgent eyes score lower than the 123 urgent eyes of the training split: median P(urgent) 0.28 against 0.40; 25 % of them sit below the training split's 10th percentile, where 10 % would be expected (Mann–Whitney p = 0.15). With so few eyes this may be chance, but it is the warning that matters in practice: the guarantee holds only if the calibration photos are like the photos the rule is then used on.

### Messidor-2, 1,744 photos (random split)

| k | Rule | Referable eyes sent home | Urgent eyes referred urgently | Healthy eyes sent home | Uncertain | Referred per 1,000 | Urgent threshold finite | Both targets met (share of draws) |
|---|---|---|---|---|---|---|---|---|
| none | B6 | 79.8 (78–82) | 32.8 (29–37) | 97.9 (98–98) | 1.2 | 56 | – | 0 % |
| 10 | C2 | 0.0 (0–0) | 100.0 (100–100) | 0.0 (0–0) | 0.0 | 1000 | 0 % | 100 % |
| 20 | C2 | 0.0 (0–0) | 100.0 (100–100) | 0.0 (0–0) | 0.3 | 997 | 0 % | 100 % |
| 30 | C2 | 0.0 (0–0) | 100.0 (100–100) | 0.0 (0–0) | 0.0 | 1000 | 0 % | 100 % |
| 40 | C2 | 0.0 (0–0) | 100.0 (100–100) | 0.0 (0–0) | 0.3 | 997 | 0 % | 100 % |
| 50 | C2 | 0.1 (0–0) | 99.8 (100–100) | 0.2 (0–0) | 1.1 | 987 | 2 % | 99 % |
| 75 | C2 | 0.5 (0–3) | 99.3 (97–100) | 1.8 (0–16) | 3.8 | 947 | 8 % | 96 % |
| 100 | C2 | 1.3 (0–8) | 97.6 (83–100) | 5.3 (0–29) | 10.6 | 852 | 22 % | 88 % |
| 150 | C2 | 3.4 (0–11) | 94.7 (78–100) | 14.1 (0–36) | 27.3 | 614 | 60 % | 75 % |
| 200 | C2 | 4.8 (0–13) | 94.3 (80–100) | 20.1 (0–40) | 34.8 | 491 | 80 % | 75 % |
| 300 | C2 | 6.2 (1–12) | 92.5 (81–100) | 25.5 (14–38) | 43.5 | 361 | 98 % | 63 % |
| 400 | C2 | 6.5 (2–12) | 91.4 (77–99) | 26.5 (18–36) | 47.1 | 316 | 100 % | 62 % |

## 3. How many labelled local photos are needed?

A finite threshold for a class needs at least 9 calibration photos of that class (α = 0.1); before that, the class is in every set. The urgent class is the rarest at both clinics, so it decides. Theory: binomial draws with the clinic's class shares. Observed: the first k at which at least 90 % of the draws have a finite urgent threshold, and what C2 does there and at the largest k.

| Setting | Urgent share | Theory: k for 9 urgent photos on average / with 90 % chance | Observed k | At that k: referable sent home / urgent referred urgently / healthy sent home / uncertain | At the largest k |
|---|---|---|---|---|---|
| IDRiD | 30.0 % | 30 / 41 | 50 | 2.1 / 93.6 / 36.8 / 14.4 | k = 300: 4.8 / 90.7 / 51.0 / 15.5 |
| IDRiD official split | 29.8 % | 31 / 41 | 40 | 2.6 / 87.7 / 24.1 / 19.1 | k = 300: 7.2 / 78.0 / 45.9 / 21.6 |
| Messidor-2 | 6.3 % | 143 / 204 | 300 | 6.2 / 92.5 / 25.5 / 43.5 | k = 400: 6.5 / 91.4 / 26.5 / 47.1 |

Answer for the report: at IDRiD about 40–50 labelled photos make the rule informative and keep both targets (random split); at Messidor-2 it takes about 200–300, because only 1 eye in 16 is urgent. More photos then tighten the thresholds only slowly: the model's ranking at the clinic limits how many healthy eyes can be sent home.

## 4. Why conformal, and why one threshold per class, at a new clinic

Same draws. B3-k refits the tuned thresholds (plain empirical 90 % points, no finite-sample correction); B4-k refits standard conformal prediction (one threshold for all classes).

**IDRiD**

| k | Urgent referred urgently: C2 / B3-k / B4-k | Referable sent home: C2 / B3-k / B4-k | Healthy sent home: C2 / B3-k / B4-k |
|---|---|---|---|
| 10 | 100.0 / 73.2 / 83.8 | 0.0 / 10.5 / 5.3 | 0.0 / 65.5 / 48.1 |
| 20 | 98.5 / 83.3 / 82.7 | 0.0 / 10.6 / 6.2 | 0.0 / 70.0 / 54.1 |
| 30 | 95.0 / 86.2 / 82.2 | 1.0 / 9.0 / 6.4 | 13.1 / 66.5 / 56.1 |
| 50 | 93.6 / 87.0 / 82.1 | 2.1 / 8.9 / 6.6 | 36.8 / 68.9 / 57.8 |
| 100 | 90.9 / 88.0 / 82.6 | 3.5 / 8.8 / 6.4 | 45.1 / 70.0 / 57.4 |
| 200 | 90.6 / 89.1 / 82.0 | 4.7 / 9.2 / 6.9 | 50.2 / 72.2 / 58.9 |

**IDRiD official split**

| k | Urgent referred urgently: C2 / B3-k / B4-k | Referable sent home: C2 / B3-k / B4-k | Healthy sent home: C2 / B3-k / B4-k |
|---|---|---|---|
| 10 | 100.0 / 63.1 / 75.0 | 0.0 / 13.7 / 7.7 | 0.0 / 60.4 / 40.9 |
| 20 | 97.6 / 75.5 / 72.4 | 0.0 / 12.3 / 8.6 | 0.0 / 62.2 / 47.5 |
| 30 | 91.6 / 76.4 / 70.8 | 1.0 / 12.8 / 9.2 | 9.2 / 63.7 / 51.4 |
| 50 | 86.6 / 75.9 / 69.9 | 3.3 / 13.7 / 9.8 | 29.2 / 68.2 / 53.8 |
| 100 | 81.7 / 77.1 / 71.1 | 5.8 / 13.8 / 9.5 | 39.7 / 70.6 / 53.6 |
| 200 | 78.8 / 76.4 / 72.8 | 6.8 / 13.9 / 9.7 | 44.7 / 72.1 / 54.0 |

**Messidor-2**

| k | Urgent referred urgently: C2 / B3-k / B4-k | Referable sent home: C2 / B3-k / B4-k | Healthy sent home: C2 / B3-k / B4-k |
|---|---|---|---|
| 10 | 100.0 / 86.0 / 54.4 | 0.0 / 10.4 / 32.7 | 0.0 / 19.6 / 58.9 |
| 20 | 100.0 / 78.0 / 60.9 | 0.0 / 14.3 / 28.2 | 0.0 / 33.3 / 57.9 |
| 30 | 100.0 / 75.5 / 56.8 | 0.0 / 12.5 / 32.3 | 0.0 / 31.2 / 63.3 |
| 50 | 99.8 / 74.0 / 56.0 | 0.1 / 13.5 / 33.1 | 0.2 / 38.5 / 65.6 |
| 100 | 97.6 / 83.6 / 58.0 | 1.3 / 11.3 / 29.7 | 5.3 / 37.3 / 64.3 |
| 200 | 94.3 / 85.8 / 57.5 | 4.8 / 11.0 / 29.3 | 20.1 / 37.8 / 65.3 |
| 400 | 91.4 / 87.2 / 57.1 | 6.5 / 10.2 / 28.9 | 26.5 / 36.3 / 65.6 |

B3-k's empirical threshold covers only n/(n + 1) of the positives it was tuned on, on average, and less when n is small: with one urgent calibration photo, half of the clinic's urgent eyes fall below it. C2's ⌈(n + 1)(1 − α)⌉ rule refuses to promise anything until n is large enough.

## 5. Recalibrate or retrain? (B7)

B7a retrains logistic regression on the k local photos only (penalty scaled so each photo counts as in the APTOS model; most likely class). B7b retrains it on APTOS plus the k local photos (equal weight per photo) and keeps the APTOS thresholds. 20 draws per k (the first draws of section 2). AUROC: referable, on the clinic's test photos.

| Setting | k | Rule | Referable sent home | Urgent referred urgently | Healthy sent home | AUROC referable |
|---|---|---|---|---|---|---|
| IDRiD | 20 | C2 | 0.0 | 98.5 | 0.0 | 0.929 |
| IDRiD | 20 | B7a | 26.1 | 45.9 | 75.8 | 0.850 |
| IDRiD | 20 | B7b | 9.8 | 89.2 | 74.5 | 0.930 |
| IDRiD | 50 | C2 | 2.1 | 93.6 | 36.8 | 0.929 |
| IDRiD | 50 | B7a | 19.2 | 60.6 | 74.2 | 0.873 |
| IDRiD | 50 | B7b | 8.3 | 91.3 | 69.6 | 0.930 |
| IDRiD | 100 | C2 | 3.5 | 90.9 | 45.1 | 0.929 |
| IDRiD | 100 | B7a | 15.3 | 61.0 | 75.2 | 0.894 |
| IDRiD | 100 | B7b | 7.5 | 91.9 | 67.5 | 0.934 |
| IDRiD | 200 | C2 | 4.7 | 90.6 | 50.2 | 0.929 |
| IDRiD | 200 | B7a | 15.3 | 65.8 | 81.8 | 0.918 |
| IDRiD | 200 | B7b | 6.7 | 93.7 | 67.9 | 0.938 |
| IDRiD official split | 20 | C2 | 0.0 | 97.6 | 0.0 | 0.929 |
| IDRiD official split | 20 | B7a | 18.0 | 39.5 | 49.6 | 0.759 |
| IDRiD official split | 20 | B7b | 11.8 | 77.0 | 67.3 | 0.892 |
| IDRiD official split | 50 | C2 | 3.3 | 86.6 | 29.2 | 0.929 |
| IDRiD official split | 50 | B7a | 18.2 | 48.0 | 64.0 | 0.827 |
| IDRiD official split | 50 | B7b | 10.9 | 82.0 | 65.6 | 0.893 |
| IDRiD official split | 100 | C2 | 5.8 | 81.7 | 39.7 | 0.929 |
| IDRiD official split | 100 | B7a | 16.7 | 51.4 | 63.8 | 0.836 |
| IDRiD official split | 100 | B7b | 8.9 | 83.5 | 60.4 | 0.891 |
| IDRiD official split | 200 | C2 | 6.8 | 78.8 | 44.7 | 0.929 |
| IDRiD official split | 200 | B7a | 19.4 | 48.7 | 68.2 | 0.848 |
| IDRiD official split | 200 | B7b | 8.4 | 83.3 | 60.5 | 0.894 |
| Messidor-2 | 20 | C2 | 0.0 | 100.0 | 0.0 | 0.764 |
| Messidor-2 | 20 | B7a | 84.9 | 3.5 | 91.7 | 0.624 |
| Messidor-2 | 20 | B7b | 73.6 | 37.8 | 96.2 | 0.763 |
| Messidor-2 | 50 | C2 | 0.1 | 99.8 | 0.2 | 0.764 |
| Messidor-2 | 50 | B7a | 75.4 | 13.0 | 91.8 | 0.702 |
| Messidor-2 | 50 | B7b | 66.4 | 43.3 | 93.7 | 0.767 |
| Messidor-2 | 100 | C2 | 1.3 | 97.6 | 5.3 | 0.764 |
| Messidor-2 | 100 | B7a | 70.4 | 11.2 | 91.6 | 0.719 |
| Messidor-2 | 100 | B7b | 60.4 | 44.8 | 91.2 | 0.773 |
| Messidor-2 | 200 | C2 | 4.8 | 94.3 | 20.1 | 0.764 |
| Messidor-2 | 200 | B7a | 64.1 | 25.1 | 91.8 | 0.754 |
| Messidor-2 | 200 | B7b | 51.9 | 55.0 | 88.2 | 0.782 |
| Messidor-2 | 400 | C2 | 6.5 | 91.4 | 26.5 | 0.764 |
| Messidor-2 | 400 | B7a | 58.0 | 31.9 | 91.3 | 0.773 |
| Messidor-2 | 400 | B7b | 44.7 | 59.9 | 85.2 | 0.793 |

## 6. Additional work (beyond the proposal)

### 6a. C2-PAC: a promise for each clinic, not only on average (δ = 0.1)

C2's guarantee is an average over clinics' calibration draws: a single clinic's coverage can fall below 90 %. C2-PAC picks, per class, the threshold whose coverage is at least 90 % with probability at least 90 % for the clinic's own draw (the coverage of the j-th smallest of n scores follows Beta(j, n + 1 − j)). It needs at least 22 photos of a class before that class can be left out of any set.

Share of draws in which a class's coverage falls below 90 %, measured on the test photos and over all the clinic's eyes of that class (the population the promise is about; on IDRiD's official split, its test photos), C2 / C2-PAC:

| Setting | k | Urgent, test photos | Urgent, all the clinic's urgent eyes | Routine, test photos | Routine, all the clinic's routine eyes | Healthy sent home: C2 / C2-PAC |
|---|---|---|---|---|---|---|
| IDRiD | 50 | 23 % / 0 % | 20 % / 0 % | 22 % / 0 % | 20 % / 0 % | 36.8 / 0.0 |
| IDRiD | 100 | 36 % / 6 % | 32 % / 2 % | 30 % / 4 % | 26 % / 2 % | 45.1 / 26.9 |
| IDRiD | 200 | 38 % / 12 % | 32 % / 2 % | 45 % / 15 % | 36 % / 4 % | 50.2 / 37.9 |
| IDRiD | 300 | 40 % / 12 % | 24 % / 0 % | 44 % / 14 % | 32 % / 1 % | 51.0 / 40.1 |
| IDRiD official split | 50 | 44 % / 1 % | 44 % / 1 % | 32 % / 0 % | 32 % / 0 % | 29.2 / 0.0 |
| IDRiD official split | 100 | 72 % / 12 % | 72 % / 12 % | 41 % / 4 % | 41 % / 4 % | 39.7 / 13.4 |
| IDRiD official split | 200 | 90 % / 30 % | 90 % / 30 % | 45 % / 8 % | 45 % / 8 % | 44.7 / 26.4 |
| IDRiD official split | 300 | 100 % / 44 % | 100 % / 44 % | 50 % / 1 % | 50 % / 1 % | 45.9 / 32.7 |
| Messidor-2 | 50 | 0 % / 0 % | 0 % / 0 % | 22 % / 0 % | 22 % / 0 % | 0.2 / 0.0 |
| Messidor-2 | 100 | 10 % / 0 % | 10 % / 0 % | 30 % / 2 % | 28 % / 2 % | 5.3 / 0.0 |
| Messidor-2 | 200 | 21 % / 0 % | 17 % / 0 % | 36 % / 6 % | 32 % / 6 % | 20.1 / 0.3 |
| Messidor-2 | 300 | 29 % / 4 % | 21 % / 2 % | 36 % / 8 % | 34 % / 4 % | 25.5 / 5.4 |
| Messidor-2 | 400 | 30 % / 6 % | 22 % / 2 % | 38 % / 10 % | 32 % / 4 % | 26.5 / 13.3 |

The promise is about the clinic's eyes of each class as a whole. The test photos are what is left after the draw: a finite sample (about 95 urgent eyes at IDRiD for k = 200) whose measured coverage scatters around the clinic's, and leans low when the draw happened to take the easier eyes. Over all the clinic's eyes (random splits), C2-PAC falls below 90 % in at most 6 % of draws (promised: at most 10 %), C2 in up to 36 %. On IDRiD's official split the shift between its two parts (section 2) breaks the promise of both.


### 6b. The camera-decorrelated model at new clinics (additional camera study)

The same experiment with the logistic regression trained with camera-decorrelated weights (w = P(class) / P(class | camera group)).

| Clinic | AUROC referable: plain / decorrelated | Rule | k | Referable sent home: plain / decorrelated | Urgent referred urgently: plain / decorrelated | Healthy sent home: plain / decorrelated |
|---|---|---|---|---|---|---|
| IDRiD | 0.929 / 0.929 | B6 (no recalibration) | – | 10.6 / 20.4 | 87.2 / 87.0 | 75.5 / 91.1 |
| IDRiD | 0.929 / 0.929 | C2 | 50 | 2.1 / 3.1 | 93.6 / 93.6 | 36.8 / 42.4 |
| IDRiD | 0.929 / 0.929 | C2 | 100 | 3.5 / 4.5 | 90.9 / 90.5 | 45.1 / 51.7 |
| IDRiD | 0.929 / 0.929 | C2 | 200 | 4.7 / 5.6 | 90.6 / 90.6 | 50.2 / 57.4 |
| IDRiD | 0.929 / 0.929 | C2 | 300 | 4.8 / 5.8 | 90.7 / 90.7 | 51.0 / 56.6 |
| Messidor-2 | 0.764 / 0.750 | B6 (no recalibration) | – | 79.8 / 56.3 | 32.8 / 38.9 | 97.9 / 86.7 |
| Messidor-2 | 0.764 / 0.750 | C2 | 50 | 0.1 / 0.1 | 99.8 / 99.8 | 0.2 / 0.2 |
| Messidor-2 | 0.764 / 0.750 | C2 | 100 | 1.3 / 1.4 | 97.6 / 97.8 | 5.3 / 5.2 |
| Messidor-2 | 0.764 / 0.750 | C2 | 200 | 4.8 / 5.0 | 94.3 / 93.5 | 20.1 / 19.6 |
| Messidor-2 | 0.764 / 0.750 | C2 | 400 | 6.5 / 6.6 | 91.4 / 91.7 | 26.5 / 25.5 |

## 7. Why the conformal scores use raw probabilities

E1's isotonic map is learnt on APTOS. It is a step function whose lowest step is exactly 0, so every photo below a certain raw score gets probability 0 for that class, and their order is lost. At APTOS few sick eyes sit there; at Messidor-2 many do.

| Clinic | Routine eyes with calibrated P(routine) = 0 | Urgent eyes with calibrated P(urgent) = 0 |
|---|---|---|
| aptos | 0.1 % | 0.2 % |
| IDRiD | 0.0 % | 0.6 % |
| Messidor-2 | 16.4 % | 17.3 % |

When more than a tenth of a class's calibration photos tie at 0, the conformal threshold of that class is 1: the class enters every set, and every eye is referred. Same draws, Messidor-2:

| k | Healthy sent home: C2 raw / calibrated | Referable sent home: raw / calibrated | Urgent referred urgently: raw / calibrated |
|---|---|---|---|
| 100 | 5.3 / 0.7 | 1.3 / 0.2 | 97.6 / 99.0 |
| 200 | 20.1 / 0.4 | 4.8 / 0.1 | 94.3 / 98.6 |
| 300 | 25.5 / 0.2 | 6.2 / 0.1 | 92.5 / 98.6 |
| 400 | 26.5 / 0.0 | 6.5 / 0.0 | 91.4 / 98.3 |

Conformal coverage holds for any score, calibrated or not; the score only decides how useful the sets are. The calibrated probabilities stay useful where they are read as risks (C3).

## Notes

- Coverage check: with random splits (IDRiD, Messidor-2), C2's per-class coverage averaged over the draws is at least 89.9 % at every k (target 90 %; 200 draws leave about ±0.3 points of simulation noise), as conformal theory promises when the calibration and test photos are exchangeable. On IDRiD's official split it falls to 78.0 % (section 2).
- Group sampling: copies of one photo and the two eyes of a Messidor-2 exam are drawn together, and a test photo whose group was drawn is left out (this removes the IDRiD photo that sits in both official splits when its twin is drawn).
- Made 2026-11-04 04:59 by e2_new_clinics.py (231 s).
