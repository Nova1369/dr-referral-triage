# Additional work: label-efficient recalibration at a new clinic (C2-IW)

All-APTOS logistic regression (E2), raw probabilities. For each clinic and label budget k, 1000 draws. C2: k random photos by whole groups. Rarity sampling: Poisson sampling of whole groups with probability proportional to (1 − λ) + λ r(x)/3, r(x) = Σ_c p(c | x) / mean p(c), k photos labelled on average; thresholds weighted by 1/π (C2-IW), unweighted, or exact weighted conformal. Each draw splits the clinic's exam groups at random into two halves: labels come from half A, and the rule is tested on half B (the clinic's future patients); every λ sees the same split and lottery numbers. In %, means over draws; α = 0.1; ± is the Monte Carlo standard error.

## Messidor-2

| k | Method | Labelled (urgent) | Coverage no referral / routine | Urgent coverage | Urgent coverage below 90 % (share of draws) | Referable sent home | Healthy sent home | Uncertain | Urgent threshold finite |
|---|---|---|---|---|---|---|---|---|---|
| 25 | C2: random photos | 25 (1.5) | 92.7 / 98.9 | 100.0 ± 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0 |
| 25 | C2-IW, λ = 0.25 (ours) | 24 (3.1) | 93.6 / 99.2 | 99.8 ± 0.1 | 1 | 0.0 | 0.0 | 0.7 | 1 |
| 25 | C2-IW, λ = 0.5 | 24 (4.6) | 94.5 / 99.3 | 99.6 ± 0.1 | 2 | 0.0 | 0.0 | 1.5 | 2 |
| 25 | Rarity sampling without weights (λ = 0.25) | 24 (3.1) | 94.0 / 98.6 | 99.5 ± 0.1 | 2 | 0.0 | 0.0 | 2.0 | 3 |
| 25 | Exact weighted conformal (λ = 0.25) | 24 (3.1) | 92.7 / 99.6 | 100.0 ± 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0 |
| 50 | C2: random photos | 50 (3.1) | 90.9 / 94.9 | 99.8 ± 0.1 | 1 | 0.1 | 0.3 | 0.9 | 2 |
| 50 | C2-IW, λ = 0.25 (ours) | 50 (6.3) | 91.5 / 95.4 | 98.6 ± 0.2 | 5 | 0.4 | 1.3 | 5.6 | 10 |
| 50 | C2-IW, λ = 0.5 | 50 (9.3) | 91.9 / 95.0 | 96.1 ± 0.3 | 14 | 0.8 | 2.8 | 14.2 | 23 |
| 50 | Rarity sampling without weights (λ = 0.25) | 50 (6.3) | 92.8 / 93.9 | 96.2 ± 0.3 | 14 | 1.1 | 4.2 | 13.2 | 23 |
| 50 | Exact weighted conformal (λ = 0.25) | 50 (6.3) | 91.2 / 96.6 | 99.7 ± 0.0 | 0 | 0.0 | 0.0 | 0.0 | 2 |
| 75 | C2: random photos | 75 (4.7) | 91.0 / 92.5 | 99.3 ± 0.1 | 3 | 0.5 | 1.8 | 3.7 | 8 |
| 75 | C2-IW, λ = 0.25 (ours) | 76 (9.4) | 91.0 / 92.4 | 95.1 ± 0.3 | 17 | 1.8 | 7.0 | 18.2 | 34 |
| 75 | C2-IW, λ = 0.5 | 75 (13.8) | 91.2 / 92.0 | 91.3 ± 0.4 | 32 | 3.3 | 12.1 | 30.7 | 58 |
| 75 | Rarity sampling without weights (λ = 0.25) | 76 (9.4) | 92.6 / 91.2 | 91.4 ± 0.4 | 31 | 3.8 | 14.0 | 30.2 | 58 |
| 75 | Exact weighted conformal (λ = 0.25) | 76 (9.4) | 90.8 / 93.3 | 98.9 ± 0.1 | 1 | 0.0 | 0.1 | 0.6 | 5 |
| 100 | C2: random photos | 100 (6.2) | 90.7 / 92.4 | 98.1 ± 0.2 | 6 | 1.2 | 4.7 | 10.0 | 21 |
| 100 | C2-IW, λ = 0.25 (ours) | 100 (12.3) | 91.1 / 92.2 | 92.6 ± 0.4 | 26 | 3.3 | 13.5 | 30.6 | 61 |
| 100 | C2-IW, λ = 0.5 | 101 (18.3) | 91.2 / 91.8 | 89.2 ± 0.4 | 40 | 4.9 | 19.2 | 43.3 | 86 |
| 100 | Rarity sampling without weights (λ = 0.25) | 100 (12.3) | 92.9 / 91.0 | 88.8 ± 0.4 | 40 | 5.1 | 20.0 | 41.0 | 81 |
| 100 | Exact weighted conformal (λ = 0.25) | 100 (12.3) | 91.0 / 92.8 | 97.8 ± 0.1 | 5 | 0.3 | 1.1 | 3.2 | 13 |
| 150 | C2: random photos | 150 (9.4) | 90.5 / 91.0 | 94.8 ± 0.3 | 19 | 3.7 | 14.8 | 25.8 | 58 |
| 150 | C2-IW, λ = 0.25 (ours) | 150 (18.8) | 90.4 / 91.4 | 90.7 ± 0.3 | 35 | 5.7 | 22.8 | 43.5 | 93 |
| 150 | C2-IW, λ = 0.5 | 150 (25.9) | 90.5 / 91.2 | 89.3 ± 0.3 | 40 | 6.1 | 24.2 | 48.1 | 99 |
| 150 | Rarity sampling without weights (λ = 0.25) | 150 (18.8) | 92.3 / 89.9 | 86.4 ± 0.4 | 53 | 7.2 | 27.4 | 49.2 | 99 |
| 150 | Exact weighted conformal (λ = 0.25) | 150 (18.8) | 90.3 / 91.7 | 95.4 ± 0.2 | 14 | 2.1 | 8.2 | 17.5 | 47 |
| 200 | C2: random photos | 200 (12.4) | 90.4 / 91.1 | 93.6 ± 0.3 | 23 | 5.0 | 20.8 | 36.1 | 82 |
| 200 | C2-IW, λ = 0.25 (ours) | 200 (24.4) | 90.4 / 91.0 | 90.5 ± 0.3 | 36 | 6.3 | 25.6 | 46.0 | 100 |
| 200 | C2-IW, λ = 0.5 | 200 (32.1) | 90.5 / 90.9 | 89.8 ± 0.3 | 41 | 6.4 | 25.8 | 47.3 | 100 |
| 200 | Rarity sampling without weights (λ = 0.25) | 200 (24.4) | 92.3 / 89.4 | 85.6 ± 0.3 | 58 | 7.7 | 29.1 | 50.1 | 100 |
| 200 | Exact weighted conformal (λ = 0.25) | 200 (24.4) | 90.3 / 91.3 | 93.3 ± 0.2 | 24 | 4.5 | 18.3 | 33.2 | 81 |
| 300 | C2: random photos | 300 (19.0) | 90.4 / 90.8 | 92.2 ± 0.2 | 30 | 6.4 | 26.0 | 43.7 | 99 |
| 300 | C2-IW, λ = 0.25 (ours) | 300 (34.3) | 90.3 / 90.6 | 90.9 ± 0.2 | 37 | 6.8 | 26.8 | 47.2 | 100 |
| 300 | C2-IW, λ = 0.5 | 300 (42.0) | 90.4 / 90.5 | 90.3 ± 0.2 | 40 | 6.9 | 27.0 | 47.8 | 100 |
| 300 | Rarity sampling without weights (λ = 0.25) | 300 (34.3) | 92.2 / 89.1 | 86.1 ± 0.3 | 62 | 8.0 | 29.9 | 50.7 | 100 |
| 300 | Exact weighted conformal (λ = 0.25) | 300 (34.3) | 90.2 / 90.8 | 92.5 ± 0.2 | 28 | 6.4 | 25.9 | 43.8 | 100 |

## APTOS resampled to 5 % referable

| k | Method | Labelled (urgent) | Coverage no referral / routine | Urgent coverage | Urgent coverage below 90 % (share of draws) | Referable sent home | Healthy sent home | Uncertain | Urgent threshold finite |
|---|---|---|---|---|---|---|---|---|---|
| 50 | C2: random photos | 50 (0.8) | 91.3 / 100.0 | 100.0 ± 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0 |
| 50 | C2-IW, λ = 0.25 (ours) | 50 (1.9) | 91.3 / 100.0 | 100.0 ± 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0 |
| 50 | C2-IW, λ = 0.5 | 50 (2.9) | 91.8 / 100.0 | 100.0 ± 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0 |
| 50 | Rarity sampling without weights (λ = 0.25) | 50 (1.9) | 94.9 / 100.0 | 100.0 ± 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0 |
| 50 | Exact weighted conformal (λ = 0.25) | 50 (1.9) | 90.7 / 99.9 | 100.0 ± 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0 |
| 100 | C2: random photos | 100 (1.7) | 90.4 / 100.0 | 100.0 ± 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0 |
| 100 | C2-IW, λ = 0.25 (ours) | 100 (3.8) | 90.6 / 99.4 | 100.0 ± 0.0 | 0 | 0.0 | 0.0 | 0.1 | 0 |
| 100 | C2-IW, λ = 0.5 | 100 (6.1) | 90.8 / 97.5 | 99.3 ± 0.1 | 3 | 0.0 | 0.9 | 3.2 | 5 |
| 100 | Rarity sampling without weights (λ = 0.25) | 100 (3.8) | 94.5 / 98.9 | 99.9 ± 0.1 | 0 | 0.0 | 0.1 | 0.5 | 1 |
| 100 | Exact weighted conformal (λ = 0.25) | 100 (3.8) | 90.4 / 98.7 | 100.0 ± 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0 |
| 150 | C2: random photos | 150 (2.5) | 90.2 / 99.5 | 100.0 ± 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0 |
| 150 | C2-IW, λ = 0.25 (ours) | 150 (5.8) | 90.4 / 96.3 | 99.4 ± 0.1 | 3 | 0.0 | 1.1 | 2.5 | 4 |
| 150 | C2-IW, λ = 0.5 | 150 (9.0) | 90.5 / 92.0 | 95.2 ± 0.3 | 19 | 0.9 | 23.4 | 7.5 | 35 |
| 150 | Rarity sampling without weights (λ = 0.25) | 150 (5.8) | 94.5 / 94.7 | 98.2 ± 0.2 | 7 | 0.2 | 4.5 | 5.1 | 11 |
| 150 | Exact weighted conformal (λ = 0.25) | 150 (5.8) | 90.2 / 96.8 | 100.0 ± 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0 |
| 200 | C2: random photos | 200 (3.3) | 90.3 / 97.9 | 99.9 ± 0.0 | 0 | 0.0 | 0.0 | 0.3 | 0 |
| 200 | C2-IW, λ = 0.25 (ours) | 200 (7.5) | 90.4 / 93.2 | 97.7 ± 0.2 | 9 | 0.5 | 12.2 | 5.1 | 20 |
| 200 | C2-IW, λ = 0.5 | 200 (11.8) | 90.5 / 90.7 | 91.0 ± 0.4 | 36 | 2.3 | 62.7 | 5.2 | 77 |
| 200 | Rarity sampling without weights (λ = 0.25) | 200 (7.5) | 94.5 / 92.5 | 95.9 ± 0.3 | 15 | 0.9 | 23.0 | 5.8 | 32 |
| 200 | Exact weighted conformal (λ = 0.25) | 200 (7.5) | 90.2 / 95.8 | 99.9 ± 0.0 | 0 | 0.0 | 0.0 | 0.0 | 1 |
| 300 | C2: random photos | 300 (5.0) | 90.3 / 94.3 | 99.4 ± 0.1 | 2 | 0.1 | 3.4 | 1.4 | 6 |
| 300 | C2-IW, λ = 0.25 (ours) | 300 (11.4) | 90.2 / 91.6 | 91.6 ± 0.4 | 32 | 2.1 | 62.8 | 4.5 | 76 |
| 300 | C2-IW, λ = 0.5 | 300 (16.1) | 90.3 / 91.0 | 91.0 ± 0.3 | 36 | 2.8 | 82.4 | 4.4 | 99 |
| 300 | Rarity sampling without weights (λ = 0.25) | 300 (11.4) | 94.4 / 91.0 | 89.8 ± 0.4 | 38 | 2.9 | 73.9 | 5.4 | 88 |
| 300 | Exact weighted conformal (λ = 0.25) | 300 (11.4) | 90.1 / 93.6 | 97.9 ± 0.2 | 8 | 0.0 | 0.0 | 0.0 | 5 |

## IDRiD

| k | Method | Labelled (urgent) | Coverage no referral / routine | Urgent coverage | Urgent coverage below 90 % (share of draws) | Referable sent home | Healthy sent home | Uncertain | Urgent threshold finite |
|---|---|---|---|---|---|---|---|---|---|
| 20 | C2: random photos | 20 (6.0) | 97.5 / 98.2 | 99.1 ± 0.1 | 4 | 0.0 | 0.0 | 4.1 | 10 |
| 20 | C2-IW, λ = 0.25 (ours) | 20 (6.5) | 98.6 / 98.9 | 99.0 ± 0.1 | 4 | 0.0 | 0.5 | 4.2 | 12 |
| 20 | C2-IW, λ = 0.5 | 20 (6.9) | 98.9 / 98.9 | 98.5 ± 0.2 | 6 | 0.1 | 0.7 | 5.7 | 14 |
| 20 | Rarity sampling without weights (λ = 0.25) | 20 (6.5) | 97.7 / 98.2 | 98.0 ± 0.2 | 8 | 0.1 | 1.7 | 7.1 | 20 |
| 20 | Exact weighted conformal (λ = 0.25) | 20 (6.5) | 97.5 / 98.8 | 99.1 ± 0.1 | 4 | 0.0 | 0.4 | 3.0 | 11 |
| 30 | C2: random photos | 30 (9.2) | 92.4 / 94.0 | 95.2 ± 0.2 | 18 | 1.1 | 15.0 | 13.5 | 59 |
| 30 | C2-IW, λ = 0.25 (ours) | 30 (9.5) | 95.6 / 95.9 | 95.6 ± 0.2 | 17 | 0.7 | 10.1 | 13.4 | 48 |
| 30 | C2-IW, λ = 0.5 | 30 (10.1) | 96.2 / 95.8 | 94.8 ± 0.3 | 20 | 0.8 | 11.4 | 16.0 | 56 |
| 30 | Rarity sampling without weights (λ = 0.25) | 30 (9.5) | 94.5 / 94.6 | 94.0 ± 0.3 | 22 | 1.2 | 16.9 | 15.7 | 64 |
| 30 | Exact weighted conformal (λ = 0.25) | 30 (9.5) | 94.4 / 95.6 | 95.7 ± 0.2 | 16 | 0.5 | 7.6 | 11.1 | 48 |
| 50 | C2: random photos | 50 (15.2) | 92.2 / 93.0 | 92.7 ± 0.2 | 27 | 2.4 | 36.9 | 15.9 | 98 |
| 50 | C2-IW, λ = 0.25 (ours) | 50 (16.0) | 93.1 / 93.3 | 93.0 ± 0.2 | 25 | 2.2 | 35.2 | 15.7 | 96 |
| 50 | C2-IW, λ = 0.5 | 50 (16.8) | 93.4 / 93.2 | 92.8 ± 0.2 | 27 | 2.2 | 35.9 | 16.2 | 98 |
| 50 | Rarity sampling without weights (λ = 0.25) | 50 (16.0) | 93.1 / 92.9 | 92.3 ± 0.3 | 29 | 2.4 | 37.4 | 16.4 | 98 |
| 50 | Exact weighted conformal (λ = 0.25) | 50 (16.0) | 92.9 / 93.4 | 93.2 ± 0.2 | 24 | 2.0 | 33.6 | 15.3 | 95 |
| 75 | C2: random photos | 75 (22.5) | 91.8 / 92.1 | 91.8 ± 0.2 | 32 | 3.0 | 42.7 | 16.1 | 100 |
| 75 | C2-IW, λ = 0.25 (ours) | 75 (23.8) | 92.1 / 92.1 | 92.4 ± 0.2 | 27 | 3.0 | 42.3 | 15.8 | 100 |
| 75 | C2-IW, λ = 0.5 | 75 (25.2) | 92.2 / 92.2 | 91.9 ± 0.2 | 30 | 3.0 | 42.4 | 16.4 | 100 |
| 75 | Rarity sampling without weights (λ = 0.25) | 75 (23.8) | 92.4 / 91.7 | 91.4 ± 0.2 | 32 | 3.3 | 43.8 | 16.7 | 100 |
| 75 | Exact weighted conformal (λ = 0.25) | 75 (23.8) | 92.0 / 92.2 | 92.7 ± 0.2 | 26 | 2.8 | 41.5 | 15.3 | 100 |
| 100 | C2: random photos | 100 (30.2) | 91.1 / 91.4 | 91.5 ± 0.2 | 35 | 3.6 | 45.9 | 15.6 | 100 |
| 100 | C2-IW, λ = 0.25 (ours) | 100 (31.9) | 91.3 / 91.7 | 91.5 ± 0.2 | 35 | 3.5 | 44.9 | 16.1 | 100 |
| 100 | C2-IW, λ = 0.5 | 100 (33.8) | 91.5 / 91.7 | 91.2 ± 0.2 | 37 | 3.5 | 45.0 | 16.4 | 100 |
| 100 | Rarity sampling without weights (λ = 0.25) | 100 (31.9) | 91.7 / 91.4 | 90.7 ± 0.2 | 40 | 3.8 | 46.1 | 16.8 | 100 |
| 100 | Exact weighted conformal (λ = 0.25) | 100 (31.9) | 91.2 / 91.8 | 91.7 ± 0.2 | 34 | 3.4 | 44.5 | 15.8 | 100 |

## λ: validity against efficiency

Each cell: urgent coverage ± Monte Carlo standard error, and in brackets the healthy eyes sent home. The rows marked 'largest weight' give the largest weight relative to the mean weight of the labelled photos (average over draws, at the largest k shown).

| Clinic | k | Rule | λ = 0.25 | λ = 0.5 | λ = 0.75 | λ = 1 |
|---|---|---|---|---|---|---|
| Messidor-2 | 75 | C2-IW (weighted) | 95.1 ± 0.3 (7.0) | 91.3 ± 0.4 (12.1) | 88.0 ± 0.4 (15.7) | 87.3 ± 0.4 (17.2) |
| Messidor-2 | 75 | without weights | 91.4 ± 0.4 (14.0) | 85.2 ± 0.4 (23.3) | 80.5 ± 0.5 (28.8) | 78.2 ± 0.4 (33.8) |
| Messidor-2 | 75 | exact weighted conformal | 98.9 ± 0.1 (0.1) | 97.7 ± 0.1 (0.0) | 96.7 ± 0.1 (0.0) | 96.0 ± 0.1 (0.0) |
| Messidor-2 | 75 | C2, random photos (for reference) | 99.3 ± 0.1 (1.8) | | | |
| Messidor-2 | 100 | C2-IW (weighted) | 92.6 ± 0.4 (13.5) | 89.2 ± 0.4 (19.2) | 88.4 ± 0.4 (21.4) | 87.0 ± 0.4 (23.8) |
| Messidor-2 | 100 | without weights | 88.8 ± 0.4 (20.0) | 83.0 ± 0.4 (28.0) | 80.4 ± 0.4 (32.2) | 79.0 ± 0.4 (36.7) |
| Messidor-2 | 100 | exact weighted conformal | 97.8 ± 0.1 (1.1) | 96.9 ± 0.1 (0.1) | 96.5 ± 0.1 (0.0) | 96.0 ± 0.1 (0.1) |
| Messidor-2 | 100 | C2, random photos (for reference) | 98.1 ± 0.2 (4.7) | | | |
| Messidor-2 | 150 | C2-IW (weighted) | 90.7 ± 0.3 (22.8) | 89.3 ± 0.3 (24.2) | 88.0 ± 0.3 (24.7) | 87.5 ± 0.3 (25.0) |
| Messidor-2 | 150 | without weights | 86.4 ± 0.4 (27.4) | 82.4 ± 0.4 (30.9) | 80.7 ± 0.3 (34.8) | 80.9 ± 0.3 (39.7) |
| Messidor-2 | 150 | exact weighted conformal | 95.4 ± 0.2 (8.2) | 95.6 ± 0.2 (2.8) | 95.6 ± 0.2 (0.7) | 95.6 ± 0.1 (0.3) |
| Messidor-2 | 150 | C2, random photos (for reference) | 94.8 ± 0.3 (14.8) | | | |
| Messidor-2 | 200 | C2-IW (weighted) | 90.5 ± 0.3 (25.6) | 89.8 ± 0.3 (25.8) | 89.3 ± 0.3 (26.1) | 88.9 ± 0.3 (26.2) |
| Messidor-2 | 200 | without weights | 85.6 ± 0.3 (29.1) | 82.6 ± 0.3 (32.4) | 82.6 ± 0.3 (35.9) | 82.9 ± 0.3 (40.7) |
| Messidor-2 | 200 | exact weighted conformal | 93.3 ± 0.2 (18.3) | 93.7 ± 0.2 (12.6) | 94.0 ± 0.2 (5.5) | 94.0 ± 0.2 (1.7) |
| Messidor-2 | 200 | C2, random photos (for reference) | 93.6 ± 0.3 (20.8) | | | |
| Messidor-2 | 300 | C2-IW (weighted) | 90.9 ± 0.2 (26.8) | 90.3 ± 0.2 (27.0) | 90.1 ± 0.2 (27.0) | 89.9 ± 0.2 (27.4) |
| Messidor-2 | 300 | without weights | 86.1 ± 0.3 (29.9) | 85.0 ± 0.3 (32.9) | 85.3 ± 0.2 (36.7) | 84.9 ± 0.2 (40.3) |
| Messidor-2 | 300 | exact weighted conformal | 92.5 ± 0.2 (25.9) | 92.3 ± 0.2 (25.6) | 92.1 ± 0.2 (24.4) | 92.0 ± 0.2 (18.6) |
| Messidor-2 | 300 | C2, random photos (for reference) | 92.2 ± 0.2 (26.0) | | | |
| Messidor-2 | 300 | largest weight / mean weight | 1.2 | 1.4 | 1.6 | 2.1 |
| APTOS resampled to 5 % referable | 150 | C2-IW (weighted) | 99.4 ± 0.1 (1.1) | 95.2 ± 0.3 (23.4) | 89.6 ± 0.4 (64.1) | 88.9 ± 0.4 (80.3) |
| APTOS resampled to 5 % referable | 150 | without weights | 98.2 ± 0.2 (4.5) | 92.3 ± 0.4 (43.2) | 87.7 ± 0.4 (79.7) | 88.0 ± 0.4 (85.5) |
| APTOS resampled to 5 % referable | 150 | exact weighted conformal | 100.0 ± 0.0 (0.0) | 99.8 ± 0.0 (0.0) | 99.2 ± 0.1 (0.0) | 97.3 ± 0.2 (0.0) |
| APTOS resampled to 5 % referable | 150 | C2, random photos (for reference) | 100.0 ± 0.0 (0.0) | | | |
| APTOS resampled to 5 % referable | 200 | C2-IW (weighted) | 97.7 ± 0.2 (12.2) | 91.0 ± 0.4 (62.7) | 90.2 ± 0.3 (81.6) | 91.4 ± 0.3 (84.2) |
| APTOS resampled to 5 % referable | 200 | without weights | 95.9 ± 0.3 (23.0) | 89.4 ± 0.4 (75.7) | 89.0 ± 0.3 (85.7) | 89.7 ± 0.3 (86.9) |
| APTOS resampled to 5 % referable | 200 | exact weighted conformal | 99.9 ± 0.0 (0.0) | 99.0 ± 0.1 (0.0) | 96.8 ± 0.2 (0.0) | 94.4 ± 0.2 (0.0) |
| APTOS resampled to 5 % referable | 200 | C2, random photos (for reference) | 99.9 ± 0.0 (0.0) | | | |
| APTOS resampled to 5 % referable | 300 | C2-IW (weighted) | 91.6 ± 0.4 (62.8) | 91.0 ± 0.3 (82.4) | 92.4 ± 0.3 (83.5) | 92.7 ± 0.3 (83.9) |
| APTOS resampled to 5 % referable | 300 | without weights | 89.8 ± 0.4 (73.9) | 89.7 ± 0.3 (86.1) | 90.4 ± 0.3 (87.0) | 91.5 ± 0.3 (87.2) |
| APTOS resampled to 5 % referable | 300 | exact weighted conformal | 97.9 ± 0.2 (0.0) | 94.2 ± 0.3 (0.0) | 93.3 ± 0.3 (0.0) | 93.2 ± 0.3 (0.1) |
| APTOS resampled to 5 % referable | 300 | C2, random photos (for reference) | 99.4 ± 0.1 (3.4) | | | |
| APTOS resampled to 5 % referable | 300 | largest weight / mean weight | 1.2 | 1.4 | 1.8 | 2.3 |
| IDRiD | 30 | C2-IW (weighted) | 95.6 ± 0.2 (10.1) | 94.8 ± 0.3 (11.4) | 94.0 ± 0.3 (13.3) | 93.0 ± 0.3 (15.2) |
| IDRiD | 30 | without weights | 94.0 ± 0.3 (16.9) | 93.2 ± 0.3 (18.9) | 92.4 ± 0.3 (20.7) | 91.6 ± 0.3 (22.5) |
| IDRiD | 30 | exact weighted conformal | 95.7 ± 0.2 (7.6) | 96.1 ± 0.2 (4.8) | 96.3 ± 0.2 (2.3) | 96.5 ± 0.2 (1.2) |
| IDRiD | 30 | C2, random photos (for reference) | 95.2 ± 0.2 (15.0) | | | |
| IDRiD | 50 | C2-IW (weighted) | 93.0 ± 0.2 (35.2) | 92.8 ± 0.2 (35.9) | 92.5 ± 0.2 (36.3) | 92.2 ± 0.3 (37.2) |
| IDRiD | 50 | without weights | 92.3 ± 0.3 (37.4) | 91.5 ± 0.3 (38.8) | 90.7 ± 0.3 (39.8) | 90.0 ± 0.3 (40.9) |
| IDRiD | 50 | exact weighted conformal | 93.2 ± 0.2 (33.6) | 93.3 ± 0.2 (31.9) | 93.3 ± 0.2 (29.4) | 93.3 ± 0.2 (25.6) |
| IDRiD | 50 | C2, random photos (for reference) | 92.7 ± 0.2 (36.9) | | | |
| IDRiD | 100 | C2-IW (weighted) | 91.5 ± 0.2 (44.9) | 91.2 ± 0.2 (45.0) | 91.1 ± 0.2 (45.3) | 90.9 ± 0.2 (44.9) |
| IDRiD | 100 | without weights | 90.7 ± 0.2 (46.1) | 90.0 ± 0.2 (47.2) | 89.2 ± 0.2 (48.2) | 88.4 ± 0.2 (49.3) |
| IDRiD | 100 | exact weighted conformal | 91.7 ± 0.2 (44.5) | 91.7 ± 0.2 (44.0) | 91.7 ± 0.2 (43.2) | 91.7 ± 0.2 (42.5) |
| IDRiD | 100 | C2, random photos (for reference) | 91.5 ± 0.2 (45.9) | | | |
| IDRiD | 100 | largest weight / mean weight | 1.1 | 1.2 | 1.2 | 1.4 |

Made 2026-11-04 20:31 by label_efficient.py (264 s).
