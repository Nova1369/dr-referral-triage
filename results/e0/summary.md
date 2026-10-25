# E0: standard conformal prediction trained on APTOS, used at other clinics

Target coverage 90% (alpha = 0.1). One threshold for all classes, set on the 15 % APTOS calibration images of each E1 outer fold. Coverage: mean over the 10 folds (lowest–highest fold in brackets). Published reference [8]: 66–77 % coverage on IDRiD at a 90 % target, about 30 % for severe DR.

## lr-grade: 5 grades, LAC score

| Test set | All | Grade 0 none | Grade 1 mild | Grade 2 moderate | Grade 3 severe | Grade 4 proliferative | Mean set size |
|---|---|---|---|---|---|---|---|
| APTOS, held-out fold (n = 367) | 90.5 (88–93) | 98.7 (97–100) | 79.2 (65–92) | 95.6 (91–99) | 51.4 (26–79) | 62.7 (48–70) | 1.40 |
| IDRiD, all 516 (n = 516) | 71.8 (68–76) | 76.2 (73–82) | 74.8 (72–80) | 92.6 (90–96) | 26.5 (14–33) | 70.8 (65–81) | 1.91 |
| IDRiD, official test (103) (n = 103) | 64.4 (57–70) | 59.1 (53–68) | 80.0 (80–80) | 88.8 (81–94) | 29.5 (11–42) | 63.1 (46–77) | 1.97 |
| Messidor-2 (n = 1744) | 62.7 (61–64) | 99.8 (100–100) | 0.5 (0–1) | 17.6 (8–25) | 8.0 (4–12) | 28.6 (14–46) | 1.11 |

## lr-grade: 5 grades, APS score

| Test set | All | Grade 0 none | Grade 1 mild | Grade 2 moderate | Grade 3 severe | Grade 4 proliferative | Mean set size |
|---|---|---|---|---|---|---|---|
| APTOS, held-out fold (n = 367) | 89.9 (87–93) | 83.2 (77–88) | 97.0 (89–100) | 99.4 (97–100) | 82.9 (74–90) | 93.9 (83–100) | 2.58 |
| IDRiD, all 516 (n = 516) | 93.1 (92–95) | 95.6 (94–97) | 98.8 (96–100) | 99.9 (99–100) | 72.6 (68–83) | 96.1 (90–98) | 3.51 |
| IDRiD, official test (103) (n = 103) | 92.2 (90–94) | 94.1 (94–94) | 100.0 (100–100) | 100.0 (100–100) | 69.5 (63–79) | 98.5 (92–100) | 3.66 |
| Messidor-2 (n = 1744) | 78.7 (73–83) | 89.1 (82–93) | 45.4 (40–54) | 82.8 (78–88) | 36.5 (24–48) | 81.1 (74–91) | 2.18 |

## lr: 3 classes (no referral / routine / urgent), LAC score

| Test set | All | Grade 0 none | Grade 1 mild | Grade 2 moderate | Grade 3 severe | Grade 4 proliferative | Mean set size |
|---|---|---|---|---|---|---|---|
| APTOS, held-out fold (n = 367) | 90.4 (87–94) | 99.7 (99–100) | 79.5 (68–89) | 88.9 (80–95) | 69.8 (53–84) | 65.1 (52–77) | 1.22 |
| IDRiD, all 516 (n = 516) | 78.9 (76–83) | 97.7 (96–99) | 96.8 (96–100) | 74.0 (66–79) | 56.2 (45–67) | 68.1 (61–74) | 1.38 |
| IDRiD, official test (103) (n = 103) | 73.3 (70–77) | 96.5 (94–97) | 100.0 (100–100) | 72.2 (66–78) | 54.2 (37–68) | 33.1 (23–46) | 1.36 |
| Messidor-2 (n = 1744) | 75.8 (75–77) | 99.9 (100–100) | 100.0 (100–100) | 5.3 (1–10) | 12.4 (8–24) | 22.0 (17–34) | 1.04 |

## lr: 3 classes (no referral / routine / urgent), APS score

| Test set | All | Grade 0 none | Grade 1 mild | Grade 2 moderate | Grade 3 severe | Grade 4 proliferative | Mean set size |
|---|---|---|---|---|---|---|---|
| APTOS, held-out fold (n = 367) | 90.0 (87–91) | 89.8 (83–93) | 90.0 (81–100) | 95.9 (92–98) | 77.8 (68–89) | 78.6 (66–90) | 1.58 |
| IDRiD, all 516 (n = 516) | 90.3 (89–91) | 98.1 (97–99) | 96.8 (96–100) | 96.7 (92–99) | 74.0 (69–80) | 73.4 (69–76) | 1.88 |
| IDRiD, official test (103) (n = 103) | 86.2 (85–87) | 96.2 (94–97) | 100.0 (100–100) | 98.8 (97–100) | 75.3 (68–79) | 40.0 (31–54) | 1.93 |
| Messidor-2 (n = 1744) | 84.3 (79–90) | 95.1 (92–98) | 98.3 (97–100) | 60.8 (44–77) | 18.0 (15–24) | 35.4 (29–46) | 1.43 |

## svm: 3 classes (no referral / routine / urgent), LAC score

| Test set | All | Grade 0 none | Grade 1 mild | Grade 2 moderate | Grade 3 severe | Grade 4 proliferative | Mean set size |
|---|---|---|---|---|---|---|---|
| APTOS, held-out fold (n = 367) | 90.8 (88–93) | 99.8 (99–100) | 78.4 (73–89) | 90.0 (86–94) | 73.5 (58–80) | 64.7 (47–80) | 1.22 |
| IDRiD, all 516 (n = 516) | 77.3 (73–81) | 98.2 (96–99) | 94.8 (92–96) | 69.5 (64–76) | 53.7 (46–65) | 69.8 (65–79) | 1.34 |
| IDRiD, official test (103) (n = 103) | 71.2 (65–76) | 95.6 (91–97) | 100.0 (100–100) | 67.2 (56–75) | 50.0 (37–63) | 36.9 (31–54) | 1.32 |
| Messidor-2 (n = 1744) | 76.7 (76–78) | 99.9 (100–100) | 100.0 (100–100) | 9.0 (5–15) | 14.7 (9–21) | 25.4 (20–37) | 1.06 |

## svm: 3 classes (no referral / routine / urgent), APS score

| Test set | All | Grade 0 none | Grade 1 mild | Grade 2 moderate | Grade 3 severe | Grade 4 proliferative | Mean set size |
|---|---|---|---|---|---|---|---|
| APTOS, held-out fold (n = 367) | 90.7 (87–93) | 92.5 (86–96) | 87.8 (78–95) | 95.9 (94–97) | 75.8 (65–89) | 75.5 (69–90) | 1.60 |
| IDRiD, all 516 (n = 516) | 88.4 (87–91) | 98.5 (97–99) | 95.2 (92–100) | 92.7 (87–96) | 68.9 (65–74) | 76.3 (73–84) | 1.84 |
| IDRiD, official test (103) (n = 103) | 83.9 (82–86) | 95.6 (91–97) | 100.0 (100–100) | 88.8 (78–94) | 69.5 (63–74) | 56.2 (46–69) | 1.89 |
| Messidor-2 (n = 1744) | 86.6 (84–89) | 97.0 (95–99) | 98.8 (98–100) | 65.3 (58–74) | 22.0 (20–24) | 39.4 (31–46) | 1.52 |

Coverage in %. Grade columns: images of that DR grade, so the urgent class is grades 3 and 4.
Made 2026-10-25 05:59 by e0_conformal.py.
