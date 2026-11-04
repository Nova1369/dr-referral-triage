# Additional work: each baseline's setting swept, compared at matched fairness

Same protocol as group_robust.py (E1's 10 outer folds; C1 on each fold's calibration photos; every APTOS photo decided out of fold). Both targets in every camera group: referable eyes sent home at most 10 %, urgent eyes referred urgently at least 90 %. * = the standard setting used in group_robust.py. In %.

## 1. Best setting of each family that meets both targets in every camera group

Chosen with hindsight on these results, which favours the baselines; ours is shown at its principled setting (t = 1). Last column: exact McNemar on all 2175 healthy eyes, sent home or not, against ours at t = 1 (eyes only that setting sends home / eyes only ours sends home).

| Family | Best setting | Healthy sent home | Worst group: referable sent home | Worst group: urgent referred urgently | AUROC referable | Against ours (t = 1) |
|---|---|---|---|---|---|---|
| Plain training (ERM) | no setting meets both targets | | | | | |
| Camera-decorrelated weights (ours) | t = 1 | 84.3 | 9.1 (H) | 91.1 | 0.965 | – |
| Group-balanced weights | t = 0.75 | 80.5 | 9.1 (H) | 91.8 | 0.969 | 10 / 92, p = 9.4e-18 |
| Group DRO | step 0.05 | 76.0 | 5.5 (H) | 92.1 | 0.965 | 0 / 179, p = 2.6e-54 |
| JTT | no setting meets both targets | | | | | |
| DFR | C = 0.3 | 27.7 | 4.1 (S) | 91.2 | 0.873 | 10 / 1240, p = 0 |

## 2. Every setting

| Family | Setting | Worst group: referable sent home (95 % CI) | Worst group: urgent referred urgently | Healthy sent home | AUROC referable | Both targets in every group |
|---|---|---|---|---|---|---|
| Plain training (ERM) | –* | 43.6 (31–57), H | 91.8 | 87.6 | 0.974 | no |
| Camera-decorrelated weights (ours) | t = 0.25 | 36.4 (25–50), H | 92.1 | 87.4 | 0.974 | no |
| Camera-decorrelated weights (ours) | t = 0.5 | 23.6 (14–36), H | 92.1 | 86.2 | 0.972 | no |
| Camera-decorrelated weights (ours) | t = 0.75 | 12.7 (6–24), H | 91.8 | 85.1 | 0.970 | no |
| Camera-decorrelated weights (ours) | t = 1* | 9.1 (4–20), H | 91.1 | 84.3 | 0.965 | yes |
| Camera-decorrelated weights (ours) | t = 1.5 | 8.7 (7–11), S | 91.8 | 71.9 | 0.940 | yes |
| Group-balanced weights | t = 0.25 | 34.5 (23–48), H | 91.1 | 86.0 | 0.974 | no |
| Group-balanced weights | t = 0.5 | 21.8 (13–34), H | 91.4 | 83.3 | 0.972 | no |
| Group-balanced weights | t = 0.75 | 9.1 (4–20), H | 91.8 | 80.5 | 0.969 | yes |
| Group-balanced weights | t = 1* | 5.5 (2–15), H | 91.8 | 76.7 | 0.965 | yes |
| Group DRO | step 0.05 | 5.5 (2–15), H | 92.1 | 76.0 | 0.965 | yes |
| Group DRO | step 0.1 | 5.5 (2–15), H | 91.8 | 75.3 | 0.965 | yes |
| Group DRO | step 0.5* | 3.6 (1–12), H | 91.4 | 70.7 | 0.963 | yes |
| Group DRO | step 1 | 3.6 (1–12), H | 91.4 | 67.4 | 0.961 | yes |
| Group DRO | step 2 | 3.6 (1–12), H | 91.4 | 65.0 | 0.959 | yes |
| JTT | ×2 | 38.2 (27–51), H | 92.1 | 85.3 | 0.974 | no |
| JTT | ×5* | 21.8 (13–34), H | 86.7 | 81.4 | 0.971 | no |
| JTT | ×10 | 18.2 (10–30), H | 86.7 | 75.2 | 0.966 | no |
| JTT | ×20* | 7.3 (3–17), H | 86.7 | 63.5 | 0.956 | no |
| JTT | ×50 | 1.1 (1–2), S | 66.7 | 35.6 | 0.930 | no |
| DFR | C = 0.003 | 3.2 (2–5), M | 91.7 | 22.4 | 0.884 | yes |
| DFR | C = 0.01 | 3.4 (2–5), S | 90.2 | 25.0 | 0.880 | yes |
| DFR | C = 0.03* | 3.9 (3–6), S | 91.2 | 26.4 | 0.877 | yes |
| DFR | C = 0.1 | 4.1 (3–6), S | 91.7 | 27.1 | 0.875 | yes |
| DFR | C = 0.3 | 4.1 (3–6), S | 91.2 | 27.7 | 0.873 | yes |

Made 2026-11-04 20:12 by group_frontier.py (539 s).
