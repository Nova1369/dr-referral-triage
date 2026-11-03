# Error analysis of the C1 rule (lr, C1, α = 0.1)

APTOS out-of-fold decisions: every photo decided once by a rule fitted without it. 43 of 488 urgent eyes (8.8 %) were not referred urgently, and 63 of 1487 referable eyes (4.2 %) were sent home. Rates by subgroup, with 95 % Wilson intervals; small groups give wide intervals.

## 1. What the missed urgent eyes received

| Decision | Missed urgent eyes | Grade 3 | Grade 4 |
|---|---|---|---|
| routine | 30 | 11 | 19 |
| no referral | 11 | 4 | 7 |
| uncertain | 2 | 1 | 1 |

P(urgent) of the missed urgent eyes: median 0.058 (quartiles 0.027–0.078), against 0.424 (0.233–0.715) for all urgent eyes. The model is fairly sure these eyes are not urgent: they are not borderline cases a slightly lower threshold would catch.

## 2. Urgent eyes not referred urgently, by subgroup

| Subgroup | Group | Eyes | Errors | Rate (95 % CI) |
|---|---|---|---|---|
| DR grade | 3 | 193 | 16 | 8.3 (5.2–13.0) |
| DR grade | 4 | 295 | 27 | 9.2 (6.4–13.0) |
| camera group | H | 15 | 1 | 6.7 (1.2–29.8) |
| camera group | M | 280 | 25 | 8.9 (6.1–12.8) |
| camera group | S | 193 | 17 | 8.8 (5.6–13.7) |
| sharpness | least sharp third | 220 | 23 | 10.5 (7.1–15.2) |
| sharpness | middle third | 174 | 15 | 8.6 (5.3–13.7) |
| sharpness | sharpest third | 94 | 5 | 5.3 (2.3–11.9) |
| quality warning | any quality warning | 39 | 2 | 5.1 (1.4–16.9) |
| quality warning | no warning | 449 | 41 | 9.1 (6.8–12.2) |
| conflicting copy | copy with another class | 21 | 2 | 9.5 (2.7–28.9) |
| conflicting copy | no conflicting copy | 467 | 41 | 8.8 (6.5–11.7) |

## 3. Referable eyes sent home, by subgroup

| Subgroup | Group | Eyes | Errors | Rate (95 % CI) |
|---|---|---|---|---|
| DR grade | 2 | 999 | 52 | 5.2 (4.0–6.8) |
| DR grade | 3 | 193 | 4 | 2.1 (0.8–5.2) |
| DR grade | 4 | 295 | 7 | 2.4 (1.2–4.8) |
| camera group | H | 55 | 26 | 47.3 (34.7–60.2) |
| camera group | M | 717 | 27 | 3.8 (2.6–5.4) |
| camera group | S | 715 | 10 | 1.4 (0.8–2.6) |
| sharpness | least sharp third | 723 | 24 | 3.3 (2.2–4.9) |
| sharpness | middle third | 498 | 14 | 2.8 (1.7–4.7) |
| sharpness | sharpest third | 266 | 25 | 9.4 (6.4–13.5) |
| quality warning | any quality warning | 100 | 4 | 4.0 (1.6–9.8) |
| quality warning | no warning | 1387 | 59 | 4.3 (3.3–5.4) |
| conflicting copy | copy with another class | 45 | 3 | 6.7 (2.3–17.9) |
| conflicting copy | no conflicting copy | 1442 | 60 | 4.2 (3.2–5.3) |

## 4. Reading the tables

- **The camera is the largest effect.** On the mostly-healthy cameras (H), 26 of 55 referable eyes are sent home, 25 of them grade 2. This is the camera shortcut studied in the additional work: the class-wise guarantee holds on average over cameras, not within each camera.
- **Grade.** Referable eyes sent home are mostly grade 2: 5.2 % (52 of 999), against 2.1 % (4 of 193) for grade 3 and 2.4 % (7 of 295) for grade 4. Urgent eyes are missed about equally often in grade 3 (8.3 % (16 of 193)) and grade 4 (9.2 % (27 of 295)), mostly by being called routine.
- **Sharpness is tangled with the camera.** Referable eyes are sent home most often in the sharpest third (9.4 % (25 of 266)), but 21 of those errors are H-camera eyes (15 % of the referable eyes in that third come from H cameras, against 0 % in the least sharp third). Urgent eyes are missed somewhat more often in blurrier photos (10.5 % (23 of 220) against 5.3 % (5 of 94)); the intervals overlap.
- **Quality warnings and conflicting copies** show no clear difference; the groups are small and the intervals overlap.
- **The misses are confident.** Missed urgent eyes have a median P(urgent) of 0.058, so a slightly lower threshold would not catch them; they look non-urgent to the model.

`missed_urgent.csv` lists every missed urgent eye with its photo path, for looking at the photos (keep them on your own computer: the licences do not allow sharing them).

Made 2026-11-03 15:59 by error_analysis.py.
