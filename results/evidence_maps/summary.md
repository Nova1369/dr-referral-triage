# Additional work: where in the photo does the model find its evidence?

Logistic regression is linear in ResNet-50's average-pooled features, so each logit splits exactly into a 7 × 7 map over the photo (class activation mapping): cell (i, j) contributes w · F(i, j) / 49, where F(i, j) is the last convolutional layer at that cell. APTOS photos are explained by the E1 fold model that never trained on them (and by the camera-decorrelated model of the same fold); IDRiD and Messidor-2 photos by the all-APTOS models of E2. The maps below are for log-odds against no referral (routine − no referral, urgent − no referral).

## 1. Checks

- Linear heads rebuilt from the saved settings: largest difference from the saved probabilities 4.0e-03 (E1 folds 4.0e-03, decorrelated folds 1.0e-06, all-APTOS models 9.9e-07).
- Features recomputed from the photos against the saved features: largest difference 2.9e-06.
- Maps against the logits: the 49 cells plus the bias reproduce every logit computed from the saved features to within 4.2e-07.

## 2. Where the evidence lies

Cells are inside the retina (≥ 99 % of the cell on the retina), on its edge, or outside it (black). Mean log-odds of grade-2 (routine) eyes for routine against no referral, split by region:

| Photos | Model | Eyes | Log-odds | Bias | Inside | Edge | Outside | Cells inside / edge / outside |
|---|---|---|---|---|---|---|---|---|
| APTOS, H cameras | E1's logistic regression | 40 | -1.36 | -1.32 | +0.15 | -0.05 | -0.14 | 25 / 20 / 4 |
| APTOS, H cameras | camera-decorrelated model | 40 | +0.83 | -1.31 | +1.11 | +0.66 | +0.38 | 25 / 20 / 4 |
| APTOS, M cameras | E1's logistic regression | 437 | +1.60 | -1.31 | +2.26 | +0.39 | +0.25 | 24 / 20 / 4 |
| APTOS, M cameras | camera-decorrelated model | 437 | +1.43 | -1.35 | +2.10 | +0.38 | +0.31 | 24 / 20 / 4 |
| APTOS, S cameras | E1's logistic regression | 522 | +2.69 | -1.32 | +2.88 | +0.88 | +0.25 | 24 / 20 / 5 |
| APTOS, S cameras | camera-decorrelated model | 522 | +1.88 | -1.37 | +2.50 | +0.59 | +0.16 | 24 / 20 / 5 |
| IDRiD | E1's logistic regression | 168 | +0.78 | -1.42 | +1.38 | +0.87 | -0.06 | 25 / 20 / 4 |
| IDRiD | camera-decorrelated model | 168 | -0.33 | -1.26 | +0.80 | +0.36 | -0.24 | 25 / 20 / 4 |
| Messidor-2 | E1's logistic regression | 347 | -3.07 | -1.42 | -1.32 | -0.36 | +0.03 | 25 / 20 / 4 |
| Messidor-2 | camera-decorrelated model | 347 | -2.13 | -1.26 | -1.23 | -0.09 | +0.44 | 25 / 20 / 4 |

The gap between grade-2 eyes on M and on H cameras (M minus H), by region:

| Model | Log-odds gap | From inside cells | From edge cells | From outside cells | From the bias |
|---|---|---|---|---|---|
| E1's logistic regression | +2.96 | +2.11 | +0.44 | +0.39 | +0.01 |
| camera-decorrelated model | +0.60 | +0.99 | -0.29 | -0.07 | -0.04 |

For E1's model, 71 % of the gap comes from cells inside the retina and 28 % from its edge and the black corners.

Mean log-odds of grade-0 eyes (routine against no referral), the same split:

| Photos | Model | Eyes | Log-odds | Inside | Edge | Outside |
|---|---|---|---|---|---|---|
| APTOS, H cameras | E1's logistic regression | 1531 | -4.55 | -2.31 | -0.78 | -0.13 |
| APTOS, H cameras | camera-decorrelated model | 1531 | -4.14 | -2.54 | -0.49 | +0.25 |
| APTOS, M cameras | E1's logistic regression | 272 | -3.24 | -2.29 | -0.16 | +0.51 |
| APTOS, M cameras | camera-decorrelated model | 272 | -4.08 | -3.19 | -0.25 | +0.71 |
| APTOS, S cameras | E1's logistic regression | 2 | +0.54 | +0.07 | +1.06 | +0.38 |
| APTOS, S cameras | camera-decorrelated model | 2 | -0.98 | -0.62 | +0.50 | +0.23 |
| IDRiD | E1's logistic regression | 168 | -2.40 | -1.59 | +0.66 | -0.05 |
| IDRiD | camera-decorrelated model | 168 | -3.86 | -2.46 | +0.10 | -0.24 |
| Messidor-2 | E1's logistic regression | 1017 | -4.09 | -2.15 | -0.56 | +0.04 |
| Messidor-2 | camera-decorrelated model | 1017 | -3.35 | -2.24 | -0.29 | +0.44 |

## 3. Do the maps point at what matters? (deletion test)

150 APTOS urgent eyes (random sample). The cells with the most urgent evidence are replaced by a blurred copy of the photo (Gaussian, radius 8 px at 224 px: lesions disappear, colour and brightness stay), and the urgent log-odds recomputed. Controls (3 random orders per eye each): random cells anywhere, and random cells on the retina first (blurring a black corner changes nothing). Mean over eyes; 95 % bootstrap CI of the difference from the evidence order.

| Cells blurred (of 49) | Drop in urgent log-odds: evidence order | Random cells | Random retina cells | Evidence minus retina control (95 % CI) | Mean P(urgent): evidence / random / retina |
|---|---|---|---|---|---|
| 0 | 0.00 | -0.00 | -0.00 | 0.00 (-0.00 to 0.00) | 0.46 / 0.46 / 0.46 |
| 1 | 0.52 | 0.08 | 0.07 | 0.45 (0.32 to 0.56) | 0.43 / 0.46 / 0.46 |
| 2 | 0.94 | 0.16 | 0.22 | 0.72 (0.55 to 0.89) | 0.40 / 0.46 / 0.45 |
| 4 | 1.51 | 0.41 | 0.56 | 0.95 (0.74 to 1.15) | 0.37 / 0.45 / 0.42 |
| 8 | 2.79 | 1.01 | 1.54 | 1.24 (0.99 to 1.49) | 0.31 / 0.43 / 0.38 |
| 16 | 5.29 | 2.35 | 4.21 | 1.08 (0.68 to 1.50) | 0.16 / 0.37 / 0.22 |
| 24 | 7.53 | 4.30 | 7.58 | -0.05 (-0.56 to 0.47) | 0.08 / 0.25 / 0.06 |

## 4. A fixed hotspot, and mirrored eyes

One cell, in the second row and second column of the 7 × 7 grid (top left, just inside the retina), carries strong evidence against referral in every dataset, while the cells in the mirror positions do not. To see whether it belongs to the eyes or to the network, 600 photos (200 per dataset) were mirrored and explained again. Mean evidence for routine against no referral, in the cell at each position of the frame:

| Photos | Top left (row 2, col 2) | Top right (row 2, col 6) | Bottom left (row 6, col 2) | Bottom right (row 6, col 6) |
|---|---|---|---|---|
| original | -0.89 | +0.12 | -0.00 | -0.04 |
| mirrored left-right | -0.88 | +0.13 | -0.01 | -0.05 |
| mirrored top-bottom | -0.90 | +0.08 | -0.01 | -0.05 |

The hotspot stays in the top-left cell of the frame when the eye is mirrored: it belongs to the network, not to the eye. A plausible cause is the asymmetric zero padding of ResNet-50's strided convolutions, which treats the top and left borders differently from the bottom and right ones (Alsallakh et al., ICLR 2021); here the dark corner of the photo meets that border. It adds about the same amount to every photo (SD 0.30 over all 5922 photos, against 3.37 for the whole log-odds), so it acts mostly like a shift of the bias.

Does the rest of each map follow the eye? Median correlation, over the photos, between a photo's map and the map of its mirror image, flipped back (the evidence follows the eye) or as it is (it stays in the frame); hotspot positions left out:

| Mirror | Cells | Flipped back (follows the eye) | As it is (stays in the frame) |
|---|---|---|---|
| left-right | inside the retina | 0.44 | 0.24 |
| left-right | edge of the retina | 0.31 | 0.57 |
| top-bottom | inside the retina | 0.22 | 0.44 |
| top-bottom | edge of the retina | 0.15 | 0.61 |

The maps mix two parts. Inside the retina, under the left-right mirror (which moves the optic disc to the other side), the evidence follows the eye (0.44 against 0.24), as evidence about the retina should. On the edge of the retina it stays in the frame (0.57 against 0.31): the outline of the retina, which mirroring does not change, drives it. Under the top-bottom mirror the frame-fixed part is larger even inside the retina. So a single photo's map should be read with care; the averages over many eyes (section 2) and the deletion test (section 3) are the reliable evidence.

How much do the predictions change when an eye is mirrored (left and right eyes are mirror images)?

| Photos | Mirror | Mean change in P(referable) | Largest change | Most likely class changes |
|---|---|---|---|---|
| APTOS (200) | left-right | 0.050 | 0.39 | 9.0 % |
| APTOS (200) | top-bottom | 0.071 | 0.52 | 10.5 % |
| IDRiD (200) | left-right | 0.085 | 0.42 | 14.0 % |
| IDRiD (200) | top-bottom | 0.108 | 0.55 | 23.0 % |
| Messidor-2 (200) | left-right | 0.030 | 0.42 | 1.5 % |
| Messidor-2 (200) | top-bottom | 0.038 | 0.36 | 2.0 % |

C1's decision changes for 24 of the 200 APTOS eyes when they are mirrored left-right, and for 19 when mirrored top-bottom.


Made 2026-11-04 11:13 by evidence_maps.py (1 s).
