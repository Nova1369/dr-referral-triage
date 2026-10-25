# k-means check: do the images group by grade or by clinic?

ResNet-50 features, standardised, PCA to 50 components (61% of the variance for all images), k-means with 20 restarts. ARI: 0 = chance, 1 = the clusters are the labels. NMI: 0–1.

## all images

| k | ARI dataset | ARI class | ARI grade | NMI dataset | NMI class | NMI grade |
|---|---|---|---|---|---|---|
| 2 | 0.210 | 0.216 | 0.228 | 0.342 | 0.170 | 0.169 |
| 3 | 0.139 | 0.259 | 0.278 | 0.292 | 0.174 | 0.193 |
| 4 | 0.143 | 0.109 | 0.144 | 0.256 | 0.147 | 0.168 |
| 5 | 0.106 | 0.136 | 0.164 | 0.237 | 0.155 | 0.173 |
| 6 | 0.102 | 0.081 | 0.106 | 0.225 | 0.146 | 0.166 |

## APTOS only

| k | ARI class | ARI grade | ARI camera (photo size) | NMI class | NMI grade | NMI camera (photo size) |
|---|---|---|---|---|---|---|
| 2 | 0.187 | 0.276 | 0.279 | 0.283 | 0.334 | 0.460 |
| 3 | 0.259 | 0.380 | 0.371 | 0.270 | 0.356 | 0.506 |
| 4 | 0.298 | 0.416 | 0.383 | 0.264 | 0.344 | 0.456 |
| 5 | 0.153 | 0.238 | 0.469 | 0.237 | 0.314 | 0.519 |
| 6 | 0.161 | 0.253 | 0.497 | 0.224 | 0.311 | 0.526 |

## Clusters (k = 2, all images) by dataset

| Cluster | APTOS | IDRiD | Messidor-2 |
|---|---|---|---|
| 0 | 2192 | 515 | 4 |
| 1 | 1470 | 1 | 1740 |

## Clusters (k = 3, all images) by dataset

| Cluster | APTOS | IDRiD | Messidor-2 |
|---|---|---|---|
| 0 | 522 | 56 | 3 |
| 1 | 1675 | 452 | 4 |
| 2 | 1465 | 8 | 1737 |

## Clusters (k = 3, all images) by class

| Cluster | no referral (0–1) | routine (2) | urgent (3–4) |
|---|---|---|---|
| 0 | 378 | 148 | 55 |
| 1 | 572 | 980 | 579 |
| 2 | 2705 | 386 | 119 |

## APTOS by camera (photo size)

Share of referable and urgent eyes per photo size, and share of the photos that k-means (k = 2, all images) puts in the same cluster as almost all Messidor-2 photos.

| Photo size | Images | Referable | Urgent | In the Messidor-2 cluster |
|---|---|---|---|---|
| 2048x1536 | 351 | 0% | 0% | 100% |
| 819x614 | 287 | 0% | 0% | 52% |
| 1050x1050 | 974 | 6% | 2% | 100% |
| 2588x1958 | 533 | 48% | 17% | 0% |
| 2416x1736 | 638 | 58% | 22% | 0% |
| other | 236 | 80% | 37% | 0% |
| 3388x2588 | 141 | 91% | 26% | 0% |
| 3216x2136 | 410 | 96% | 21% | 0% |
| 1504x1000 | 92 | 98% | 29% | 0% |

Made 2026-10-25 06:02 by kmeans_check.py.
