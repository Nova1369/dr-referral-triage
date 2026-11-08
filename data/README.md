# Data

No dataset files are in this repository. All three datasets are public but come under their own terms,
and the Messidor-2 licence forbids redistributing its images, including processed copies. Download them
yourself and place them under `datasets/` (ignored by git), or edit the paths at the top of `run_all.sh`.

| Dataset | Where to get it | Put it at | What the code reads |
|---|---|---|---|
| APTOS 2019 Blindness Detection [1] | https://www.kaggle.com/competitions/aptos2019-blindness-detection/data (accept the competition rules) | `datasets/aptos2019-blindness-detection/` | `train.csv`, `train_images/*.png` (3,662 photos) |
| IDRiD, part B "Disease Grading" [2] | https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid | `datasets/IDRiD/` | the train/test images and the two grading CSVs (any folder nesting) |
| Messidor-2 images [15, 18] | https://www.adcis.net/en/third-party/messidor2/ (licence form) | `datasets/Messidor - 2/IMAGES/` | the 1,748 images and `messidor-2.csv` (exam list) |
| Messidor-2 grades, Krause et al. [16] | https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades | `datasets/Messidor - 2/messidor_data.csv` | adjudicated DR grade and gradability |

Messidor-2 images: Kindly provided by the Messidor program partners (see https://www.adcis.net/en/third-party/messidor/).
Messidor-2 grades: Krause et al. [16].

## What step 1 makes from them (not committed)

`src/dr_preprocess.py` writes `data_512/`: 512 x 512 cropped photos, `manifest.csv` (one row per photo: dataset,
grade, 3-class label `ref3`, fold, duplicate group, quality features) and, after step 2, `features/*.npz`
(2,048 ResNet-50 features per photo). These derived files contain or identify licensed images, so they stay
local. Expected counts: 5,922 gradable photos = 3,662 APTOS + 516 IDRiD + 1,744 Messidor-2.

References are numbered as in the report (`report/main.pdf`).
