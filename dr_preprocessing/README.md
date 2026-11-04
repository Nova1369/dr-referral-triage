# Project code: preprocessing, features, models, the referral rules C1–C3 and experiments E0–E3

The scripts, run in this order:

| Step | Script | What it makes |
|---|---|---|
| 1 | `dr_preprocess.py` | clean 512 × 512 images from the raw APTOS 2019, IDRiD and Messidor-2 downloads, plus one table (`manifest.csv`) that every later step reads (sections 1–7) |
| 2 | `extract_features.py` | 2,048 ResNet-50 features per image, the input to the course models (section 8) |
| 3 | `run_models.py`, `summarize_e1.py` | E1: baselines B0–B2 and the 11 course models, 10-fold cross-validation on APTOS, with tables and a figure (section 9) |
| 4 | `e0_conformal.py` | E0: standard conformal prediction calibrated on APTOS, tested on IDRiD and Messidor-2, as in [8] (section 10) |
| 5 | `kmeans_check.py` | k-means: do the images group by grade, or by clinic and camera? (section 11) |
| 6 | `c1_referral.py` | C1: the referral rule with one conformal threshold per class, against the simpler rules B3, B4 and B5 (section 12) |
| 7 | `e2_new_clinics.py` | C2 and E2: the thresholds refitted at IDRiD and Messidor-2 from k labelled local photos, against B6 and B7 (section 13) |
| 8 | `c3_prevalence.py` | C3 and E3: the rule at lower prevalence, and the workload forecast from unlabelled photos, against B8 and B9 (section 14) |
| 9 | `error_analysis.py` | which eyes C1 gets wrong, by grade, camera, image quality and label conflicts (section 15) |
| 10 | `extract_variants.py`, `ablations.py` | ablations (512 px, Graham filter, PCA size) and robustness (blur, dark, JPEG) (section 16) |
| extra | `camera_shortcut.py`, `camera_fixes.py`, `evidence_maps.py`, `mirror_fix.py`, `group_robust.py`, `group_frontier.py`, `label_efficient.py` | additional work, beyond the proposal: the APTOS camera shortcut, fixes tuned under the E1 protocol, where in the photo the model finds its evidence, mirror-averaged features, the camera-group guarantee against strong baselines, and label-efficient recalibration at a new clinic (sections 17, 18 and 21–24) |

`run_all.sh` runs every step in order (`bash run_all.sh`, or `bash run_all.sh 3` to start at step 3); edit the paths at its top first. On Windows, run its lines one by one.

**Step 1.** For each photo, `dr_preprocess.py`:

1. finds the circular retina (field of view), including photos where the circle is cut off at the top and bottom, as in many APTOS images;
2. crops a square around that circle, resizes it to 512 px, and blacks out everything outside it (this also removes camera notches and text in the border);
3. computes simple image-quality features (brightness, contrast, sharpness, colour, over- and under-exposure);
4. computes a small signature used to find duplicate photos.

For the whole set it then:

- groups duplicates (APTOS has no patient IDs, and some photos appear more than once, sometimes with different grades);
- maps grades 0–4 to our three classes (0–1 no referral, 2 routine, 3–4 urgent);
- assigns every APTOS image to one of 10 cross-validation folds. The folds are stratified by grade, copies of the same photo always stay in the same fold, and the folds come out identical on every machine.

Colours are **not** changed in the saved images. CLAHE and Ben Graham's normalisation are included as functions; Graham's is compared as an ablation in section 16.

## 1. Get the data

| Dataset | Where | What you need |
|---|---|---|
| APTOS 2019 | [Kaggle competition](https://www.kaggle.com/competitions/aptos2019-blindness-detection). Accept the rules, then download (≈10 GB) or open a Kaggle notebook with the data attached. | `train.csv` and `train_images/` (the test images are not used: their labels were never released) |
| IDRiD | [IEEE DataPort](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) | `B. Disease Grading` (images and the two label CSVs). The segmentation part is not needed. |
| Messidor-2 images | [ADCIS](https://www.adcis.net/en/third-party/messidor2/). Fill in the licence form. | the `IMAGES` folder |
| Messidor-2 grades | [Kaggle: MESSIDOR-2 DR Grades](https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades) (Krause et al. 2018) | the CSV in that download (`messidor_data.csv`) |

**Messidor-2 licence:** the images may not be redistributed, and that includes our 512 px versions. Keep them off public Kaggle datasets, public notebooks and GitHub. Each of us who needs them should download them from ADCIS directly.

Unzip everything, keeping the original folder names. The script searches inside the folder you give it, so the exact nesting does not matter (`B. Disease Grading/B. Disease Grading/...` is fine). The `data/idrid/grading/train/images` layout from our earlier `organize.py` also works.

## 2. Install

```bash
pip install -r requirements.txt        # numpy, pandas, opencv-python-headless, Pillow
```

Kaggle and Colab already have everything.

## 3. Run

**First, a trial run on 20 images per split** (about a minute). Then open the `qc/` sheets (section 5).

```bash
python dr_preprocess.py --aptos PATH/aptos2019-blindness-detection --idrid "PATH/IDRiD" \
  --messidor PATH/messidor-2 --messidor-labels PATH/messidor_data.csv --out trial --limit 20
```

**Then the full run.** Same command, without `--limit`, into the real output folder:

```bash
python dr_preprocess.py --aptos PATH/aptos2019-blindness-detection --idrid "PATH/IDRiD" \
  --messidor PATH/messidor-2 --messidor-labels PATH/messidor_data.csv --out data_512
```

- **Time:** roughly 10–30 minutes for all ~5,900 photos on a 4-core laptop. Decoding the large APTOS PNGs takes most of it.
- **Disk:** about 1.4 GB as PNG (lossless). `--format jpg` gives about 0.3 GB.
- **Resuming:** if the run stops (Colab timeout, laptop sleep), run the same command again. Finished images are cached in `data_512/cache.jsonl` and skipped. Where a session has a time limit, add `--time-budget 1500` (seconds). The script then stops cleanly after that time, and running it again continues.
- **Partial runs:** you can give only the datasets you have so far, for example only `--aptos`. Running again later with all three reuses the finished images.
- **Windows:** put paths with spaces in quotes, as above.
- **Kaggle:** add the APTOS competition data to a notebook, upload the script, then run:
  ```
  !python dr_preprocess.py --aptos /kaggle/input/aptos2019-blindness-detection --out /kaggle/working/data_512
  ```
  For IDRiD, upload it as a private dataset. Process Messidor-2 on your own machine (licence).
- **Colab:** copy the zip files to `/content` and unzip them there before running. Reading thousands of files straight from Google Drive is very slow.

`python dr_preprocess.py --help` lists every option. Useful ones:

- `--size` (default 512)
- `--workers` (number of parallel processes)
- `--dup-threshold` (default 0.95: copies of the same photo)
- `--group-threshold` (default 0.80: probably the same eye; only used to keep images together in splits)
- `--folds` (default 10)
- `--seed` (default 42)
- `--time-budget` (seconds; 0 = no limit)

**Run it once and share `manifest.csv` with the team.** Then all three of us use exactly the same images, groups and folds.

## 4. Outputs (in `data_512/`)

| File | What it is |
|---|---|
| `images/<dataset>/<split>/<name>.png` | 512 × 512 processed photos, retina centred, black outside the circle |
| `manifest.csv` | one row per usable image: the table all later steps read |
| `excluded.csv` | every label row that was not used, with the reason (file missing, ungradable, unreadable, blank) |
| `duplicates.csv` | all image pairs with signature similarity ≥ 0.5; `is_copy = True` (≥ 0.95) for copies of the same photo, `grouped = True` (≥ 0.80) for pairs kept together in splits |
| `summary.md` | counts per dataset, class shares, circle-fit statistics, duplicates, fold balance |
| `qc/*.jpg` | contact sheets for checking by eye (section 5) |

Main `manifest.csv` columns:

| Column | Meaning |
|---|---|
| `uid` | `dataset/split/name`, unique. IDRiD train and test both contain an `IDRiD_001`. |
| `dataset`, `split` | `aptos/train`, `idrid/train`, `idrid/test` (the official 103-image test split), `messidor2/test` |
| `grade`, `ref3`, `ref3_name`, `referable` | DR grade 0–4; our class 0/1/2 = no referral / routine / urgent; grade ≥ 2 |
| `dme` | macular oedema label (IDRiD 0–2, Messidor-2 0/1). Kept, but not used for our classes. |
| `fold` | APTOS cross-validation fold 0–9 (−1 for the other datasets) |
| `group_id` | images that must stay on the same side of any split: copies of one photo, photos that are probably the same eye (similarity ≥ 0.80), and for Messidor-2 the two eyes of one examination (if the exam list was found) |
| `dup_cluster`, `dup_cluster_size` | copies of the same photo; size 1 = no copy found |
| `dup_max_sim` | similarity to the most similar other image (≥ 0.95 = copy) |
| `dup_grade_conflict`, `dup_class_conflict` | the copies of this photo carry different grades / classes |
| `out_path` | processed image, relative to the output folder |
| `orig_w`, `orig_h` | original photo size. This is a rough proxy for the camera, useful for the fairness-by-site analysis. |
| `fov_method`, `fov_iou`, `fov_cx`, `fov_cy`, `fov_r` | how the circle was found (`fit` normally) and how well it matches the retina (1 = perfect); centre and radius in original pixels |
| `q_*` | quality features inside the retina: share of the circle inside the photo, brightness, contrast, mean R/G/B, dark share, saturated share, sharpness (Laplacian variance), histogram entropy. These feed baseline B2 (image quality only). |
| `qc_flags` | warnings; the image is kept. `no_fov`, `fov_check`, `cut_off`, `dark`, `overexposed`, `low_sharpness` (lowest 2 % in its dataset), `upscaled` |

## 5. Check before moving on (10 minutes, worth it)

1. **`summary.md` → Labels:** every split should say *matches published counts*. If not, a label file or image folder is wrong or incomplete.
2. **`qc/samples_<dataset>.jpg`:** the green circle should follow the edge of the retina in each raw photo.
3. **`qc/flagged.jpg`:** look at the worst circle fits and the dark or unusual photos. Note any that are unusable.
4. **`qc/duplicates.jpg`:**
   - The *COPY* pairs should be the same photo.
   - The *SAME EYE?* pairs are only kept together in splits. If some are actually different eyes, that does no harm.
   - The *not grouped* pairs should be different photos.
   - If you disagree, change `--dup-threshold` or `--group-threshold` and re-run; it is quick, because the images are cached.
5. **Cross-dataset copies:** `summary.md` reports pairs across datasets. Any APTOS photo that also appears in IDRiD or Messidor-2 would leak into the external tests, so we would drop it from APTOS training.
6. **Copies with conflicting grades:** decide one rule (keep all, majority grade, or drop) and state it in the report.

## 6. Using the outputs

```python
import numpy as np
import pandas as pd
from PIL import Image
import dr_preprocess as dp                            # this script, in the same folder

m = pd.read_csv("data_512/manifest.csv")
aptos = m[m.dataset == "aptos"]

for k in range(10):                                   # E1: 10-fold cross-validation on APTOS
    train, test = aptos[aptos.fold != k], aptos[aptos.fold == k]
    # inner 5-fold tuning and the 15% calibration hold-out: split `train` by group_id
    # (e.g. sklearn GroupKFold / GroupShuffleSplit) so copies never cross the split

arr = np.array(Image.open("data_512/" + aptos.out_path.iloc[0]))   # 512 x 512 x 3, uint8 RGB
retina = dp.valid_mask(arr)                           # True inside the retina
arr_clahe, arr_graham = dp.enhance_clahe(arr), dp.enhance_graham(arr)   # optional, for the ablation
```

For E2:

- train on all of APTOS;
- test on `idrid` (train + test, and separately the official `idrid/test` split) and on `messidor2`;
- when drawing the *k* labelled local images for recalibration (C2), sample whole `group_id`s.

## 7. Choices to describe in the mid report

- **Crop:** a square around the fitted retina circle, so the retina diameter is the same (512 px) in every image whatever the camera. The circle is found by thresholding the black border and running a RANSAC circle fit that ignores straight cut-off edges and notches. On all 5,922 photos the fitted circle was used for every image except 16 APTOS photos, where a circle from the bounding box fitted better. The median overlap (IoU) between the circle and the retina mask was 0.995 (APTOS), 0.997 (IDRiD) and 0.993 (Messidor-2).
- **No colour normalisation in the saved images.** CLAHE and Graham are kept as optional variants for the ablation.
- **Duplicates:** cosine similarity of a 32 × 32 high-pass green-channel signature from the centre of the retina.
  - **Copies** (similarity ≥ 0.95): in our tests, copies that were re-saved as JPEG, halved in size, brightened or framed differently scored 0.97–1.00. On the real data, copies scored ≥ 0.98, and no pair fell between 0.95 and 0.98.
  - **Probably the same eye** (0.80–0.95): these are mostly the same eye photographed again, plus some different eyes that look alike, because the optic disc and macula sit in similar places. They are only used to keep images together in splits.
- **Folds:** 10 folds, stratified by grade 0–4, with each duplicate group kept in one fold. The folds use a fixed seed (42) and do not depend on the library version.

**Limitations:** without patient IDs, the left and right eye of the same patient can still fall in different APTOS folds; only copies and look-alike photos are grouped. Mirrored copies are not detected.

## 8. Step 2: image features (`extract_features.py`)

Each 512 px image is resized to 224 × 224 (the ImageNet input size), normalised with the ImageNet mean and standard deviation, and passed through a ResNet-50 (He et al. 2016) pretrained on ImageNet. The 2,048 values after the last pooling layer are the image's features. The network is not trained on our images (frozen features), so this runs only once.

PCA is **not** applied here. It is part of each model and is fitted inside every training fold, so the test folds and the external datasets never influence it.

**Install:** `pip install torch torchvision scikit-learn`. Kaggle and Colab already have them. No GPU is needed.

**Weights:** all three of us should use the same file, so that our features are identical: [`resnet50_ram-a26f946b.pth`](https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnet50_ram-a26f946b.pth) (102 MB). These are timm's ImageNet weights for the standard ResNet-50 (79.0 % top-1), and they load into torchvision's ResNet-50. Put the file in `models/`.

**Run** (from the folder with the scripts):

```bash
python extract_features.py --data ../data_512 --weights ../models/resnet50_ram-a26f946b.pth
```

- **Time:** about 7 minutes for all 5,922 images on 4 CPU cores (about 15 images per second).
- **Resuming:** finished blocks of 256 images are kept in `features/parts_resnet50_224/` and skipped when you run again. With `--time-budget 1500` the script stops cleanly after 1,500 seconds, and the same command continues. The parts folder records the weights, input size and image list, and the script refuses to reuse parts made with different settings.
- **Full resolution (optional ablation):** `--size 512` keeps small lesions that 224 px blurs away, but is about 7 times slower on a CPU. On a Mac, add `--device mps` to use the Apple GPU. The output is then `resnet50_512.npz`.

**Outputs** (in `data_512/features/`):

| File | What it is |
|---|---|
| `resnet50_224.npz` | `X`: 5,922 × 2,048 features (float32); `uid`: the image ids, in the same row order as `manifest.csv` |
| `resnet50_224.json` | settings, the weights file and its SHA-256, the PyTorch version |
| `parts_resnet50_224/` | the blocks used for resuming; can be deleted once the `.npz` exists |

Loading the features, and the pattern for every model (the scaler and the PCA are refitted on each training fold):

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

m = pd.read_csv("data_512/manifest.csv")
with np.load("data_512/features/resnet50_224.npz") as z:
    X, uid = z["X"], z["uid"]
assert (uid == m.uid.values).all()                 # row i of X is row i of the manifest

aptos = (m.dataset == "aptos").values
y, fold = m.ref3.values, m.fold.values
for k in range(10):                                # E1 outer folds
    train, test = aptos & (fold != k), aptos & (fold == k)
    model = make_pipeline(StandardScaler(), PCA(n_components=128), LogisticRegression(max_iter=5000))
    model.fit(X[train], y[train])
    p = model.predict_proba(X[test])                # columns: no referral, routine, urgent
```

**Checks on our features (25 Oct 2026):**

- The row order matches `manifest.csv`, and every value is finite.
- Copies of the same photo get the same features: median relative difference 0.000 (largest 0.05), against 0.58 for two random photos (never below 0.24).
- PCA of the standardised APTOS features: 21 components keep 50 % of the variance, 227 keep 80 %, 486 keep 90 % and 755 keep 95 %. The variance is spread widely, so the number of components should be tuned in the inner folds (for example 64, 128 or 256) rather than set by a variance cut-off.
- A quick sanity check (logistic regression on 128 components, nothing tuned, so these are not results):

| Trained on → tested on | AUROC referable | AUROC urgent | 3-class accuracy |
|---|---|---|---|
| APTOS folds 1–9 → APTOS fold 0 (367 images) | 0.968 | 0.901 | 0.815 |
| all of APTOS → IDRiD (516) | 0.916 | 0.871 | 0.634 |
| all of APTOS → Messidor-2 (1,744) | 0.729 | 0.905 | 0.746 |

The features clearly carry the grade. On Messidor-2, however, the model calls 1,702 of the 1,744 images "no referral", so most of the 457 referable eyes would be sent home. Its 74.6 % accuracy is about what calling every eye "no referral" would score (73.8 %). This is the new-clinic problem the project addresses (E2, C2).

## 9. Step 3: baselines and course models (E1)

`run_models.py` trains every model under the E1 protocol and saves its predictions; `summarize_e1.py` turns the predictions into tables and figures.

**Protocol**, in each of the 10 outer folds:

- **Test:** the APTOS images of that fold, so every APTOS image is tested exactly once.
- **Calibration images:** 15 % of the other nine folds, chosen by `group_id`, are kept aside for the conformal step (C1). The models never train on them; their predictions are saved for later.
- **Tuning:** the remaining 85 % are split into 5 inner folds (by group, stratified by grade). The setting with the best mean of AUROC(referable) and AUROC(urgent) is refitted on all 85 %.
- **Features:** standardised and reduced with PCA inside every training split. The number of components is one of the tuned settings.
- **Calibration of probabilities:** one-vs-rest isotonic regression on the inner out-of-fold predictions. Raw and calibrated probabilities are both saved.
- **External sets:** every fitted model also predicts IDRiD and Messidor-2, so E0 and E2 reuse the same models.

**Models:**

| Key | Model | Tuned settings |
|---|---|---|
| `b1` | B1 most likely class | none |
| `b2` | B2 image quality only: gradient boosting on the photo columns of `manifest.csv` (quality features, photo and retina size) | learning rate, tree size |
| `b2a`, `b2b` | B2 with the size columns only, and with the quality columns only | as B2 |
| `lr` | logistic regression (the reference model) | PCA size, C |
| `gnb` | Gaussian naive Bayes | PCA size |
| `lda`, `qda` | linear and quadratic discriminant analysis | PCA size; regularisation for QDA |
| `knn` | k-nearest neighbours, distance-weighted | PCA size, k |
| `cart` | decision tree | PCA size, depth, leaf size |
| `rf` | random forest, 200 trees | PCA size, features per split |
| `ada` | AdaBoost, 100 small trees | PCA size, tree depth, learning rate |
| `gb` | gradient boosting (histogram-based) | PCA size, learning rate, tree size |
| `svm` | SVM with an RBF kernel; probabilities by Platt scaling | PCA size, C, γ |
| `mlp` | multi-layer perceptron | PCA size, hidden layers, weight decay |

B0 (refer none, or refer everyone urgently) needs no training; the summary adds it.

**Run:**

```bash
python run_models.py --data ../data_512                              # all models -> ../results/e1
python run_models.py --data ../data_512 --models lr --target grade   # 5-grade logistic regression, for E0
python summarize_e1.py --data ../data_512 --results ../results/e1
```

- **Time:** on 2 CPU cores, about 15 minutes for the baselines, the simple models and the 5-grade run, and about an hour more for SVM, MLP, random forest and the two boosting models. `--models b1,b2,lr` runs a subset, and `--jobs` sets the number of parallel jobs.
- **Resuming:** each finished (model, fold) pair is saved at once and skipped on the next run. `--time-budget` stops the script cleanly.
- Needs `scikit-learn`, plus `matplotlib` for the figures.

**Outputs** (in `results/e1/`):

| File | What it is |
|---|---|
| `summary.md` | the tables: ranking (AUROC with 95 % CIs, QWK, accuracy), decisions at the most likely class, calibration, confusion matrices, the same-camera AUROC, the difference to logistic regression, the chosen settings |
| `metrics.csv`, `per_fold.csv`, `best_params.csv` | the same numbers in full, the AUROC of every fold, the setting chosen in every fold |
| `auroc_models.png` | AUROC with 95 % CIs for every trained model |
| `same_camera_auroc.png` | referable AUROC over all pairs against pairs from the same camera, for every trained model |
| `<model>/fold<k>.csv.gz` | the predictions: `uid`, `role` (test / cal / external), raw probabilities `p0`–`p2`, calibrated `c0`–`c2` (`p0`–`p4` and `c0`–`c4` in `lr-grade`) |
| `<model>/fold<k>_tuning.csv` | every setting tried in that fold, with its inner-CV AUROCs |
| `run_config.json`, `run_config-grade.json`, `run_log.txt` | grids, seeds and library versions; the console output of every run |

**Reading the tables:**

- AUROCs use the raw probabilities. QWK, accuracy, sensitivity and specificity use the most likely class: there is no tuned threshold and no conformal step yet.
- The **same-camera AUROC** compares each referable eye only with non-referable eyes photographed with the same camera (photo size). This removes the camera shortcut described in section 11.
- 95 % CIs come from 1,000 bootstrap resamples of whole groups, the same resamples for every model, so differences between models are paired.
- Copies of one photo that carry different grades are all kept, each with its own grade. They always fall in the same fold.

## 10. Step 4: E0, reproducing [8] (`e0_conformal.py`)

This is standard split conformal prediction, as in [8]:

- **Threshold:** one threshold for all classes, set at a 90 % target on the APTOS calibration images (the 15 % kept aside in each E1 fold).
- **Test:** the same threshold is used unchanged on the held-out APTOS fold, IDRiD and Messidor-2.
- **Coverage:** the share of images whose true class is in their prediction set.
- **Scores:** LAC (1 − p(y | x)) and APS.
- **Models:** the 5-grade logistic regression (`lr-grade`) matches the 5-grade setting of [8]. The 3-class model shows the same for our classes.

```bash
python e0_conformal.py --data ../data_512 --results ../results/e1 --models lr-grade,lr,svm
```

Any model folder of `results/e1` can be added to `--models`; the figure shows the first two.

Outputs (in `results/e0/`): `summary.md`, `coverage.csv` (every fold, test set and grade), `coverage_by_grade.png`.

## 11. k-means check (`kmeans_check.py`)

```bash
python kmeans_check.py --data ../data_512
```

This runs k-means (k = 2–6) on the PCA-reduced features of all 5,922 images. The clusters are compared with the dataset and the class, and, within APTOS, with the camera (photo size), using the adjusted Rand index and normalised mutual information.

Outputs (in `results/kmeans/`): `summary.md`, `kmeans.csv`, `aptos_by_camera.csv`, `pca_scatter.png` (all images by dataset and by class) and `aptos_cameras.png` (APTOS photos by camera).

**The camera shortcut (important for everyone).** In APTOS, the photo size identifies the camera, and the cameras saw very different patients:

| Photo sizes | APTOS photos | Referable |
|---|---|---|
| 2048×1536 and 819×614 | 638 | 0 % |
| 1050×1050 | 974 | 6 % |
| 2588×1958 and 2416×1736 | 1,171 | 48–58 % |
| 3216×2136, 3388×2588 and 1504×1000 | 643 | 91–98 % |

A model can therefore score well on APTOS just by recognising the camera. B2a (size only) shows how far that alone goes, and the same-camera AUROC shows what remains without it. The ResNet-50 features see the cameras too: on the first two principal components, the photos from the mostly-healthy cameras form a cloud of their own (`aptos_cameras.png`). With k = 2, k-means puts 1,740 of the 1,744 Messidor-2 photos into that cloud's cluster, and 515 of the 516 IDRiD photos into the other. Keep this in mind when comparing APTOS results with IDRiD and Messidor-2.

## 12. Step 6: C1, a referral rule with one conformal threshold per class (`c1_referral.py`)

```bash
python c1_referral.py --data ../data_512 --results ../results/e1     # about half a minute
```

Nothing is retrained: the script reuses the E1 predictions (section 9).

- **Fitting.** In every outer fold, each rule is fitted on that fold's 15 % calibration images and applied to the held-out fold, so every APTOS photo is decided once by a rule that never saw it.
- **New clinics.** The same thresholds are also applied unchanged to IDRiD and Messidor-2. This is what happens at a new clinic without recalibration, where C2 starts.
- **Inputs.** The model's raw probabilities. The conformal step calibrates by itself. On APTOS the isotonic-calibrated probabilities give the same result (summary section 8), but at a new clinic the isotonic map learnt on APTOS sets many sick eyes' probabilities to exactly 0, and the thresholds become useless (section 13).
- **Every model.** C1 is also run with every course model.

**Rules compared** (α = 0.1, i.e. 90 % targets; α = 0.05 is reported too):

| Rule | What it does |
|---|---|
| Most likely class | the plain decision, as in E1 |
| B3 tuned thresholds [6, 17] | two thresholds set on the calibration images so that 90 % of their referable eyes clear the first (on P(referable)) and 90 % of their urgent eyes the second (on P(urgent)) |
| B4 standard conformal | one conformal threshold for all classes, as in E0 [8] |
| B5 fixed "uncertain" band | uncertain (to a grader) if the top class probability is below 0.6, else the most likely class |
| **C1** class-wise conformal | one conformal threshold per class, each set only on the calibration images of that class; the prediction set gives the decision with the table below |
| C1-g (additional variant) | C1's sets, but a set holding "urgent" and another class goes to a grader instead of an urgent referral |
| C1-q | as C1, with one threshold per class and image-sharpness tertile (Mondrian by image quality) |

C1's decision table, from the proposal (B4 and C1-q use it too):

| Prediction set | Decision |
|---|---|
| contains "urgent" | urgent referral |
| only "routine" | routine referral |
| only "no referral" | no referral (rescreen) |
| anything else ({no referral, routine}, or empty) | uncertain: a human grader decides |

**Outputs** (in `results/c1/`):

- `summary.md`: all tables. These cover the rules compared, coverage per class against conformal theory, decisions by DR grade, α = 0.05, every model, subgroups by camera and sharpness, the new clinics without recalibration, and raw probabilities.
- `metrics.csv` and `per_fold.csv`: the numbers in full.
- `thresholds.csv`: every threshold.
- `decisions_lr.csv.gz`: every decision of the main model.
- Three figures: `targets_by_rule.png`, `coverage_by_grade.png` and `workload_by_model.png`.

**Findings (4 Nov 2026), logistic regression, α = 0.1:**

- **C1 meets both targets on APTOS.**
  - 4.2 % of referable eyes are sent home (95 % CI 3.2–5.2; target at most 10 %).
  - 91.2 % of urgent eyes are referred urgently (88.3–93.8; target at least 90 %).
  - Each class's coverage matches conformal theory: 89.7 / 89.3 / 91.2 % observed, against 90.2 / 90.4 / 91.1 % expected.
- **The simpler rules miss the urgent target.** Urgent eyes referred urgently: standard conformal (B4) 67.0 %, the 0.6 cut-off (B5) 34.4 %, the most likely class 46.3 %.
- **Tuned thresholds (B3) come close** (7.9 % sent home, 88.5 % urgent), for two reasons:
  - **Urgent threshold.** B3's urgent threshold sits one calibration image above C1's, so its expected coverage is 89.6 %, just under the target. The gap grows with fewer calibration images (for 20 images: 85.7 % against 90.5 %), which matters for C2.
  - **Pooled target.** B3 targets referable eyes pooled, so it sends home 10.0 % of routine eyes. C1 keeps routine and urgent eyes separately under 10 %.
- **The price is a heavy urgent load.**
  - At APTOS prevalence, C1 refers 451 eyes per 1,000, 336 of them urgently.
  - Only 36 % of urgent referrals are truly urgent, because 64 % of grade-2 eyes are flagged urgent.
  - 87.8 % of healthy eyes are sent home, and 1.0 % of all eyes go to a grader.
- **C1-g trades urgent referrals for grader time.** It keeps the same 4.2 % sent home, and 91.6 % of urgent eyes are referred urgently or sent to a grader. Urgent referrals fall from 336 to 58 per 1,000 (68 % of them truly urgent), but 28.8 % of all eyes go to a grader.
- **C1 meets both targets with every course model.** Better models send more healthy eyes home: 87.8 % with logistic regression against 71.5 % with naive Bayes. B1, which knows nothing, refers everyone. QDA's raw probabilities are pinned at 0 or 1, so its scores tie and it refers 730 eyes per 1,000; its calibrated probabilities do much better.
- **A subgroup warning.** On APTOS's mostly-healthy cameras, C1 sends home 26 of the 55 referable eyes (47 %). The guarantee holds for each class on average over cameras, not within each camera (sections 15, 17 and 18).
- **New clinics without recalibration.**
  - IDRiD stays near the targets: 9.6 % sent home, 89.6 % urgent.
  - Messidor-2 fails: 79.7 % of referable eyes sent home, 37.1 % of urgent eyes referred urgently.
  - This is the case for C2.
- **C1-q (by sharpness tertile) is safer but costly:** 523 referrals per 1,000, and only 70.5 % of healthy eyes sent home. Some class × tertile cells have too few calibration images, so that class enters every set of the tertile.
- **At α = 0.05,** C1 sends home 2.4 % of referable eyes and refers 96.5 % of urgent eyes urgently, at 453 referrals per 1,000.
- **Calibrated probabilities give the same picture on APTOS** (C1: 4.4 % and 91.4 %).

## 13. Step 7: C2 and E2, the thresholds refitted at a new clinic (`e2_new_clinics.py`)

```bash
python e2_new_clinics.py --data ../data_512     # about 4 minutes on 2 cores
```

**The model.** Logistic regression trained on all 3,662 APTOS photos (C = 0.003, no PCA reduction: E1's most frequent setting). It never sees an IDRiD or Messidor-2 photo.

**The experiment.** For each clinic and each k (10 to 300 at IDRiD, up to 400 at Messidor-2), 200 times:

1. Draw k labelled local photos by whole groups.
2. Refit the rule on them.
3. Test it on the clinic's other photos.

There are three settings: IDRiD (random split); IDRiD's official split (k from its training photos, tested on its 103 test photos); and Messidor-2 (random split).

**Rules:**

| Rule | What it does |
|---|---|
| B6 | no recalibration: C1's thresholds from APTOS |
| **C2** | C1's three thresholds refitted on the k local photos |
| B3-k, B4-k | tuned thresholds, or one conformal threshold, refitted on the k photos |
| B7a | logistic regression retrained on the k local photos only |
| B7b | logistic regression retrained on APTOS plus the k local photos, keeping the APTOS thresholds |
| C2-PAC (additional) | per class, the threshold that keeps the clinic's own coverage above 90 % with 90 % probability, not only on average |
| C2 with the camera-decorrelated model (additional work) | as C2, with the model of section 17 |

**Outputs** (in `results/e2/`): `summary.md`, `draws.csv` (every draw), `curves.csv` (means and 5–95 % ranges), `aptos_model.csv.gz` (the model's probabilities), and three figures: `e2_targets.png`, `e2_retrain.png` and `e2_extensions.png`.

**Findings (4 Nov 2026), α = 0.1:**

- **Without recalibration (B6) Messidor-2 fails:** 79.8 % of referable eyes are sent home. IDRiD is close to the targets (10.6 % sent home, 87.2 % urgent).
- **A class needs at least 9 local photos of its own** before its threshold can be finite; until then it is in every set and every eye is referred. The urgent class is the rarest, so it decides:
  - **IDRiD:** about 40–50 photos. At k = 50, C2 sends home 2.1 % of referable eyes and refers 93.6 % of urgent eyes urgently; it sends home 36.8 % of healthy eyes (51 % at k = 300).
  - **Messidor-2:** only 6.3 % of eyes are urgent, so it takes about 200–300 photos. At k = 300: 6.2 % sent home, 92.5 % urgent, 25.5 % of healthy eyes sent home, 43.5 % to a grader.
  - **The ceiling is the model's ranking.** The APTOS model ranks Messidor-2 eyes poorly (AUROC 0.764), which limits how many healthy eyes can safely be sent home.
- **The guarantee holds at the clinics.** With random splits, every class's average coverage is at least 89.9 % at every k (target 90 %; simulation noise about ±0.3 points).
- **A warning from IDRiD's official split.** There the urgent target is missed (78–88 % for k ≥ 40). Its 32 urgent test eyes score lower than the training split's (median P(urgent) 0.28 against 0.40; Mann–Whitney p = 0.15). With one fixed test set this may be chance, but the guarantee needs calibration photos that resemble the later photos.
- **Conformal against tuning.** With few photos, B3-k's plain 90 % point under-covers the urgent class: 73 % of urgent eyes referred urgently at k = 10 and 83 % at k = 20 at IDRiD, against C2's 100 % and 98.5 %. One threshold for all classes (B4-k) stays at about 82 % at IDRiD and 57 % at Messidor-2.
- **Recalibrate rather than retrain.**
  - **B7a** (retrained on the k local photos only) misses most urgent eyes at Messidor-2.
  - **B7b** (APTOS plus the local photos, APTOS thresholds) is close at IDRiD but still sends home 45–74 % of referable eyes at Messidor-2.
  - Refitting only the thresholds is what restores the targets.
- **Raw, not calibrated, probabilities for the conformal scores.**
  - **The problem.** At Messidor-2, 16–17 % of routine and urgent eyes get a calibrated probability of exactly 0 for their class. The isotonic map from APTOS has a lowest step of 0, which ties all these eyes.
  - **The consequence.** The class thresholds become 1, and every eye is referred: 0.4 % of healthy eyes sent home at k = 200, against 20.1 % with raw probabilities.
- **Additional work:**
  - **C2-PAC** makes the promise hold for each clinic's own draw, not just on average. Over all the clinic's eyes of a class (random splits), its coverage falls below 90 % in at most 6 % of draws (promised: at most 10 %), against up to 36 % for C2. Measured on the test photos alone, the share looks higher (up to 15 % at IDRiD), because a finite test set scatters around the clinic's true coverage. The cost: at IDRiD (k = 200), 37.9 % instead of 50.2 % of healthy eyes are sent home.
  - **The camera-decorrelated model** sends home more healthy eyes at IDRiD after recalibration (57.4 % against 50.2 % at k = 200), with the same targets; at Messidor-2 there is no gain.
  - **C2-IW** (section 24) chooses which local photos to label and weights them, and needs about a third fewer labels at Messidor-2.

## 14. Step 8: C3 and E3, prevalence and workload (`c3_prevalence.py`)

```bash
python c3_prevalence.py --data ../data_512 --results ../results     # under a minute
```

**Part A: APTOS with fewer sick eyes.** The out-of-fold C1 decisions are resampled to 5 %, 12.5 % and 25 % referable eyes, keeping the grade mix; 200 repeats. This is pure label shift.

- In each repeat, APTOS is split by group into two halves. Half A gives each class's decision rates; half B is the clinic.
- The workload is forecast as 1000 × Σ π_y r_y, with the class shares π_y coming from:
  - **B8:** APTOS's shares (no correction);
  - **EM** [12]: estimated from the clinic's unlabelled photos;
  - **BBSE** (additional): estimated from the unlabelled photos with the model's confusion matrix;
  - **B9:** the true shares (oracle).

**Part B: the real clinics.** The class shares of IDRiD and Messidor-2 are estimated from their unlabelled photos, and C2's workload is forecast from its k labelled photos.

**Outputs** (in `results/c3/`): `summary.md`, `workload_aptos.csv`, `prevalence_clinics.csv`, `workload_clinics.csv`, and two figures: `c3_workload.png` and `c3_prevalence.png`.

**Findings (4 Nov 2026):**

- **C1's guarantee does not move with prevalence.**
  - At 5 %, 12.5 %, 25 % and 40.6 % referable eyes, C1 sends home 4.1–4.4 % of referable eyes and refers 91.1–91.2 % of urgent eyes urgently [11].
  - Standard conformal prediction's marginal coverage rises to 95.5 % at 5 % prevalence, while its urgent eyes stay at 67 %. A marginal guarantee cannot protect a rare class.
- **The workload moves a lot.** At 5 % referable eyes, C1 refers 154 eyes per 1,000 (101 urgently), and a referral is right 30.7 % of the time (PPV). At APTOS's 40.6 %: 451 referrals, PPV 85.2 %.
- **Forecasting from unlabelled photos works under label shift.**
  - **Without a prevalence correction (B8),** the forecast is 297 referrals per 1,000 too many at 5 % prevalence, and the PPV is overestimated by 54 points.
  - **With EM or BBSE,** the forecast is within a few referrals per 1,000 (mean absolute error 6–11), as close as with the true shares (B9: 10–12, the noise of the decision rates themselves), and the PPV within a few points.
  - **EM estimates the share of referable eyes** at 5.4 % when the truth is 5 % (BBSE 5.0 %, with a wider spread: 2.7–7.4 % against 4.5–6.3 %).
- **Bayes-adjusted risks are calibrated again.** At 5 % prevalence, the ECE of P(referable) falls from 0.092 to 0.012 with EM's shares.
- **At a real clinic EM fails, because more than the shares has changed.** At Messidor-2, EM estimates 99.5 % no referral (truth 73.8 %): the photos themselves differ (camera, population, grading).
- **The k labelled photos that C2 needs anyway give unbiased shares,** and forecast C2's workload well. At Messidor-2 (k = 200), the referrals per 1,000 are off by +7 and the PPV by +0.3 points.

## 15. Step 9: error analysis and subgroups (`error_analysis.py`)

```bash
python error_analysis.py --data ../data_512 --results ../results
```

The script reads C1's out-of-fold decisions. It lists every missed urgent eye (`results/errors/missed_urgent.csv`, with photo paths, so you can look at them on your own computer) and gives error rates with Wilson intervals by DR grade, camera group, sharpness, quality warnings and conflicting copies (`summary.md`, `subgroups.csv`, `errors_by_subgroup.png`).

**Findings (4 Nov 2026), C1 with logistic regression:**

- **The missed urgent eyes.** 43 of 488 urgent eyes are not referred urgently: 30 of them are called routine, 11 are sent home and 2 go to a grader.
- **They are confident misses.** Their median P(urgent) is 0.058, against 0.424 for all urgent eyes, so a slightly lower threshold would not catch them.
- **Grade 3 and grade 4** are missed about equally often (8.3 % and 9.2 %).
- **Referable eyes sent home** are mostly grade 2 (5.2 %, against 2.1–2.4 % for grades 3–4).
- **The camera dominates.** On the mostly-healthy cameras, 26 of 55 referable eyes are sent home (47 %), against 3.8 % and 1.4 % on the other camera groups.
- **Sharpness is tangled with the camera.** The sharpest third has the most referable eyes sent home, but 21 of its 25 such errors come from mostly-healthy cameras.
- **No clear difference** for quality warnings or conflicting copies.

## 16. Step 10: ablations and robustness (`extract_variants.py`, `ablations.py`)

These need the images, so the features are made where the images are (your computer). On 4 CPU cores, the 224 px variants take 2–4 minutes each and the 512 px features about 15 minutes.

```bash
python extract_features.py --data ../data_512 --weights ../models/resnet50_ram-a26f946b.pth --size 512
for v in graham blur dark jpeg; do
  python extract_variants.py --data ../data_512 --weights ../models/resnet50_ram-a26f946b.pth --variant $v
done
python ablations.py --data ../data_512 --results ../results     # about 3 minutes
```

- **Image ablations.** Logistic regression with a fixed setting, on 224 px features (as in E1), on 512 px features, and on Graham-filtered photos.
- **PCA size,** read from E1's own tuning logs.
- **Robustness.** E1's logistic regression applied to blurred, darkened and JPEG-compressed test photos, with C1's thresholds from the original photos or recalibrated on equally altered ones. Its predictions on the original photos reproduce E1's saved ones to within 0.002.

Outputs (in `results/ablations/`): `summary.md`, `image_ablation.csv`, `image_ablation_c2.csv`, `pca_size.csv`, `robustness.csv`, `ablations.png`.

**Findings (4 Nov 2026):**

- **512 px helps a little, mostly at new clinics:**
  - urgent AUROC on APTOS 0.930 against 0.920;
  - Messidor-2 referable AUROC 0.782 against 0.760;
  - after C2 at IDRiD (k = 100), 54.5 % of healthy eyes sent home against 46.6 %.
- **The Graham filter does not help:** APTOS 0.970 against 0.974, Messidor-2 0.747 against 0.760.
- **PCA size hardly matters** for logistic regression: the inner-fold score ranges from 0.9433 (256 components) to 0.9443 (1,024–2,048).
- **Worse photos shift the decisions, and recalibrating on equally poor photos restores the targets:**
  - **Blur** (AUROC 0.918) doubles the referable eyes sent home, to 8.2 %.
  - **Half brightness** drops urgent eyes referred urgently to 87.1 %; recalibrated, 91.8 %.
  - **JPEG at quality 20** cuts the urgent AUROC to 0.851.
  - **The cost of recalibrating on poor photos is usefulness:** with blurred photos, only 49.1 % of healthy eyes are sent home and 23 % of eyes go to a grader.

## 17. Additional work: the APTOS camera shortcut (`camera_shortcut.py`)

This is **not** one of the proposal's deliverables. It goes beyond the proposal, so the report keeps it apart from the deliverables.

**Question.** APTOS photos come from several cameras that saw very different patients (section 11). Do the models learn the camera instead of the retina? Does that explain who is missed at IDRiD and Messidor-2? And what helps?

```bash
python camera_shortcut.py --data ../data_512 --results ../results/e1     # about 2–3 minutes
```

It needs the E1 predictions of `lr`, `svm` and `b2a` (section 9). Outputs go to `results/camera_shortcut/`: `summary.md` (all tables), the numbers as CSV files, and three figures: `aptos_misses.png`, `external_h.png` and `fix_preview.png`.

**What it does:**

1. **Camera groups.** APTOS cameras (photo sizes) are grouped by their share of referable eyes: H (mostly healthy, ≤ 10 %), M (mixed) and S (mostly sick, ≥ 80 %).
2. **Probes.** Classifiers are trained on **grade-0 eyes only**, so disease cannot help them. They predict the camera from the features, the camera from the quality columns, and the dataset from the features. One of them gives the camera-style score h = P(photo looks like an H camera).
3. **Who is missed.** Misses, false referrals and standard-conformal (E0) coverage per camera group on APTOS, compared with IDRiD and Messidor-2.
4. **First look at fixes.**
   - **Camera-decorrelated weights:** each photo is weighted by P(class) / P(class | camera group), so that in the weighted data the camera says nothing about the class.
   - **Training on the mixed cameras only,** with random subsets of the same size and class mix as the control.
   - Both use one fixed setting (logistic regression, C = 0.003), not a tuned comparison yet.

**Findings (25 Oct 2026):**

- **The features know the camera.** Among eyes with no retinopathy at all, the camera is recognised with 91 % balanced accuracy from the ResNet-50 features (5 cameras, chance 20 %). It is recognised with 96 % from the quality columns alone, which is why B2b scores like B2. H cameras are separated from the others perfectly (AUROC 1.00), and the dataset is recognised with 98 % balanced accuracy.
- **The models use it.**
  - Logistic regression misses 64 % of the referable eyes photographed with H cameras, against 16 % on M cameras and 6 % on S cameras. On S cameras it refers 68 % of the healthy eyes.
  - Inside the 1050×1050 camera, it still ranks eyes well (AUROC 0.955): the eyes are seen, but the decision is pulled towards the camera's usual answer.
  - Even within APTOS, standard conformal prediction covers only 45 % of the referable eyes on H cameras, behind the overall 90 %.
- **Messidor-2 looks like an H camera.** Every Messidor-2 photo has h > 0.5 (median 1.00), and it behaves like one, only more so: 90 % of its referable eyes are missed, and E0 coverage of those eyes is 8 %. IDRiD sits in between (median h 0.60, 27 % missed).
- **Decorrelating the training data makes decisions independent of the camera.**
  - In 10-fold cross-validation, the referable eyes missed per camera group (H / M / S) go from 58 / 15 / 5 % to 27 / 20 / 21 %, and the healthy eyes referred on S cameras from 73 % to 22 %.
  - The same-camera AUROC is unchanged (0.922 → 0.925). Only the overall AUROC drops (0.975 → 0.966), because part of it came from the camera.
  - At Messidor-2, sensitivity doubles (12 % → 25 %). At IDRiD it falls (75 % → 61 %): IDRiD photos partly resemble the mostly-sick cameras, so the shortcut had been flattering IDRiD.
- **Training on the mixed cameras only is worse** at Messidor-2 (AUROC 0.632, against 0.745 for random subsets of the same size). The only APTOS photos that resemble Messidor-2 are the H-camera ones, and they are left out. So the fix has to keep those photos but break their link to the class.
- **Even without the shortcut, sensitivity at Messidor-2 stays low (25 %).** The camera explains part of the failure at a new clinic, not all of it, which supports recalibrating with local images (C2).

**Followed up** in section 18 (the fixes tuned under the E1 protocol, two kinds of linear concept erasure, and per-camera thresholds), in section 13 (the decorrelated model with E2's recalibration) and in section 23 (against the standard methods for spurious correlations).

## 18. Additional work: camera fixes, tuned under the E1 protocol (`camera_fixes.py`)

This is also **not** a deliverable. It tests section 17's fixes properly.

```bash
python camera_fixes.py --data ../data_512 --results ../results     # about 20 minutes on 2 cores
```

**What it does.** Logistic regression is retrained in each of E1's 10 outer folds, with C tuned in 5 grouped inner folds, on the same calibration photos as E1. C1 (α = 0.1) is then fitted on each fold's calibration photos. Three fixes are compared against no fix (E1's own predictions):

- **Camera-decorrelated weights** (section 17): each training photo is weighted by P(class) / P(class | camera group).
- **Camera erasure fitted on healthy eyes (LEACE,** linear concept erasure [Belrose et al. 2023]**):** the features are moved so that every camera group has the same mean of every linear combination of features. The eraser is fitted on grade-0 training eyes only, so that it removes camera style, not disease.
- **Camera erasure within each grade (LEACE):** the same kind of eraser, estimated within each DR grade on all training eyes. The camera groups then have the same means inside every grade, sick eyes included, while the differences between grades are kept.

The script also checks each eraser with camera probes (can a linear classifier still tell H from M cameras?), and explains the results (summary section 5).

**Outputs** (in `results/camera_fixes/`): `summary.md`, `metrics.csv`, `by_camera_group.csv`, `erasure_check.csv`, `chosen_C.csv`, the fold predictions of each fix (`weights/`, `erasure/`, `erasure_grade/`, same format as E1), and `camera_fixes.png`.

**Findings (4 Nov 2026):**

- **The weights fix the subgroup failure of section 12.**
  - Referable eyes sent home by C1 on the mostly-healthy cameras fall from 47.3 % (26 of 55) to 9.1 % (5 of 55). Every camera group is now under 10 % (H 9.1, M 4.5, S 4.6 %).
  - Overall, C1 still meets both targets: 4.7 % sent home, 92.0 % of urgent eyes referred urgently.
  - **The price:** 83.8 % instead of 87.8 % of healthy eyes are sent home. The overall AUROC falls from 0.973 to 0.965, because part of it came from the camera; the same-camera AUROC rises from 0.915 to 0.922.
- **Why the models miss these eyes.** The plain model gives grade-2 eyes from H cameras a median P(referable) of 0.15, against 0.86 on M cameras: in its training data almost every H-camera photo is healthy (1,557 healthy, 40 routine, 15 urgent). The weights give those 55 sick eyes 11–14 times the average weight, and the median rises to 0.64.
- **Erasure fitted on healthy eyes removes the camera only from healthy eyes.** A camera probe on healthy eyes falls from AUROC 1.000 to 0.500 (chance), but on referable eyes it stays at 0.998. The H-camera misses stay (52.7 %).
- **Erasure within each grade removes most of the camera signal from sick eyes too** (probe 0.996 → 0.625), and the H-camera misses fall from 47.3 % to 36.4 %: better, but far above the 10 % target. It also costs ranking within cameras (same-camera AUROC 0.915 → 0.901), so the directions it removes carry some disease information as well.
- **The lesson:** hiding the camera from the model helps little; changing what the training data teach about each camera (the weights) is what works.
- **At the new clinics, before recalibration,** the weights help Messidor-2 (referable eyes sent home 79.7 → 53.9 %) and hurt IDRiD (9.6 → 17.6 %). Erasure within each grade gives the best Messidor-2 AUROC (0.763) but still sends home 68.8 % of its referable eyes. After C2 recalibration, both clinics meet the targets with the plain and the weighted model, and the weighted model sends more healthy eyes home at IDRiD (section 13).
- **One threshold per camera group is not possible with APTOS.** Per fold, the mostly-healthy cameras have about 5 routine and 2 urgent calibration photos, and a class needs at least 9 for a finite threshold. So the fix has to be in the model, not in the calibration.

## 19. Deliverables and additional work

The faculty asked us to keep the two apart in the final report. Everything in sections 1–16 delivers the proposal; the items below go beyond it.

| Additional item | Where | What it adds |
|---|---|---|
| APTOS camera shortcut study | section 17, `camera_shortcut.py` | shows that the models partly recognise the camera and that this explains who is missed |
| Camera-decorrelated weights and two kinds of LEACE erasure, tuned | section 18, `camera_fixes.py` | the weights bring every camera group under the 10 % target; erasure helps little, and the probes show why |
| C1-g, a grader variant of C1 | section 12, `c1_referral.py` | urgent referrals fall from 336 to 58 per 1,000 for the same safety, at the cost of grader time (29 % of eyes) |
| C2-PAC, a per-clinic guarantee | section 13, `e2_new_clinics.py` | coverage of at least 90 % with probability at least 90 % for each clinic's own draw, not only on average |
| The decorrelated model with C2 | section 13, `e2_new_clinics.py` | more healthy eyes sent home at IDRiD for the same targets |
| BBSE, a second prevalence estimator | section 14, `c3_prevalence.py` | an independent check on EM |
| Exact evidence maps, a deletion test and a mirror test | section 21, `evidence_maps.py` | shows where the camera shortcut lives in the photo and that the maps point at what matters; finds a fixed network hotspot, and that mirroring an eye changes 12 % of C1's decisions |
| Mirror-averaged features | section 22, `mirror_fix.py` | the same decision for an eye and its mirror image (before: 11 % of APTOS decisions changed), with no loss of accuracy |
| The camera-group guarantee against strong baselines | section 23, `group_robust.py`, `group_frontier.py` | our weights against group-balanced weights, DFR, Group DRO, JTT and per-group thresholds, with significance tests: six methods meet the 10 % target in every camera group, ours at the smallest cost in healthy eyes (significant on 2,175 healthy eyes, and still true when each baseline's setting is chosen with hindsight); none of them makes APTOS's thresholds work at Messidor-2 |
| C2-IW, label-efficient recalibration (a new method) | section 24, `label_efficient.py` | label the photos most likely to be rare and correct with weights: coverage on target in every setting, about a third fewer labels at Messidor-2, and recalibration becomes possible at 5 % prevalence; a bolder setting (λ = 0.5) saves more but falls up to 0.8 points short; without the weights coverage falls to 86–89 %, and exact weighted conformal prediction gains nothing |

## 20. Reproducibility check (4 Nov 2026)

The results of steps 6–10 and of sections 18, 23 and 24 (`results/c1`, `e2`, `c3`, `errors`, `ablations`, `camera_fixes`, `group_robust`, `label_efficient`) were made with Python 3.13 (scikit-learn 1.9, numpy 2.5, pandas 3.0) on Linux (x86-64). Six scripts were then run again on a second machine, a Linux virtual machine on our Mac (ARM; Python 3.10, scikit-learn 1.7, numpy 2.2, pandas 2.3), from the files in this folder:

| Script | Result on the second machine | Time |
|---|---|---|
| `c1_referral.py` | identical outputs (every decision and every metric) | 8 s |
| `c3_prevalence.py` | identical tables | 5 s |
| `error_analysis.py` | identical tables and list of missed eyes | 1 s |
| `e2_new_clinics.py` | the logistic regression's probabilities differ by at most 0.005 between library versions (solver tolerance); the summary tables differ by at most 0.1 percentage points, and one 5–95 % range end by 1 point | 81 s on 4 cores |
| `group_robust.py` | the H-camera results and every McNemar test identical; the other rates within 1.1 points, except one cell that differs by one eye of 37 (2.7 points); the share of draws meeting both targets within 3 points. No conclusion changes | 93 s on 4 cores |
| `label_efficient.py` | identical outputs | 57 s |

`group_frontier.py` was not run on the second machine (10 minutes); its standard settings reproduce `group_robust.py`'s numbers exactly.

The isotonic-calibrated probabilities can move more (up to 0.07), because a tiny change can cross one of their steps. This is one more reason the conformal rules use the raw probabilities (section 13).

## 21. Additional work: where in the photo does the model find its evidence? (`evidence_maps.py`)

Also **not** a deliverable. Our logistic regression is linear in ResNet-50's 2,048 average-pooled features, so each of its log-odds splits exactly into a 7 × 7 map over the photo (class activation mapping, Zhou et al. 2016): the cell at (i, j) contributes w · F(i, j) / 49, where F(i, j) is the last convolutional layer there. The maps add up to the very log-odds the referral rules use. They are not an approximation, unlike gradient-based saliency.

The stages, in order. `maps`, `deletion`, `flip` and `examples` need the photos and PyTorch, so run them on the computer that has the images (about 7 minutes in all on 4 CPU cores):

```bash
python evidence_maps.py heads    --data ../data_512 --results ../results     # the linear heads, checked (2 minutes)
python evidence_maps.py maps     --data ../data_512 --weights ../models/resnet50_ram-a26f946b.pth
python evidence_maps.py deletion --data ../data_512 --weights ../models/resnet50_ram-a26f946b.pth
python evidence_maps.py flip     --data ../data_512 --weights ../models/resnet50_ram-a26f946b.pth
python evidence_maps.py summary  --data ../data_512 --results ../results
python evidence_maps.py examples --data ../data_512 --results ../results
```

`maps` and `deletion` can be stopped and run again (`--time-budget`); their parts go to `--parts` (default: inside the output folder) and can be deleted afterwards.

**What it does:**

- **Which model explains which photo.** APTOS photos are explained by the E1 fold model that never trained on them, and by the camera-decorrelated model of the same fold (section 18). IDRiD and Messidor-2 photos are explained by E2's all-APTOS models.
- **Where the evidence lies.** Each cell is classed as inside the retina, on its edge, or outside it (black). The log-odds of every photo are then split by region, and averaged by camera group and by dataset.
- **Deletion test.** For 150 APTOS urgent eyes, the cells with the most urgent evidence are blurred, and the prediction is recomputed. The controls blur random cells, and random cells on the retina.
- **Flip test.** 200 photos per dataset are mirrored left-right and top-bottom, then explained again.

**Outputs** (in `results/evidence_maps/`): `summary.md`, `heads.npz` and `heads_check.csv`, `maps.npz` (every photo's maps), `regions.csv.gz`, `deletion.csv`, `flip.npz` and `flip.csv`, and four figures: `evidence_by_camera.png`, `deletion.png`, `flip.png` and `examples.png`. `examples.png` shows APTOS and IDRiD patients' photos (never Messidor-2, whose licence forbids it), so keep it on your computer.

**Findings (4 Nov 2026):**

- **Checks.** The features recomputed from the photos match the saved ones to within 3 × 10⁻⁶. The maps reproduce every log-odds to within 4 × 10⁻⁷. The rebuilt heads match the saved predictions to within 0.004.
- **The camera shortcut is mostly in the retina, not in the black border.**
  - Grade-2 eyes on the mostly-healthy (H) cameras get a mean routine log-odds of −1.36, against +1.60 on M cameras.
  - 71 % of that gap comes from cells inside the retina, and 28 % from its edge and the black corners. So the model reads how the retina looks through that camera, not the shape of the photo's outline.
  - The decorrelated model shrinks the gap from 2.96 to 0.60, mainly by finding more evidence inside the retina of H-camera eyes (+0.15 → +1.11). The mean maps show that evidence in the centre of the retina, where it is for the other cameras (`evidence_by_camera.png`).
- **The maps point at what matters.** Blurring the 4 cells with the most urgent evidence lowers the urgent log-odds by 1.51, against 0.56 for 4 random retina cells. P(urgent) falls from 0.46 to 0.37, against 0.42. Up to 16 cells, the gap stays clear of chance (95 % CIs exclude 0).
- **Messidor-2 grade-2 eyes look healthy inside the retina** to the APTOS model (−1.32 log-odds from inside cells), which matches its failure there without recalibration (sections 10 and 13).
- **A fixed hotspot of the network.**
  - One cell near the top left of the frame carries between −0.6 and −1.1 log-odds against routine referral in every dataset, while the cells in the mirror positions do not.
  - Mirroring the photos leaves it where it is, so it belongs to the network, not to the eye. A plausible cause is the asymmetric zero padding of ResNet-50's strided convolutions (Alsallakh et al., ICLR 2021).
  - It adds about the same amount to every photo (SD 0.30, against 3.37 for the whole log-odds), so it acts mostly like a shift of the bias.
- **Mirrored eyes do not always get the same decision.**
  - Left and right eyes are mirror images, yet mirroring a photo changes the most likely class for 9 % of APTOS photos and 14 % of IDRiD photos.
  - It changes C1's decision for 24 of 200 APTOS eyes (12 %).
  - Inside the retina, the left-right mirror moves the evidence with the eye (median correlation 0.44, against 0.24 for the map left as it is). On the edge of the retina, the evidence stays in the frame. So a single photo's map should be read with care; the averages and the deletion test are the reliable evidence.

## 22. Additional work: mirror-averaged features (`mirror_fix.py`)

Also **not** a deliverable. It follows from section 21: left and right eyes are mirror images, but ResNet-50's features are not mirror-invariant, so mirroring a photo can change C1's decision. The fix averages the features of each photo with those of its mirror image. A photo and its mirror image then have exactly the same features, so every model on them gives both the same answer. It costs one extra pass of the network per photo, and no retraining of the network.

```bash
python extract_variants.py --data ../data_512 --weights ../models/resnet50_ram-a26f946b.pth --variant hflip   # 3 minutes, needs the images
python mirror_fix.py --data ../data_512 --results ../results                                                  # 4 minutes
```

**What it does.** E1's logistic regression is refitted in each outer fold with the setting E1 chose there, on the photo's own features (as E1) and on the mirror-averaged ones. C1 is fitted on each fold's calibration photos. Then:
- each held-out photo is mirrored, to count how often C1's decision changes;
- at the new clinics, the all-APTOS model and C2 are run as in E2.

**Outputs** (in `results/mirror/`): `summary.md`, `metrics.csv`, `c2.csv`.

**Findings (4 Nov 2026):**

- **With the photo's own features, the orientation matters.** Mirroring a photo changes C1's decision for 11.0 % of APTOS eyes, and for 16.7 % of referable eyes. 6.2 % of referable eyes are sent home in at least one of the two orientations, against 4.2 % in the original one. So the original figure partly depends on which way each photo happens to face.
- **Mirror-averaged features remove this,** at no cost in accuracy.
  - The AUROCs are the same or slightly higher: referable 0.974 against 0.973, same camera 0.920 against 0.915, urgent 0.924 against 0.920.
  - C1 still meets both targets: 4.4 % of referable eyes sent home, 91.0 % of urgent eyes referred urgently, 87.7 % of healthy eyes sent home.
- **At the new clinics nothing changes:** the AUROCs and C2's results stay within simulation noise (for example, Messidor-2 with 400 photos: 6.6 % sent home and 91.5 % urgent, against 6.9 % and 91.9 %).
- **Recommendation:** use mirror-averaged features in any deployed version. They make the decision for an eye independent of how the photo was taken, a property a screening rule should have.

## 23. Additional work: the camera-group guarantee against strong baselines (`group_robust.py`)

Also **not** a deliverable. Section 18 compared our camera-decorrelated weights only with no fix and with erasure. This section asks what a reviewer would ask: do the standard methods for spurious correlations, or simply one conformal threshold per camera group, do better?

```bash
python group_robust.py --data ../data_512 --results ../results     # about 5 minutes on 2 cores
```

It needs E1's folds and calibration photos, E2's clinic pools, and for the two erasure rows the fold predictions of section 18 (`results/camera_fixes/`).

**What it does.** Every method is logistic regression on the same standardised features with the same C (0.003, E1's most frequent choice), refitted in each of E1's 10 outer folds; C1 (α = 0.1) is fitted on the fold's calibration photos. For training, the camera groups are recomputed from each fold's training photos only.

| Method | What it changes | Source |
|---|---|---|
| Plain training (ERM) | nothing | |
| Camera-decorrelated weights (ours) | each photo weighted by P(class) / P(class \| camera group) | section 18 |
| Group-balanced weights | every (class, camera group) cell weighs the same | Sagawa et al. 2020; Idrissi et al. 2022 |
| DFR | group-balanced subsamples, 10 models averaged (C = 0.03) | Kirichenko et al. 2023 |
| Group DRO | minimises the loss of the worst (class, camera group) cell: multiplicative weights over the 9 cells | Sagawa et al. 2020 |
| JTT | the photos the plain model gets wrong (in inner folds) weighted ×5 or ×20; needs no camera labels | Liu et al. 2021 |
| Per-group thresholds | plain model; C1 with one threshold per class and camera group, from the fold's calibration photos, or cross-fitted from the other nine folds' out-of-fold scores (about 6 times more photos per cell) | Mondrian conformal prediction |
| Erasure (LEACE), two kinds | read from section 18, for reference | Belrose et al. 2023 |

**Evaluation.** Every APTOS photo is decided once, by the fold that held it out. Per camera group: the two targets, with 95 % Wilson intervals. On the 55 referable eyes from mostly-healthy (H) cameras: exact McNemar tests against plain training and against our weights. At IDRiD and Messidor-2: each trained method refitted on all of APTOS, with APTOS's thresholds (B6) and with C2 (200 draws, the same for every method).

**Outputs** (in `results/group_robust/`): `summary.md`, `metrics.csv`, `by_group.csv`, `mcnemar_h.csv`, `mcnemar_healthy.csv` (the cost in healthy eyes, tested against ours), `clinics.csv`, `oof_<method>.csv.gz` (every method's out-of-fold probabilities and decisions) and `group_robust.png` (each method's worst camera group against its share of healthy eyes sent home).

**Findings (4 Nov 2026):**

- **Plain training fails the worst camera group:** it sends home 43.6 % (95 % CI 31–57) of the referable eyes from H cameras, against the 10 % target.
- **Six methods bring every camera group under 10 % and keep urgent referral at 90 % or more in every group:** our weights (9.1 % on H cameras), group-balanced weights (5.5 %), Group DRO (3.6 %), DFR (0.0 %), cross-fitted per-group thresholds (7.3 %) and per-group thresholds from the fold's own calibration photos (0.0 %, but only by referring 812 of every 1,000 eyes). Each sends home significantly fewer H-camera referable eyes than plain training (exact McNemar, p ≤ 4 × 10⁻⁶). With 55 eyes, none is significantly better than ours on that group (p ≥ 0.06).
- **They differ in what they cost.** Healthy eyes sent home: ours 84.3 %, group-balanced 76.7 %, cross-fitted thresholds 75.8 %, Group DRO 70.7 %, DFR 26.4 %, thresholds from the fold's own calibration photos 15.8 % (plain training 87.6 %). Our weights meet the targets at the smallest cost, and the difference is not chance: on the 2,175 healthy APTOS eyes, every other passing method sends home significantly fewer (exact McNemar; group-balanced weights: 0 eyes that only it sends home against 165 that only ours does, p = 4 × 10⁻⁵⁰; cross-fitted thresholds 10 against 194, p = 2 × 10⁻⁴⁵). A plausible reason: they keep each class's share of the training data and remove only the link between camera and class, while the other methods also change the class mix (DFR also discards most photos: AUROC 0.877).
- **JTT, which needs no camera labels, falls short:** with ×5 it still sends home 21.8 % of H-camera referable eyes; with ×20, 7.3 %, but then urgent referral falls below 90 % on H and M cameras (86.7 % and 89.3 %).
- **Per-group thresholds from the fold's own calibration photos are useless:** H cameras have too few sick calibration photos (a class needs at least 9), so their thresholds are infinite and the rule refers 812 of every 1,000 eyes. Cross-fitting supplies the photos and works, at a higher cost than our weights.
- **Erasure does not work** (section 18): 52.7 % and 36.4 % of H-camera referable eyes sent home.
- **Fairness across APTOS's cameras does not carry over to a new clinic.** With APTOS's thresholds (B6), every method fails at Messidor-2 (36–52 % of urgent eyes referred urgently). After C2, ours sends home the most healthy eyes at IDRiD (53.7 % with 100 local labels, against 29.8–48.4 % for the other methods, plain training 46.6 %) and is level with plain training at Messidor-2 (25.2 % against 26.0 % with 300 labels; the others 17.2–22.6 %). So local recalibration remains necessary, and our weights do not make it worse.

**Each baseline at its best setting (`group_frontier.py`).** The comparison above runs every method at one standard setting. A baseline might win at another, so each family's own knob is swept under the same protocol: the strength t of our weights, w = (P(class) / P(class | camera group))^t, and of group-balanced weights; Group DRO's step; JTT's upweighting factor; DFR's C. For each family we then take the setting that sends the most healthy eyes home among those meeting both targets in every camera group. That setting is chosen with hindsight on the very results it is judged by, which favours the baselines.

```bash
python group_frontier.py --data ../data_512 --results ../results     # about 10 minutes on 2 cores
```

Outputs (in `results/group_frontier/`): `summary.md`, `frontier.csv` and `frontier.png`. The standard settings reproduce the numbers above exactly (a built-in check).

- **Even with hindsight, no baseline matches ours.** Healthy eyes sent home at the best setting that meets both targets in every group: ours 84.3 % (t = 1); group-balanced weights 80.5 % (strength 0.75; on the healthy eyes 10 against 92, p = 9 × 10⁻¹⁸); Group DRO 76.0 % (step 0.05); DFR 27.7 % (C = 0.3). No setting of JTT meets both targets: at ×50 every group is under 10 %, but only 66.7 % of the worst group's urgent eyes are referred urgently.
- **t = 1 is also the right strength of our own weights.** At t = 0.75, 12.7 % of H-camera referable eyes are still sent home; t = 1.5 meets the target but sends home only 71.9 % of healthy eyes. t = 1 is the value at which the camera says nothing about the class, so it was not tuned.

## 24. Additional work: label-efficient recalibration at a new clinic, C2-IW (`label_efficient.py`)

Also **not** a deliverable: a new method that extends C2.

**The problem.** C2 needs labelled local photos, and a class needs at least 9 of them before its threshold is finite (α = 0.1); until then the class is in every set and every eye is referred. Urgent eyes are rare (6.3 % of Messidor-2), so with random photos C2 needs about 300 labels before it sends a quarter of healthy eyes home (section 13).

**The method.** Choose which photos to label, with known probabilities that favour the rare classes, and undo the bias with weights.

1. **Rarity score** of every unlabelled local photo, from the APTOS model alone: r(x) = Σ_c p(c | x) / m_c, where m_c is the mean of p(c | x) over the clinic's photos. It is mixed with uniform sampling for safety: r'(x) = (1 − λ) + λ · r(x) / 3, whose mean is 1.
2. **Sampling.** Each exam group (the eyes of one exam together) is labelled independently, with probability π proportional to its mean r', capped at 1, so that k photos are labelled on average.
3. **Weighted thresholds.** For each class, the labelled photos are weighted by w = 1/π, which restores the clinic's own distribution of scores (inverse-probability weighting). The threshold is the weighted quantile at level (1 − α)(1 + 1/n_eff), where n_eff = (Σw)² / Σw² is the effective number of labelled photos of that class. With equal weights this is exactly C2.

**What the threshold is.** It is the exact weighted conformal threshold (Tibshirani et al. 2019) with the test photo's own weight replaced by its expected value under the clinic's class distribution, Σw² / Σw: with that plug-in, the exact rule's level (1 − α)(Σw + w_test) / Σw becomes (1 − α)(1 + 1/n_eff). The exact rule protects photos that were unlikely to be labelled by giving them larger thresholds; the plug-in protects them only on average, so its coverage is approximate. The more uneven the weights (the larger λ), the larger the possible shortfall. Because of the uniform part, no weight exceeds 1/(1 − λ) times its value under uniform sampling. We use **λ = 0.25**, the largest value of the grid (0.25, 0.5, 0.75, 1) whose urgent coverage stays within Monte Carlo noise of 90 % in every setting.

```bash
python label_efficient.py --data ../data_512 --results ../results     # 1 minute on a 4-core laptop, 5 in a 2-core container
```

It needs E2's all-APTOS model (`results/e2/aptos_model.csv.gz`).

**Evaluation.** 1,000 draws per clinic and label budget k. In each draw the clinic's exam groups are split at random into two halves: labels are drawn from half A, and the rule is tested on half B, which plays the clinic's future patients. (Testing on the photos left unlabelled would be biased against rarity sampling, which takes the photos it prefers out of that set.) Every λ and every threshold rule uses the same draws: the same split and the same lottery numbers for the exam groups (common random numbers), so their differences are not sampling noise. The rules:
- C2 (random photos, as in E2);
- C2-IW at λ = 0.25, 0.5, 0.75 and 1;
- the same rarity sampling without the weights;
- exact weighted conformal prediction, which puts the test photo's own weight at +∞.

The settings are Messidor-2 (6.3 % urgent), IDRiD (30 % urgent) and APTOS resampled to 5 % referable eyes (all non-referable APTOS photos and referable ones sampled by grade; the model's out-of-fold probabilities).

**Outputs** (in `results/label_efficient/`): `summary.md` (per clinic, the main rules at every k, with Monte Carlo standard errors; then a table of every λ and rule), `draws.csv.gz` (every draw), `curves.csv` (means, standard errors and the share of draws below 90 % coverage) and `label_efficient.png`.

**Findings (5 Nov 2026), C2-IW with λ = 0.25:**

- **Coverage on target everywhere.** Mean urgent coverage is 90.5 % or more at every budget in every setting (lowest: 90.5 ± 0.3 % at Messidor-2 with 200 labels).
- **About a third fewer labels at Messidor-2.** Healthy eyes sent home: 13.5 % with 100 labels (C2 needs about 150 for 14.8 %), 22.8 % with 150 (C2: 20.8 % with 200) and 25.6 % with 200 (C2: 26.0 % with 300). At most 6.8 % of referable eyes are sent home.
- **At 5 % prevalence, recalibration becomes possible.** With 300 labels, C2-IW sends home 62.8 % of healthy eyes, with urgent coverage 91.6 %. C2 sees about 5 urgent photos and sends home 3.4 %.
- **Nothing to gain at IDRiD**, where urgent eyes are common (30 %): C2-IW sends home slightly fewer healthy eyes than C2 (35.2 % against 36.9 % with 50 labels; 10.1 % against 15.0 % with 30), because uneven weights cost some effective sample size. Coverage stays on target.
- **λ = 0.5 saves more but falls short where the model ranks poorly.** At Messidor-2 it sends home 19.2 % of healthy eyes with 100 labels, but its urgent coverage is 89.2–89.8 % at 100–200 labels (up to 2.3 standard errors below 90 %). At 5 % prevalence it stays on target and sends home 62.7 % with 200 labels. λ = 1 falls to 87.0–89.9 % at Messidor-2. The shortfall is largest at Messidor-2, where the APTOS model ranks eyes worst (AUROC 0.764): many urgent eyes there look healthy, so they are rarely labelled and carry large weights.
- **The weights are essential.** The same sampling without them covers only 85.6–88.8 % of Messidor-2's urgent eyes from 100 labels on (λ = 0.25), and 78–85 % at λ = 1.
- **Exact weighted conformal prediction is valid but gains nothing.** At λ = 0.25 it sends home 25.9 % of Messidor-2's healthy eyes with 300 labels, the same as C2 (26.0 %), and fewer at every smaller budget; at 5 % prevalence it sends home none.
- **Limits.** The plug-in's validity is shown by simulation in three settings, not proven, and λ = 0.25 was chosen from the same grid it is judged on (the λ table shows every value). Like C2, C2-IW holds on average over draws, not for each clinic's own draw. A per-clinic version, with C2-PAC's bound, is the natural next step.
