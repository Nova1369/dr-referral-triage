# Mid report code: preprocessing, features, models, E0

The scripts, run in this order:

| Step | Script | What it makes |
|---|---|---|
| 1 | `dr_preprocess.py` | clean 512 × 512 images from the raw APTOS 2019, IDRiD and Messidor-2 downloads, plus one table (`manifest.csv`) that every later step reads (sections 1–7) |
| 2 | `extract_features.py` | 2,048 ResNet-50 features per image, the input to the course models (section 8) |
| 3 | `run_models.py`, `summarize_e1.py` | E1: baselines B0–B2 and the 11 course models, 10-fold cross-validation on APTOS, with tables and a figure (section 9) |
| 4 | `e0_conformal.py` | E0: standard conformal prediction calibrated on APTOS, tested on IDRiD and Messidor-2, as in [8] (section 10) |
| 5 | `kmeans_check.py` | k-means: do the images group by grade, or by clinic and camera? (section 11) |
| extra | `camera_shortcut.py` | additional work, beyond the proposal: the APTOS camera shortcut and what it does at new clinics (section 12) |

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

Colours are **not** changed in the saved images. CLAHE and Ben Graham's normalisation are included as functions, so we can compare them later as an ablation.

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

## 12. Additional work: the APTOS camera shortcut (`camera_shortcut.py`)

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

**Findings so far (25 Oct 2026):**

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

**Planned next:**

- A tuned comparison of the fixes with the E1 protocol (nested cross-validation).
- Removing the camera information from the features as a second fix (linear concept erasure).
- Class-wise conformal coverage per camera group (a Mondrian version by camera).
- How the fixes interact with E2 recalibration.
