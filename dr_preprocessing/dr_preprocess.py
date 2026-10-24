#!/usr/bin/env python3
"""
dr_preprocess.py - raw fundus photos -> clean square images + one manifest table.

Datasets
  APTOS 2019      training and internal test     (Kaggle: train.csv + train_images/)
  IDRiD           new clinic 1                   (B. Disease Grading: 413 train + 103 test)
  Messidor-2      new clinic 2                   (images from ADCIS + adjudicated grades, Krause et al.)

Per image
  1. read the raw photo (large JPEGs are decoded at reduced size, which is much faster)
  2. find the circular field of view (FOV) with a RANSAC circle fit; this also works when
     the circle is cut off at the top and bottom, as in many APTOS photos
  3. crop a square around that circle (black padding where the circle leaves the photo),
     resize to SIZE x SIZE and black out everything outside the circle
  4. compute simple image-quality features and a small signature used to find duplicates

For the whole set
  5. find near-duplicate images (cosine similarity of the signatures) and group them
  6. map DR grade 0-4 to the three referral classes: 0-1 no referral, 2 routine, 3-4 urgent
  7. put APTOS images into 10 cross-validation folds, stratified by grade, keeping each
     duplicate group inside one fold
  8. write manifest.csv, excluded.csv, duplicates.csv, summary.md and QC contact sheets

Runs are resumable: finished images are cached in OUT/cache.jsonl and skipped next time.
Usage: python dr_preprocess.py --help   (examples in README.md)
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

__version__ = "1.1"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
REF3_NAMES = {0: "no_referral", 1: "routine", 2: "urgent"}

# Published grade counts (grades 0..4). Only used to check that the right label files were read.
#   APTOS: train.csv of the Kaggle competition.  IDRiD: Porwal et al., Med. Image Anal. 2020, Table 4.
#   Messidor-2: adjudicated grades of Krause et al., Ophthalmology 2018 (1,744 gradable images).
EXPECTED_COUNTS = {
    ("aptos", "train"): (1805, 370, 999, 193, 295),
    ("idrid", "train"): (134, 20, 136, 74, 49),
    ("idrid", "test"): (34, 5, 32, 19, 13),
    ("messidor2", "test"): (1017, 270, 347, 75, 35),
}

QUALITY_COLS = ["q_valid_frac", "q_mean", "q_std", "q_r", "q_g", "q_b",
                "q_dark", "q_sat", "q_lapvar", "q_entropy"]


def grade_to_ref3(grade: int) -> int:
    """DR grade 0-4 -> referral class: 0 = no referral (0-1), 1 = routine (2), 2 = urgent (3-4)."""
    return 0 if grade <= 1 else (1 if grade == 2 else 2)


# --------------------------------------------------------------------------------------------
# Image operations (also importable on their own, e.g. `from dr_preprocess import detect_fov`)
# --------------------------------------------------------------------------------------------

def read_rgb(path, hint: int = 1024):
    """Read an image as an RGB uint8 array. JPEGs are decoded at the smallest 1/2, 1/4 or 1/8
    scale that keeps both sides >= `hint` pixels. Returns (array, original_width, original_height)."""
    with Image.open(path) as im:
        ow, oh = im.size
        if im.format == "JPEG":
            im.draft("RGB", (hint, hint))
        orientation = im.getexif().get(0x0112, 1)
        im = ImageOps.exif_transpose(im).convert("RGB")
        arr = np.array(im)
    if orientation in (5, 6, 7, 8):          # rotated by 90 degrees in the EXIF data
        ow, oh = oh, ow
    return arr, ow, oh


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = mask
    flood = pad.copy()
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    pad[flood == 0] = 1                      # background not reachable from the corner = holes
    return pad[1:-1, 1:-1]


def _kasa(x, y):
    """Least-squares circle through points (x, y). Returns (cx, cy, r) or None."""
    A = np.c_[x, y, np.ones(len(x))]
    (D, E, F), *_ = np.linalg.lstsq(A, -(x ** 2 + y ** 2), rcond=None)
    cx, cy = -D / 2, -E / 2
    r2 = cx * cx + cy * cy - F
    return (float(cx), float(cy), math.sqrt(r2)) if r2 > 0 else None


def _fit_circle_ransac(pts: np.ndarray, n_iter: int = 400, seed: int = 0):
    """RANSAC circle fit on boundary points, refined by least squares on the inliers.
    Straight cut-off edges and camera notches do not lie on the circle, so they are ignored."""
    rng = np.random.default_rng(seed)
    n = len(pts)
    i = rng.integers(0, n, n_iter)
    jit = rng.integers(-(n // 12), n // 12 + 1, (n_iter, 2))
    j = (i + n // 3 + jit[:, 0]) % n          # half the triples are spread along the contour
    k = (i + 2 * n // 3 + jit[:, 1]) % n
    half = n_iter // 2                          # the other half are fully random
    j[half:] = rng.integers(0, n, n_iter - half)
    k[half:] = rng.integers(0, n, n_iter - half)
    ax, ay, bx, by, qx, qy = pts[i, 0], pts[i, 1], pts[j, 0], pts[j, 1], pts[k, 0], pts[k, 1]
    d = 2 * (ax * (by - qy) + bx * (qy - ay) + qx * (ay - by))
    ok = np.abs(d) > 1e-6
    d = np.where(ok, d, 1.0)
    a2, b2, q2 = ax ** 2 + ay ** 2, bx ** 2 + by ** 2, qx ** 2 + qy ** 2
    ux = (a2 * (by - qy) + b2 * (qy - ay) + q2 * (ay - by)) / d
    uy = (a2 * (qx - bx) + b2 * (ax - qx) + q2 * (bx - ax)) / d
    r = np.hypot(ax - ux, ay - uy)
    dist = np.abs(np.hypot(pts[None, :, 0] - ux[:, None], pts[None, :, 1] - uy[:, None]) - r[:, None])
    tol = np.maximum(1.0, 0.01 * r)[:, None]
    count = np.where(ok, (dist < tol).sum(1), 0)
    best = int(np.argmax(count))
    inliers = dist[best] < tol[best]
    if inliers.sum() < 20:
        return None
    fit = _kasa(pts[inliers, 0], pts[inliers, 1])
    for _ in range(2):
        if fit is None:
            return None
        dd = np.abs(np.hypot(pts[:, 0] - fit[0], pts[:, 1] - fit[1]) - fit[2])
        inliers = dd < max(1.0, 0.01 * fit[2])
        if inliers.sum() < 20:
            break
        fit = _kasa(pts[inliers, 0], pts[inliers, 1])
    return fit


def _circle_iou(mask: np.ndarray, circle) -> float:
    hs, ws = mask.shape
    cx, cy, r = circle
    yy, xx = np.ogrid[:hs, :ws]
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r          # circle clipped to the photo
    m = mask.astype(bool)
    return float(np.logical_and(disc, m).sum() / max(np.logical_or(disc, m).sum(), 1))


def detect_fov(img: np.ndarray, work: int = 640):
    """Find the circular field of view.
    Returns ((cx, cy, r) in pixels of `img`, method, iou) where method is 'fit' (RANSAC circle),
    'bbox' (circle from the bounding box) or 'none' (nothing found: whole photo used), and iou is
    the overlap between the chosen circle and the thresholded FOV mask (1.0 = perfect)."""
    h, w = img.shape[:2]
    s = min(1.0, work / max(h, w))
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) \
        if s < 1 else img
    s = small.shape[1] / w
    v = cv2.medianBlur(np.ascontiguousarray(small.max(axis=2)), 5)
    hs, ws = v.shape
    k = max(3, min(hs, ws) // 40)
    corners = np.concatenate([v[:k, :k].ravel(), v[:k, -k:].ravel(), v[-k:, :k].ravel(), v[-k:, -k:].ravel()])
    thresh = max(15.0, float(np.median(corners)) + 10.0)       # black border is ~0-10
    mask = (v > thresh).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return (w / 2, h / 2, max(w, h) / 2), "none", 0.0
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = _fill_holes((lab == biggest).astype(np.uint8))
    x0, y0, bw, bh = cv2.boundingRect(mask)
    bbox_circle = (x0 + bw / 2, y0 + bh / 2, max(bw, bh) / 2)
    candidates = [("bbox", bbox_circle, _circle_iou(mask, bbox_circle))]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        pts = max(contours, key=len)[:, 0, :].astype(np.float64)
        m = 2                                                   # drop points on the photo edge
        keep = (pts[:, 0] > m) & (pts[:, 0] < ws - 1 - m) & (pts[:, 1] > m) & (pts[:, 1] < hs - 1 - m)
        if keep.sum() >= 30:
            fit = _fit_circle_ransac(pts[keep])
            if fit is not None and 0.2 * min(hs, ws) < fit[2] < 1.5 * max(hs, ws):
                candidates.append(("fit", fit, _circle_iou(mask, fit)))
    method, (cx, cy, r), iou = candidates[0]
    if len(candidates) == 2 and candidates[1][2] >= iou - 0.01:   # prefer the fitted circle unless clearly worse
        method, (cx, cy, r), iou = candidates[1]
    return (cx / s, cy / s, r / s), method, iou


@lru_cache(maxsize=8)
def _disc(size: int, frac: float) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2
    return (xx - c) ** 2 + (yy - c) ** 2 <= (frac * size / 2) ** 2


def crop_to_circle(img: np.ndarray, circle, size: int = 512, mask_frac: float = 0.98):
    """Square crop around the FOV circle -> size x size, black outside the circle.
    Returns (image, valid) where valid marks pixels inside the circle that came from the photo
    (False in the black padding where the circle was cut off)."""
    h, w = img.shape[:2]
    cx, cy, r = circle
    side = max(2, int(round(2 * r)))
    x0, y0 = int(round(cx - r)), int(round(cy - r))
    canvas = np.zeros((side, side, 3), np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x0 + side), min(h, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    out = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA if side > size else cv2.INTER_CUBIC)
    f = size / side
    yy, xx = np.ogrid[:size, :size]
    inside_photo = ((xx >= (sx0 - x0) * f + 1) & (xx < (sx1 - x0) * f - 1) &
                    (yy >= (sy0 - y0) * f + 1) & (yy < (sy1 - y0) * f - 1))
    valid = _disc(size, mask_frac) & inside_photo
    out[~valid] = 0
    return out, valid


def valid_mask(img: np.ndarray, mask_frac: float = 0.98) -> np.ndarray:
    """Valid-pixel mask for an image saved by this script (inside the circle and not padding)."""
    return _disc(img.shape[0], mask_frac) & (img.max(axis=2) > 0)


def quality_features(img: np.ndarray, valid: np.ndarray, mask_frac: float = 0.98) -> dict:
    """Simple image-quality features, computed inside the valid FOV only."""
    n = int(valid.sum())
    if n < 500:
        return {c: float("nan") for c in QUALITY_COLS}
    f = img.astype(np.float32)
    R, G, B = f[..., 0], f[..., 1], f[..., 2]
    lum = 0.299 * R + 0.587 * G + 0.114 * B
    L = lum[valid]
    hist = np.bincount(np.clip(L / 4, 0, 63).astype(np.int64), minlength=64).astype(np.float64)
    p = hist[hist > 0] / n
    inner = cv2.erode(valid.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool)
    lap = cv2.Laplacian(G, cv2.CV_32F, ksize=3)
    return {
        "q_valid_frac": n / int(_disc(img.shape[0], mask_frac).sum()),  # share of the circle present
        "q_mean": float(L.mean() / 255),                                  # brightness
        "q_std": float(L.std() / 255),                                    # contrast
        "q_r": float(R[valid].mean() / 255),
        "q_g": float(G[valid].mean() / 255),
        "q_b": float(B[valid].mean() / 255),
        "q_dark": float((L < 25).mean()),                                 # share of very dark pixels
        "q_sat": float((img.max(axis=2)[valid] >= 250).mean()),           # share of saturated pixels
        "q_lapvar": float(lap[inner].var()) if inner.sum() > 500 else float("nan"),  # sharpness
        "q_entropy": float(-(p * np.log2(p)).sum()),                      # histogram entropy (bits)
    }


def signature(img: np.ndarray, valid: np.ndarray, grid: int = 32, frac: float = 0.67) -> np.ndarray:
    """Unit-length vector used to find duplicates: high-pass green channel (vessel pattern) of the
    central square of the FOV, downsampled to grid x grid. Robust to re-saving, resizing,
    brightness changes and different black borders; different photos have cosine similarity ~0."""
    g = img[..., 1].astype(np.float32)
    s = img.shape[0]
    m = valid.astype(np.float32)
    sigma = s / 30
    local_mean = cv2.GaussianBlur(g * m, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(m, (0, 0), sigma), 1e-3)
    hp = (g - local_mean) * m                 # masked high-pass: no artefacts at cut-off edges
    half, c = int(s / 2 * frac), s // 2
    crop = np.ascontiguousarray(hp[c - half:c + half, c - half:c + half])
    v = cv2.resize(crop, (grid, grid), interpolation=cv2.INTER_AREA).ravel()
    v = v - v.mean()
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def enhance_clahe(img: np.ndarray, mask_frac: float = 0.98, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """CLAHE on the L channel (LAB colour space). Optional; not applied to the saved images."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(lab[..., 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    out[~valid_mask(img, mask_frac)] = 0
    return out


def enhance_graham(img: np.ndarray, mask_frac: float = 0.98, alpha: float = 4.0) -> np.ndarray:
    """Ben Graham's local-colour normalisation (Kaggle DR 2015): 4 * (image - local mean) + 128.
    The local mean ignores the black background. Optional; not applied to the saved images."""
    valid = valid_mask(img, mask_frac)
    f = img.astype(np.float32)
    m = valid.astype(np.float32)
    sigma = img.shape[0] / 30
    den = np.maximum(cv2.GaussianBlur(m, (0, 0), sigma), 1e-3)[..., None]
    local = cv2.GaussianBlur(f * m[..., None], (0, 0), sigma) / den
    out = np.clip(alpha * (f - local) + 128, 0, 255).astype(np.uint8)
    out[~valid] = 0
    return out


def save_image(rgb: np.ndarray, path: Path, fmt: str = "png", jpg_quality: int = 95) -> None:
    """Write atomically (no half-written files if interrupted); works with non-ASCII paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_PNG_COMPRESSION, 3] if fmt == "png" else [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)]
    ok, buf = cv2.imencode("." + fmt, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), params)
    if not ok:
        raise IOError(f"could not encode {path}")
    tmp = path.with_name(path.name + ".tmp")
    buf.tofile(str(tmp))
    os.replace(tmp, path)


def process_one(task: dict) -> dict:
    """Worker: one raw photo -> saved image + features. Never raises; errors are reported."""
    cv2.setNumThreads(1)
    res = {"uid": task["uid"], "src": task["src"], "status": "ok", "error": ""}
    try:
        img, ow, oh = read_rgb(task["src"], hint=2 * task["size"])
        circle, method, iou = detect_fov(img)
        out, valid = crop_to_circle(img, circle, task["size"], task["mask_frac"])
        q = quality_features(out, valid, task["mask_frac"])
        sig = signature(out, valid)
        save_image(out, Path(task["out_dir"]) / task["out_rel"], task["fmt"], task["jpg_quality"])
        sc = ow / img.shape[1]
        res.update(out_rel=task["out_rel"], orig_w=ow, orig_h=oh, fov_method=method, fov_iou=round(iou, 4),
                   fov_cx=round(circle[0] * sc, 1), fov_cy=round(circle[1] * sc, 1),
                   fov_r=round(circle[2] * sc, 1), **{k: round(v, 6) for k, v in q.items()},
                   sig=base64.b64encode(sig.astype(np.float16).tobytes()).decode("ascii"))
    except Exception as e:  # noqa: BLE001 - report any failure for this image and carry on
        res.update(status="error", error=f"{type(e).__name__}: {str(e).replace(task['src'], Path(task['src']).name)}")
    return res


# --------------------------------------------------------------------------------------------
# Labels and dataset discovery
# --------------------------------------------------------------------------------------------

ID_COLS = ["image_id", "image", "id_code", "image name", "image_name", "filename", "file", "name", "id"]
GRADE_COLS = ["adjudicated_dr_grade", "diagnosis", "retinopathy grade", "dr_grade", "dr grade",
              "grade", "level", "label"]
DME_COLS = ["adjudicated_dme", "risk of macular edema", "dme"]
GRADABLE_COLS = ["adjudicated_gradable", "gradable"]


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _pick_col(columns, candidates):
    lookup = {_norm(c): c for c in columns}
    for cand in candidates:
        if _norm(cand) in lookup:
            return lookup[_norm(cand)]
    return None


def image_key(name) -> str:
    """File name or label id -> matching key (no folder, no image extension, lower case)."""
    base = re.split(r"[\\/]", str(name).strip())[-1]
    stem, ext = os.path.splitext(base)
    return (stem if ext.lower() in IMG_EXTS else base).lower()


def read_label_table(path) -> pd.DataFrame:
    """Read a label CSV whatever its column names (APTOS, IDRiD and Messidor-2 styles)."""
    df = pd.read_csv(path, sep=None, engine="python", dtype=str, skipinitialspace=True)
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    id_col, grade_col = _pick_col(df.columns, ID_COLS), _pick_col(df.columns, GRADE_COLS)
    if id_col is None or grade_col is None:
        raise ValueError(f"{path}: could not find image-name and DR-grade columns in {list(df.columns)}")
    dme_col, gradable_col = _pick_col(df.columns, DME_COLS), _pick_col(df.columns, GRADABLE_COLS)
    out = pd.DataFrame({"image_id": df[id_col].fillna("").astype(str).str.strip()})
    out["grade"] = pd.to_numeric(df[grade_col], errors="coerce")
    out["dme"] = pd.to_numeric(df[dme_col], errors="coerce") if dme_col else np.nan
    out["gradable"] = pd.to_numeric(df[gradable_col], errors="coerce") if gradable_col else np.nan
    out = out[out["image_id"] != ""].reset_index(drop=True)
    out["key"] = out["image_id"].map(image_key)
    return out


def _walk(root: Path, max_depth: int = 6):
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        if len(Path(dirpath).relative_to(root).parts) >= max_depth:
            dirnames[:] = []
        dirnames.sort()
        yield Path(dirpath), dirnames, sorted(filenames)


def _n_images(files) -> int:
    return sum(1 for f in files if os.path.splitext(f)[1].lower() in IMG_EXTS)


def index_images(folder: Path) -> dict:
    idx = {}
    for d, _, files in _walk(folder, 4):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                idx.setdefault(image_key(f), d / f)
    return idx


def _attach(tab: pd.DataFrame, idx: dict, dataset: str, split: str, label_file) -> pd.DataFrame:
    df = tab.copy()
    df.insert(0, "dataset", dataset)
    df.insert(1, "split", split)
    df["src_path"] = df["key"].map(lambda k: str(idx[k]) if k in idx else "")
    df["label_file"] = str(label_file)
    return df


def load_aptos(root, labels=None) -> pd.DataFrame:
    root = Path(root)
    if labels is None:
        labels = next((d / "train.csv" for d, _, files in _walk(root, 3) if "train.csv" in files), None)
    if labels is None or not Path(labels).is_file():
        raise FileNotFoundError(f"APTOS: no train.csv under {root} (or give --aptos-labels)")
    labels = Path(labels)
    img_dir = next((d / n for d, dirs, _ in _walk(root, 3) for n in dirs if n.lower() == "train_images"), None)
    img_dir = img_dir or labels.parent
    return _attach(read_label_table(labels), index_images(img_dir), "aptos", "train", labels)


def load_idrid(root) -> pd.DataFrame:
    """IDRiD 'B. Disease Grading' as downloaded (also accepts train/images + train/labels.csv)."""
    root = Path(root)
    grading = [d for d, _, _ in _walk(root, 4) if "disease grading" in d.name.lower()]
    if grading:                                   # ignore the segmentation part if it is there too
        root = min(grading, key=lambda d: len(d.parts))
    tree = list(_walk(root, 6))
    csvs = [d / f for d, _, files in tree for f in files if f.lower().endswith(".csv")]
    parts = []
    for split, word in (("train", "train"), ("test", "test")):
        labels = sorted(c for c in csvs if word in c.name.lower()) or \
            sorted(c for c in csvs if word in str(c.parent.relative_to(root)).lower())
        img_dirs = [d for d, _, files in tree if word in d.name.lower() and _n_images(files) > 0] or \
            [d for d, _, files in tree if d.name.lower() == "images" and word in d.parent.name.lower()
             and _n_images(files) > 0]
        if not labels or not img_dirs:
            print(f"  IDRiD: no {split} {'labels' if not labels else 'images'} found under {root}; skipping {split}")
            continue
        parts.append(_attach(read_label_table(labels[0]), index_images(img_dirs[0]), "idrid", split, labels[0]))
    if not parts:
        raise FileNotFoundError(f"IDRiD: no disease-grading images and labels found under {root}")
    return pd.concat(parts, ignore_index=True)


def read_exam_groups(path, known_keys) -> dict:
    """Messidor-2 examination list: any table whose rows list the images of one exam
    (the two eyes of one patient). Returns {image key: exam id}."""
    path = Path(path)
    rows = []
    if path.suffix.lower() in (".xls", ".xlsx"):
        rows = pd.read_excel(path, header=None, dtype=str).fillna("").values.tolist()
    else:
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            rows = [re.split(r"[,;\t]+", line.strip()) for line in fh]
    exam = {}
    for i, row in enumerate(rows):
        keys = [image_key(t) for t in row if str(t).strip()]
        keys = [k for k in keys if k in known_keys]
        if len(keys) >= 2:
            for k in keys:
                exam.setdefault(k, f"exam{i:04d}")
    return exam


def load_messidor2(root, labels=None, pairs=None):
    root = Path(root)
    tree = list(_walk(root, 4))
    dirs = [(d, _n_images(files)) for d, _, files in tree if _n_images(files) > 0]
    if not dirs:
        raise FileNotFoundError(f"Messidor-2: no images under {root}")
    img_dir = max(dirs, key=lambda t: (t[0].name.lower() == "images", t[1]))[0]
    tables = [d / f for d, _, files in tree for f in files if f.lower().endswith((".csv", ".xls", ".xlsx"))]
    if labels is None:
        for t in (t for t in tables if t.suffix.lower() == ".csv"):
            try:
                head = pd.read_csv(t, nrows=3, sep=None, engine="python")
            except Exception:  # noqa: BLE001
                continue
            if _pick_col(head.columns, ["adjudicated_dr_grade"]) is not None:
                labels = t
                break
    if labels is None:
        raise FileNotFoundError("Messidor-2: give the adjudicated grades with --messidor-labels "
                                "(messidor_data.csv, see README)")
    idx = index_images(img_dir)
    df = _attach(read_label_table(labels), idx, "messidor2", "test", labels)
    exam = {}
    if pairs is None:                            # look for the examination list next to the images
        for t in tables:
            if Path(t) != Path(labels):
                try:
                    exam = read_exam_groups(t, set(idx))
                except Exception:  # noqa: BLE001
                    exam = {}
                if len(exam) > 0.5 * len(idx):
                    pairs = t
                    break
                exam = {}
    elif Path(pairs).is_file():
        exam = read_exam_groups(pairs, set(idx))
    df["exam_id"] = df["key"].map(exam)
    unlabelled = sorted(set(idx) - set(df["key"]))
    return df, unlabelled, (str(pairs) if exam else "")


# --------------------------------------------------------------------------------------------
# Whole-set steps: duplicates, folds
# --------------------------------------------------------------------------------------------

class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def similar_pairs(sigs: np.ndarray, min_sim: float, block: int = 1024):
    """All pairs (i, j, cosine) with i < j and cosine >= min_sim, plus each image's best match."""
    n = len(sigs)
    X = sigs.astype(np.float32)
    best = np.full(n, -1.0, np.float32)
    out = []
    for s in range(0, n, block):
        S = X[s:s + block] @ X.T
        rows = np.arange(S.shape[0])
        S[rows, s + rows] = -1.0                 # ignore self-similarity
        best[s:s + block] = S.max(1)
        ii, jj = np.nonzero(S >= min_sim)
        gi = ii + s
        keep = jj > gi
        out += [(int(a), int(b), float(S[a - s, b])) for a, b in zip(gi[keep], jj[keep])]
    return out, best


def assign_folds(df: pd.DataFrame, n_splits: int, seed: int) -> tuple:
    """APTOS outer cross-validation folds: stratified by grade (0-4), each group (copies of one photo)
    kept inside one fold. Greedy assignment with a fixed-seed shuffle, so the folds are identical
    on every machine and library version (unlike StratifiedGroupKFold, whose shuffling changed
    between scikit-learn versions)."""
    fold = pd.Series(-1, index=df.index, dtype=int)
    sub = df[df["dataset"] == "aptos"]
    if len(sub) == 0:
        return fold, "no APTOS images"
    k = min(n_splits, sub["group_id"].nunique())
    if k < 2:
        return fold, "too few APTOS images for folds"
    counts = pd.crosstab(sub["group_id"], sub["grade"]).reindex(columns=range(5), fill_value=0).sort_index()
    C = counts.to_numpy(float)
    order = np.random.RandomState(seed).permutation(len(C))       # RandomState streams never change
    order = order[np.argsort(-C[order].sum(1), kind="stable")]     # larger groups first
    target = np.maximum(C.sum(0) / k, 1e-9)
    F = np.zeros((k, C.shape[1]))
    assign = np.empty(len(C), int)
    for g in order:
        delta = ((2 * (F - target) * C[g] + C[g] ** 2) / target).sum(1)   # change in squared deviation
        best = int(np.lexsort((F.sum(1), delta))[0])                       # ties -> the smaller fold
        F[best] += C[g]
        assign[g] = best
    fold.loc[sub.index] = sub["group_id"].map(dict(zip(counts.index, assign))).to_numpy()
    return fold, f"{k} folds" + ("" if k == n_splits else f" (reduced from {n_splits}: small sample)")


# --------------------------------------------------------------------------------------------
# QC contact sheets
# --------------------------------------------------------------------------------------------

def _font(size: int):
    for f in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _cell(images, caption, tile, font):
    """One labelled cell: one or more images side by side with a caption underneath."""
    w = tile * len(images) + 4 * (len(images) - 1)
    lines = caption.split("\n")[:3]
    cell = Image.new("RGB", (w, tile + 6 + 15 * len(lines)), (255, 255, 255))
    for i, arr in enumerate(images):
        im = Image.fromarray(arr)
        im.thumbnail((tile, tile))
        cell.paste(im, (i * (tile + 4) + (tile - im.width) // 2, (tile - im.height) // 2))
    draw = ImageDraw.Draw(cell)
    for k, line in enumerate(lines):
        draw.text((3, tile + 2 + 15 * k), line, fill=(25, 25, 25), font=font)
    return cell


def _sheet(cells, ncol, title, path: Path, font):
    if not cells:
        return
    cw, ch = max(c.width for c in cells), max(c.height for c in cells)
    nrow = math.ceil(len(cells) / ncol)
    sheet = Image.new("RGB", (ncol * (cw + 10) + 10, nrow * (ch + 10) + 44), (255, 255, 255))
    ImageDraw.Draw(sheet).text((10, 10), title, fill=(0, 0, 0), font=_font(20))
    for i, c in enumerate(cells):
        sheet.paste(c, (10 + (i % ncol) * (cw + 10), 44 + (i // ncol) * (ch + 10)))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)


def _load_out(out_dir: Path, rel: str) -> np.ndarray:
    return np.array(Image.open(out_dir / rel).convert("RGB"))


def _before_thumb(row, tile: int) -> np.ndarray:
    """Raw photo, downscaled, with the fitted circle drawn in green."""
    img, ow, _ = read_rgb(row["src_path"], hint=tile)
    s = tile * 1.6 / max(img.shape[:2])
    small = cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))),
                       interpolation=cv2.INTER_AREA).copy()
    f = small.shape[1] / ow
    cv2.circle(small, (round(row["fov_cx"] * f), round(row["fov_cy"] * f)), max(1, round(row["fov_r"] * f)),
               (0, 255, 0), 2)
    return small


def make_qc_sheets(man: pd.DataFrame, dup: pd.DataFrame, out_dir: Path, seed: int, thresholds: tuple):
    qc = out_dir / "qc"
    font = _font(12)
    rng = np.random.default_rng(seed)
    tile = 180

    def before_after(row, extra=""):
        cap = f"{row['image_id'][:24]}  grade {int(row['grade'])}\n{row['fov_method']} iou {row['fov_iou']:.3f}{extra}"
        try:
            before = _before_thumb(row, tile)
        except Exception:  # noqa: BLE001 - raw file moved: show the output only
            before = np.zeros((tile, tile, 3), np.uint8)
        return _cell([before, _load_out(out_dir, row["out_path"])], cap, tile, font)

    for ds, g in man.groupby("dataset"):
        pick = g.iloc[rng.permutation(len(g))[:16]]
        _sheet([before_after(r) for _, r in pick.iterrows()], 4,
               f"{ds}: 16 random images - raw photo with fitted circle | processed", qc / f"samples_{ds}.jpg", font)

    flagged = man[man["qc_flags"].fillna("") != ""]
    if len(flagged):
        order = flagged.sort_values("fov_iou").head(24)
        _sheet([before_after(r, f"\n{r['qc_flags']}") for _, r in order.iterrows()], 4,
               f"Flagged images (lowest circle fit first; {len(flagged)} flagged in total) - check these",
               qc / "flagged.jpg", font)

    cells = []
    for col, label, asc in (("q_mean", "darkest", True), ("q_mean", "brightest", False),
                            ("q_lapvar", "least sharp", True)):
        for _, r in man.sort_values(col, ascending=asc).head(6).iterrows():
            cells.append(_cell([_load_out(out_dir, r["out_path"])],
                               f"{label}: {r['dataset']}\n{col} {r[col]:.3g}  grade {int(r['grade'])}", tile, font))
    _sheet(cells, 6, "Image-quality extremes (rows: darkest, brightest, least sharp)", qc / "quality_extremes.jpg", font)

    if len(dup):
        paths = dict(zip(man["uid"], man["out_path"]))
        grades = dict(zip(man["uid"], man["grade"]))
        show = pd.concat([dup[dup["is_copy"]].sort_values("similarity").head(8),                      # weakest copies
                          dup[dup["grouped"] & ~dup["is_copy"]].sort_values("similarity").head(12),   # weakest same-eye
                          dup[~dup["grouped"]].head(8)])                                           # strongest others
        cells = []
        for _, d in show.iterrows():
            tag = "COPY" if d["is_copy"] else ("SAME EYE? (grouped)" if d["grouped"] else "not grouped")
            cap = (f"{tag}  sim {d['similarity']:.3f}\n"
                   f"grades {int(grades[d['uid_a']])} / {int(grades[d['uid_b']])}  {d['uid_a'].split('/')[0]}"
                   f"-{d['uid_b'].split('/')[0]}")
            cells.append(_cell([_load_out(out_dir, paths[d["uid_a"]]), _load_out(out_dir, paths[d["uid_b"]])],
                               cap, tile, font))
        _sheet(cells, 4, f"Similar pairs: weakest copies (>= {thresholds[0]}), weakest 'probably same eye' pairs "
                         f"({thresholds[1]}-{thresholds[0]}), strongest pairs not grouped", qc / "duplicates.jpg",
               font)

    cells = []
    for _, r in man.iloc[rng.permutation(len(man))[:4]].iterrows():
        base = _load_out(out_dir, r["out_path"])
        cells.append(_cell([base, enhance_clahe(base), enhance_graham(base)],
                           f"{r['image_id'][:30]}\nsaved | CLAHE | Graham", tile, font))
    _sheet(cells, 2, "Optional colour normalisation (not applied to the saved images)", qc / "enhancement_options.jpg",
           font)


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------

def _percent(counts):
    tot = sum(counts)
    return " / ".join(f"{100 * c / tot:.1f}%" for c in counts) if tot else "-"


def write_summary(path, man, exc, dup, threshold, fold_note, exam_note, config, elapsed, unlabelled, limit=0,
                  label_rows=None):
    L = ["# Preprocessing summary", "",
         f"- run: {time.strftime('%Y-%m-%d %H:%M')}, dr_preprocess.py {__version__}, python {platform.python_version()}, "
         f"OpenCV {cv2.__version__}, {elapsed / 60:.1f} min",
         f"- settings: size {config['size']} px, mask {config['mask_frac']}, format {config['fmt']}, "
         f"copy threshold {threshold[0]}, group threshold {threshold[1]}, seed {config['seed']}", ""]
    if limit:
        L += [f"**Trial run (--limit {limit}): only a random sample of each split was processed, so the counts "
              f"below describe the sample.**", ""]
    L += ["## Images", "", "| dataset / split | label rows | processed | excluded |" +
          (" not in sample |" if limit else ""), "|---|---|---|---|" + ("---|" if limit else "")]
    for (ds, sp), g in man.groupby(["dataset", "split"]):
        e = exc[(exc["dataset"] == ds) & (exc["split"] == sp)] if len(exc) else exc
        n_rows = (label_rows or {}).get((ds, sp), len(g) + len(e))
        L.append(f"| {ds} / {sp} | {n_rows} | {len(g)} | {len(e)} |" +
                 (f" {n_rows - len(g) - len(e)} |" if limit else ""))
    if len(exc):
        L += ["", "Excluded (details in excluded.csv): " +
              ", ".join(f"{n} x {r}" for r, n in Counter(exc["reason"]).most_common())]
    if unlabelled:
        L += ["", f"Messidor-2 images without a label row (not used): {len(unlabelled)}"]
    L += ["", "## Labels", "",
          "Grades 0/1/2/3/4 and referral classes (no referral 0-1 / routine 2 / urgent 3-4) of the processed images.",
          "The check compares all label rows with the published counts.", "",
          "| dataset / split | grades 0/1/2/3/4 | referral classes | shares | check |", "|---|---|---|---|---|"]
    for (ds, sp), g in man.groupby(["dataset", "split"]):
        gc = [int((g["grade"] == k).sum()) for k in range(5)]
        rc = [int((g["ref3"] == k).sum()) for k in range(3)]
        all_rows = pd.concat([g[["grade"]], exc[(exc["dataset"] == ds) & (exc["split"] == sp) &
                                                exc["grade"].between(0, 4)][["grade"]]]) if len(exc) else g[["grade"]]
        allc = tuple(int((all_rows["grade"] == k).sum()) for k in range(5))
        exp = EXPECTED_COUNTS.get((ds, sp))
        check = "-" if exp is None or limit else ("matches published counts" if allc == exp else f"differs from {exp}")
        L.append(f"| {ds} / {sp} | {'/'.join(map(str, gc))} | {'/'.join(map(str, rc))} | {_percent(rc)} | {check} |")
    L += ["", "## Field of view", "", "| dataset | fitted circle | bbox fallback | none | median IoU | flagged |",
          "|---|---|---|---|---|---|"]
    for ds, g in man.groupby("dataset"):
        mc = Counter(g["fov_method"])
        L.append(f"| {ds} | {mc.get('fit', 0)} | {mc.get('bbox', 0)} | {mc.get('none', 0)} | "
                 f"{g['fov_iou'].median():.3f} | {(g['qc_flags'].fillna('') != '').sum()} |")
    L += ["", "Flags (column qc_flags; flagged images are kept): no_fov = no circle found, whole photo used; "
          "fov_check = circle fit IoU < 0.95; cut_off = < 60% of the circle inside the photo; dark = mean brightness "
          "< 0.08; overexposed = > 5% saturated pixels; low_sharpness = lowest 2% sharpness within its dataset; "
          "upscaled = FOV diameter smaller than the output size.", "",
          "## Original photo sizes (a rough proxy for camera / site)", ""]
    for ds, g in man.groupby("dataset"):
        top = Counter(zip(g["orig_w"].astype(int), g["orig_h"].astype(int))).most_common(6)
        L.append(f"- {ds}: " + ", ".join(f"{w}x{h} ({n})" for (w, h), n in top) +
                 (f", {g.groupby(['orig_w', 'orig_h']).ngroups - len(top)} other sizes" if
                  g.groupby(['orig_w', 'orig_h']).ngroups > len(top) else ""))
    dup_thr, group_thr = threshold
    L += ["", "## Near-duplicates", ""]
    copies = dup[dup["is_copy"]] if len(dup) else dup
    if len(copies):
        multi = man[man["dup_cluster_size"] > 1]
        L += [f"- copies of the same photo (similarity >= {dup_thr}): {len(copies)} pairs, "
              f"{multi['dup_cluster'].nunique()} clusters with {len(multi)} images (column dup_cluster).",
              f"- copy clusters whose copies have different grades: "
              f"{man.loc[man['dup_grade_conflict'], 'dup_cluster'].nunique()}; different referral classes: "
              f"{man.loc[man['dup_class_conflict'], 'dup_cluster'].nunique()}."]
    else:
        L.append(f"- no copies (similarity >= {dup_thr}).")
    same_eye = dup[dup["grouped"] & ~dup["is_copy"]] if len(dup) else dup
    L.append(f"- probably the same eye in another photo ({group_thr} <= similarity < {dup_thr}): {len(same_eye)} "
             f"pairs. They only join split groups (group_id); some may be different eyes that look alike.")
    grouped = dup[dup["grouped"]] if len(dup) else dup
    cross = grouped[grouped["uid_a"].str.split("/").str[0] != grouped["uid_b"].str.split("/").str[0]] \
        if len(grouped) else grouped
    L.append(f"- grouped pairs across datasets: {len(cross)}" + (" - check these, they would leak between training "
                                                                  "and the external test sets" if len(cross) else ""))
    L += ["- duplicates.csv lists all pairs with similarity >= 0.5 (columns is_copy, grouped); see qc/duplicates.jpg.",
          "- similarity of each image to its closest other image: " +
          ", ".join(f"{q}% <= {man['dup_max_sim'].quantile(q / 100):.2f}" for q in (50, 90, 99)) + "."]
    if exam_note:
        L.append(f"- Messidor-2 examinations: {exam_note}")
    L += ["", "## APTOS cross-validation folds", "", f"- {fold_note}"]
    ap = man[(man["dataset"] == "aptos") & (man["fold"] >= 0)]
    if len(ap):
        L += ["", "| fold | images | no referral / routine / urgent |", "|---|---|---|"]
        for f, g in ap.groupby("fold"):
            rc = [int((g["ref3"] == k).sum()) for k in range(3)]
            L.append(f"| {f} | {len(g)} | {'/'.join(map(str, rc))} ({_percent(rc)}) |")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")
    return "\n".join(L)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Preprocess APTOS 2019, IDRiD and Messidor-2 fundus photos.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--aptos", help="APTOS folder (contains train.csv and train_images/)")
    p.add_argument("--aptos-labels", help="APTOS label file, if not train.csv inside --aptos")
    p.add_argument("--idrid", help="IDRiD folder containing 'B. Disease Grading'")
    p.add_argument("--messidor", help="Messidor-2 folder (contains the IMAGES folder)")
    p.add_argument("--messidor-labels", help="Messidor-2 adjudicated grades, e.g. messidor_data.csv")
    p.add_argument("--messidor-pairs", help="optional Messidor-2 examination list (images of one exam per row)")
    p.add_argument("--out", default="data_512", help="output folder")
    p.add_argument("--size", type=int, default=512, help="output width and height in pixels")
    p.add_argument("--mask-frac", type=float, default=0.98, help="radius of the kept circle, as a fraction")
    p.add_argument("--format", dest="fmt", choices=["png", "jpg"], default="png", help="output image format")
    p.add_argument("--jpg-quality", type=int, default=95, help="JPEG quality if --format jpg")
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)),
                   help="parallel processes (0 = no multiprocessing)")
    p.add_argument("--limit", type=int, default=0, help="only process N random images per dataset split (trial run)")
    p.add_argument("--time-budget", type=float, default=0,
                   help="stop starting new images after this many seconds (0 = no limit); run again to continue")
    p.add_argument("--dup-threshold", type=float, default=0.95,
                   help="signature similarity at which two images count as copies of the same photo")
    p.add_argument("--group-threshold", type=float, default=0.80,
                   help="lower similarity at which two images (probably the same eye) are kept in the same split group")
    p.add_argument("--folds", type=int, default=10, help="number of APTOS cross-validation folds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-qc", action="store_true", help="skip the QC contact sheets")
    p.add_argument("--overwrite", action="store_true", help="ignore the cache and reprocess everything")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {"size": args.size, "mask_frac": args.mask_frac, "fmt": args.fmt, "jpg_quality": args.jpg_quality,
              "seed": args.seed}
    cfg_path, cache_path = out_dir / "run_config.json", out_dir / "cache.jsonl"
    if cfg_path.exists() and not args.overwrite:
        old = json.loads(cfg_path.read_text())
        diff = {k: (old.get(k), v) for k, v in config.items() if k != "seed" and old.get(k) != v}
        if diff:
            sys.exit(f"{out_dir} was made with different settings {diff}. Use a new --out folder or --overwrite.")
    cfg_path.write_text(json.dumps(config, indent=2))
    if args.overwrite and cache_path.exists():
        cache_path.unlink()

    # ---- 1. label tables + image files
    print("Reading labels and finding images ...")
    tables, unlabelled, exam_note = [], [], ""
    if args.aptos:
        tables.append(load_aptos(args.aptos, args.aptos_labels))
    if args.idrid:
        tables.append(load_idrid(args.idrid))
    if args.messidor:
        mdf, unlabelled, pairs_file = load_messidor2(args.messidor, args.messidor_labels, args.messidor_pairs)
        n_exam = mdf["exam_id"].nunique()
        exam_note = (f"{n_exam} examinations read from {pairs_file}; both images of an exam share a group_id"
                     if pairs_file else "no examination list found, so the two eyes of a patient are not grouped "
                                        "(give --messidor-pairs)")
        tables.append(mdf)
    if not tables:
        sys.exit("Give at least one dataset: --aptos, --idrid and/or --messidor (see --help).")
    meta = pd.concat(tables, ignore_index=True)
    if "exam_id" not in meta:
        meta["exam_id"] = np.nan
    meta["stem"] = [Path(p).stem if p else k for p, k in zip(meta["src_path"], meta["image_id"].map(image_key))]
    meta["uid"] = meta["dataset"] + "/" + meta["split"] + "/" + meta["stem"]

    reason = pd.Series("", index=meta.index)
    reason[meta["uid"].duplicated(keep="first")] = "repeated label row"
    reason[(reason == "") & (meta["src_path"] == "")] = "image file not found"
    ungradable = (meta["gradable"] == 0) | ((meta["dataset"] == "messidor2") & meta["grade"].isna())
    reason[(reason == "") & ungradable] = "ungradable"
    bad = ~meta["grade"].isin([0, 1, 2, 3, 4])
    reason[(reason == "") & bad] = "missing or invalid grade"
    meta["reason"] = reason
    todo = meta[reason == ""].copy()
    if args.limit:
        todo = pd.concat([g.sample(n=min(args.limit, len(g)), random_state=args.seed)
                          for _, g in todo.groupby(["dataset", "split"], sort=False)])
    todo["out_path"] = [f"images/{d}/{s}/{st}.{args.fmt}" for d, s, st in zip(todo["dataset"], todo["split"], todo["stem"])]
    for (ds, sp), g in meta.groupby(["dataset", "split"]):
        print(f"  {ds}/{sp}: {len(g)} label rows, {int((g['reason'] == '').sum())} usable, "
              f"{int((g['reason'] != '').sum())} excluded")

    # ---- 2. per-image processing (cached, parallel)
    cache = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    cache[r["uid"]] = r
                except json.JSONDecodeError:
                    pass                          # a line cut off by an interruption
    tasks, results = [], {}
    for r in todo.itertuples():
        c = cache.get(r.uid)
        if c and c["status"] == "ok" and c.get("src") == r.src_path and c.get("out_rel") == r.out_path \
                and (out_dir / r.out_path).exists():
            results[r.uid] = c
        else:
            tasks.append({"uid": r.uid, "src": r.src_path, "out_dir": str(out_dir), "out_rel": r.out_path,
                          "size": args.size, "mask_frac": args.mask_frac, "fmt": args.fmt,
                          "jpg_quality": args.jpg_quality})
    print(f"Processing {len(tasks)} images ({len(results)} already done) with {args.workers or 1} process(es) ...")
    if tasks:
        t1, done = time.time(), 0
        deadline = t1 + args.time_budget if args.time_budget else None
        step = max(1, len(tasks) // 20)
        with open(cache_path, "a", encoding="utf-8") as fh:
            def record(res):
                nonlocal done
                results[res["uid"]] = res
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                done += 1
                if done % step == 0 or done == len(tasks):
                    rate = done / max(time.time() - t1, 1e-6)
                    print(f"  {done}/{len(tasks)}  {rate:.1f} img/s  ~{(len(tasks) - done) / rate / 60:.1f} min left",
                          flush=True)

            def out_of_time():
                return deadline is not None and time.time() > deadline

            queue = iter(tasks)
            if args.workers and args.workers > 1:
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    pending = set()

                    def refill():                  # keep a few tasks in flight; stop adding when out of time
                        while len(pending) < 2 * args.workers and not out_of_time():
                            t = next(queue, None)
                            if t is None:
                                return
                            pending.add(ex.submit(process_one, t))
                    refill()
                    while pending:
                        finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for fut in finished:
                            pending.discard(fut)
                            record(fut.result())
                        refill()
            else:
                for t in queue:
                    if out_of_time():
                        break
                    record(process_one(t))
        if done < len(tasks):
            print(f"\nTime budget reached: {done} images processed in this run, {len(tasks) - done} left. "
                  f"Run the same command again to continue (finished images are kept).")
            sys.exit(3)

    res = pd.DataFrame([results[u] for u in todo["uid"]])
    keep_cols = ["uid", "dataset", "split", "image_id", "grade", "dme", "exam_id", "src_path", "out_path"]
    man = todo[keep_cols].merge(res.drop(columns=["src", "out_rel"], errors="ignore"), on="uid", how="left")
    failed = man[man["status"] != "ok"]
    man = man[man["status"] == "ok"].reset_index(drop=True)
    exc = meta.loc[meta["reason"] != "", ["uid", "dataset", "split", "image_id", "grade", "src_path", "reason"]]
    if len(failed):
        f = failed[["uid", "dataset", "split", "image_id", "grade", "src_path"]].copy()
        f["reason"] = "processing error: " + failed["error"].astype(str)
        exc = pd.concat([exc, f], ignore_index=True)
        print(f"  {len(failed)} images failed - see excluded.csv")
    blank = (man["fov_method"] == "none") & (man["q_mean"].fillna(0) < 0.02)
    if blank.any():
        b = man.loc[blank, ["uid", "dataset", "split", "image_id", "grade", "src_path"]].copy()
        b["reason"] = "blank image (no field of view found)"
        exc = pd.concat([exc, b], ignore_index=True)
        man = man[~blank].reset_index(drop=True)
        print(f"  {int(blank.sum())} blank images excluded - see excluded.csv")
    if len(man) == 0:
        exc.to_csv(out_dir / "excluded.csv", index=False)
        sys.exit("No image was processed successfully; see excluded.csv.")

    # ---- 3. labels -> referral classes, quality flags
    man["grade"] = man["grade"].astype(int)
    man["orig_w"], man["orig_h"] = man["orig_w"].astype(int), man["orig_h"].astype(int)
    man["ref3"] = man["grade"].map(grade_to_ref3)
    man["ref3_name"] = man["ref3"].map(REF3_NAMES)
    man["referable"] = (man["grade"] >= 2).astype(int)
    flags = [[] for _ in range(len(man))]
    sharp_cut = man.groupby("dataset")["q_lapvar"].transform(lambda s: s.quantile(0.02))
    for i, r in enumerate(man.itertuples()):
        if r.fov_method == "none":
            flags[i].append("no_fov")
        elif r.fov_iou < 0.95:
            flags[i].append("fov_check")
        if r.q_valid_frac < 0.60:
            flags[i].append("cut_off")
        if r.q_mean < 0.08:
            flags[i].append("dark")
        if r.q_sat > 0.05:
            flags[i].append("overexposed")
        if r.q_lapvar <= sharp_cut.iloc[i]:
            flags[i].append("low_sharpness")
        if 2 * r.fov_r < args.size:
            flags[i].append("upscaled")
    man["qc_flags"] = [";".join(f) for f in flags]

    # ---- 4. near-duplicates -> copy clusters; likely same eye + Messidor-2 examinations -> split groups
    print("Searching for near-duplicates ...")
    group_thr = min(args.group_threshold, args.dup_threshold)
    sigs = np.stack([np.frombuffer(base64.b64decode(s), np.float16).astype(np.float32) for s in man["sig"]])
    pairs, best = similar_pairs(sigs, min_sim=min(0.5, group_thr))
    man["dup_max_sim"] = np.round(best, 4)
    dsu = _DSU(len(man))
    for a, b, s in pairs:
        if s >= args.dup_threshold:
            dsu.union(a, b)
    roots = [dsu.find(i) for i in range(len(man))]
    did = {r: f"d{n:05d}" for n, r in enumerate(dict.fromkeys(roots))}
    man["dup_cluster"] = [did[r] for r in roots]                 # copies of the same photo
    man["dup_cluster_size"] = man.groupby("dup_cluster")["uid"].transform("size")
    man["dup_grade_conflict"] = man.groupby("dup_cluster")["grade"].transform("nunique") > 1
    man["dup_class_conflict"] = man.groupby("dup_cluster")["ref3"].transform("nunique") > 1
    for a, b, s in pairs:                                         # probably the same eye, another photo
        if group_thr <= s < args.dup_threshold:
            dsu.union(a, b)
    for _, idx in man.groupby("exam_id").groups.items():          # the two eyes of one patient
        idx = list(idx)
        for b in idx[1:]:
            dsu.union(idx[0], b)
    roots = [dsu.find(i) for i in range(len(man))]
    gid = {r: f"g{n:05d}" for n, r in enumerate(dict.fromkeys(roots))}
    man["group_id"] = [gid[r] for r in roots]                    # keep together when splitting
    man["group_size"] = man.groupby("group_id")["uid"].transform("size")
    dup = pd.DataFrame([(man.at[a, "uid"], man.at[b, "uid"], round(min(s, 1.0), 4), int(man.at[a, "grade"]),
                         int(man.at[b, "grade"]), s >= args.dup_threshold, s >= group_thr) for a, b, s in pairs],
                       columns=["uid_a", "uid_b", "similarity", "grade_a", "grade_b", "is_copy", "grouped"])
    dup = dup.sort_values("similarity", ascending=False).reset_index(drop=True)

    # ---- 5. APTOS folds
    man["fold"], fold_note = assign_folds(man, args.folds, args.seed)

    # ---- 6. outputs
    cols = ["uid", "dataset", "split", "image_id", "grade", "ref3", "ref3_name", "referable", "dme", "fold",
            "group_id", "group_size", "dup_cluster", "dup_cluster_size", "dup_max_sim", "dup_grade_conflict",
            "dup_class_conflict", "exam_id",
            "out_path", "src_path", "orig_w", "orig_h", "fov_method", "fov_iou", "fov_cx", "fov_cy", "fov_r"] + \
        QUALITY_COLS + ["qc_flags"]
    man[cols].to_csv(out_dir / "manifest.csv", index=False)
    exc.to_csv(out_dir / "excluded.csv", index=False)
    dup.to_csv(out_dir / "duplicates.csv", index=False)
    if not args.no_qc:
        print("Drawing QC sheets ...")
        try:
            make_qc_sheets(man, dup, out_dir, args.seed, (args.dup_threshold, group_thr))
        except Exception as e:  # noqa: BLE001 - QC sheets are a convenience; never lose the run for them
            print(f"  QC sheets failed: {type(e).__name__}: {e}")
    text = write_summary(out_dir / "summary.md", man, exc, dup, (args.dup_threshold, group_thr), fold_note,
                         exam_note, config, time.time() - t0, unlabelled, args.limit,
                         meta.groupby(["dataset", "split"]).size().to_dict())
    print("\n" + text)
    print(f"Done. Outputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
