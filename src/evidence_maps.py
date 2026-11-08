#!/usr/bin/env python3
"""
evidence_maps.py - additional work: where in the photo does the model find its evidence?

Our logistic regression is linear in ResNet-50's 2,048 average-pooled features, so its log-odds split exactly into
a 7 x 7 map over the photo (class activation mapping, Zhou et al. 2016). For cell (i, j) of the last convolutional
layer, e_c(i, j) = w_c . F(i, j) / 49, and b_c + (the sum of e_c over the 49 cells) = logit_c. The maps explain the
very predictions the referral rules use; they are not an approximation.

Stages, in this order (heads and summary need no PyTorch; maps, deletion and examples need the photos and PyTorch):
  heads     the linear heads in feature space, checked against the saved predictions:
              E1's logistic regression in each outer fold (the setting E1 chose there, PCA folded into the weights),
              the camera-decorrelated model of camera_fixes.py in each fold,
              E2's all-APTOS models, plain and decorrelated (used for IDRiD and Messidor-2).
  maps      for every photo, the 7 x 7 maps of the three logits for the plain and the decorrelated model (APTOS: the
            fold model that never trained on the photo; IDRiD and Messidor-2: the all-APTOS model), and the share of
            each cell that lies on the retina (field of view). Checked: features recomputed from the photos match
            the saved ones, and the maps add up to the logits. Resumable (parts of 256 photos, --time-budget).
  deletion  a faithfulness test on a sample of APTOS urgent eyes: the cells with the most urgent evidence are
            blurred, a few at a time, and the urgent log-odds recomputed. Two controls: random cells anywhere, and
            random cells on the retina (blurring the black corners changes nothing).
            Resumable (chunks of 25 eyes).
  flip      the same photos mirrored left-right and top-bottom (200 per dataset): does the evidence move with the
            eye, or stay where it is in the frame? And how much do the predictions change for a mirrored eye?
  examples  evidence overlays for a few APTOS and IDRiD photos. Never Messidor-2: its licence forbids showing its
            images. The figure shows patients' photos, so keep it on your computer.
  summary   tables and figures from the saved maps.

Outputs (in --out, default ../results/evidence_maps): heads.npz, heads_check.csv, maps.npz, deletion.csv, flip.npz,
regions.csv.gz, summary.md, evidence_by_camera.png, deletion.png, flip.png, examples.png.

Usage:
  python evidence_maps.py heads    --data ../data_512 --results ../results
  python evidence_maps.py maps     --data ../data_512 --weights ../models/resnet50_ram-a26f946b.pth
  python evidence_maps.py deletion --data ../data_512 --weights ../models/resnet50_ram-a26f946b.pth
  python evidence_maps.py flip     --data ../data_512 --weights ../models/resnet50_ram-a26f946b.pth
  python evidence_maps.py examples --data ../data_512 --results ../results
  python evidence_maps.py summary  --data ../data_512 --results ../results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

GRID = 7                     # cells per side of ResNet-50's last layer at 224 px
SIZE = 224
CELL = SIZE // GRID          # 32 px
MODELS = ("plain", "dec")    # E1's logistic regression; the camera-decorrelated model
MODEL_NAMES = {"plain": "E1's logistic regression", "dec": "camera-decorrelated model"}
FOV_IN, FOV_OUT = 0.99, 0.01  # a cell is inside the retina, on its edge, or outside it
DEL_STEPS = [0, 1, 2, 4, 8, 16, 24]
DEL_RANDOM = 3
DEL_CHUNK = 25


def common(args):
    man = pd.read_csv(Path(args.data) / "manifest.csv")
    with np.load(Path(args.data) / "features" / "resnet50_224.npz") as z:
        X, uid = z["X"], z["uid"]
    assert (uid == man["uid"].astype(str).to_numpy()).all(), "features do not match manifest.csv"
    return man, X


def softmax(L):
    L = L - L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def out_dir(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------------------------------------------------------------
# stage 1: the linear heads in feature space
# --------------------------------------------------------------------------------------------

def lr_head(scaler, coef, intercept, pca=None, n=None):
    """Weights W (3 x 2048) and bias b with logits = X @ W.T + b, for StandardScaler -> [PCA, first n] -> LR."""
    Wz = coef if pca is None else coef @ pca.components_[:n]
    b = intercept.copy() if pca is None else intercept - pca.mean_ @ Wz.T
    b = b - (scaler.mean_ / scaler.scale_) @ Wz.T
    return Wz / scaler.scale_, b


def stage_heads(args):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    from c1_referral import camera_groups
    from camera_fixes import decorrelating_weights as cf_weights
    from e2_new_clinics import C_LR, decorrelating_weights as e2_weights

    def fit(Z, y, C, w=None):
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return LogisticRegression(C=C, max_iter=5000).fit(Z, y, sample_weight=w)

    t0 = time.time()
    out, R = out_dir(args), Path(args.results)
    man, X = common(args)
    y = man["ref3"].to_numpy(int)
    g = camera_groups(man).fillna("other").to_numpy()
    row = pd.Series(np.arange(len(man)), index=man["uid"])
    apt = np.flatnonzero((man["dataset"] == "aptos").to_numpy())
    best = pd.read_csv(R / "e1" / "best_params.csv")
    best = best[best["model"] == "lr"].set_index("fold")["best"].map(json.loads)
    chosen = pd.read_csv(R / "camera_fixes" / "chosen_C.csv")
    chosen = chosen[chosen["fix"] == "weights"].set_index("fold")["C"]

    names, Ws, bs, checks = [], [], [], []
    fold_of = np.full(len(man), -1)
    for k in range(10):
        f = pd.read_csv(R / "e1" / "lr" / f"fold{k}.csv.gz")
        test, cal = row[f.loc[f["role"] == "test", "uid"]].to_numpy(), row[f.loc[f["role"] == "cal", "uid"]].to_numpy()
        order = row[f["uid"]].to_numpy()
        fold_of[test] = k
        fit_idx = np.setdiff1d(apt, np.concatenate([test, cal]))
        # E1's logistic regression, its chosen setting in this fold
        n, C = best[k]["pca"], best[k]["C"]
        sc = StandardScaler().fit(X[fit_idx])
        pca = PCA(n_components=min(2048, len(fit_idx), X.shape[1]), svd_solver="full").fit(sc.transform(X[fit_idx]))
        m = fit(pca.transform(sc.transform(X[fit_idx]))[:, :n], y[fit_idx], C)
        W, b = lr_head(sc, m.coef_, m.intercept_, pca, n)
        L = X[order] @ W.T + b
        checks.append({"head": f"plain_f{k}", "model_vs_head": float(np.abs(L - m.decision_function(
            pca.transform(sc.transform(X[order]))[:, :n])).max()), "head_vs_saved": float(np.abs(
                softmax(L) - f[["p0", "p1", "p2"]].to_numpy()).max())})
        names.append(f"plain_f{k}"), Ws.append(W), bs.append(b)
        # the camera-decorrelated model of camera_fixes.py
        sc = StandardScaler().fit(X[fit_idx])
        m = fit(sc.transform(X[fit_idx]), y[fit_idx], float(chosen[k]), cf_weights(y[fit_idx], g[fit_idx]))
        W, b = lr_head(sc, m.coef_, m.intercept_)
        fw = pd.read_csv(R / "camera_fixes" / "weights" / f"fold{k}.csv.gz")
        L = X[row[fw["uid"]].to_numpy()] @ W.T + b
        checks.append({"head": f"dec_f{k}", "model_vs_head": float(np.abs(
            L - m.decision_function(sc.transform(X[row[fw["uid"]].to_numpy()]))).max()),
            "head_vs_saved": float(np.abs(softmax(L) - fw[["p0", "p1", "p2"]].to_numpy()).max())})
        names.append(f"dec_f{k}"), Ws.append(W), bs.append(b)
        print(f"fold {k} done ({time.time() - t0:.0f} s)", flush=True)

    # E2's all-APTOS models, for IDRiD and Messidor-2
    sc = StandardScaler().fit(X[apt])
    e2 = pd.read_csv(R / "e2" / "aptos_model.csv.gz")
    assert (e2["uid"].astype(str) == man["uid"].astype(str)).all()
    cam = camera_groups(man).to_numpy()
    for name, w in (("plain", None), ("dec", e2_weights(y[apt], cam[apt]))):
        m = fit(sc.transform(X[apt]), y[apt], C_LR, w)
        W, b = lr_head(sc, m.coef_, m.intercept_)
        L = X @ W.T + b
        ext = np.setdiff1d(np.arange(len(man)), apt)     # E2 saved out-of-fold probabilities for APTOS photos
        checks.append({"head": f"{name}_all", "model_vs_head": float(np.abs(L - m.decision_function(sc.transform(X))).max()),
                       "head_vs_saved": float(np.abs(softmax(L[ext]) - e2[[f"{name}_p{c}" for c in range(3)]].to_numpy()[ext]).max())})
        names.append(f"{name}_all"), Ws.append(W), bs.append(b)

    np.savez(out / "heads.npz", names=np.array(names), W=np.stack(Ws).astype(np.float64), b=np.stack(bs),
             fold_of=fold_of, uid=np.array(man["uid"].astype(str).tolist()))     # fixed-width text: no pickle
    ck = pd.DataFrame(checks)
    ck.to_csv(out / "heads_check.csv", index=False, float_format="%.3g")
    print(ck.to_string(index=False))
    print(f"wrote {out / 'heads.npz'} ({time.time() - t0:.0f} s)")


def heads_for_rows(heads, man):
    """Index of the plain and the decorrelated head for every photo (APTOS: its test fold; others: all-APTOS)."""
    names = list(heads["names"])
    fold = heads["fold_of"]
    out = np.zeros((len(man), 2), int)
    for i, k in enumerate(fold):
        out[i] = [names.index(f"plain_f{k}"), names.index(f"dec_f{k}")] if k >= 0 else \
                 [names.index("plain_all"), names.index("dec_all")]
    return out


# --------------------------------------------------------------------------------------------
# stage 2: the maps (PyTorch, photos)
# --------------------------------------------------------------------------------------------

def torch_setup(args):
    import torch
    from extract_features import build_model
    torch.set_num_threads(args.threads or os.cpu_count() or 1)
    return torch, build_model(Path(args.weights), torch.device("cpu"))


def load_224(path):
    """The model's input exactly as extract_features.py makes it, and the raw 224 px photo."""
    from PIL import Image
    from extract_features import IMAGENET_MEAN, IMAGENET_STD
    im = Image.open(path).convert("RGB")
    if im.size != (SIZE, SIZE):
        im = im.resize((SIZE, SIZE), Image.BICUBIC, reducing_gap=3.0)
    raw = np.asarray(im, np.uint8)
    return raw, ((raw.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


def to_input(raws):
    from extract_features import IMAGENET_MEAN, IMAGENET_STD
    a = (np.asarray(raws, np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(a.transpose(0, 3, 1, 2))


def last_layer(torch, model, x):
    """ResNet-50 up to its last convolutional block: N x 2048 x 7 x 7 (the features are its mean)."""
    with torch.inference_mode():
        t = torch.from_numpy(x).contiguous(memory_format=torch.channels_last)
        for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"):
            t = getattr(model, name)(t)
        return t.float().numpy()


def fov_cells(raw):
    """Share of each 32 x 32 cell that lies on the retina (the preprocessing blacks out everything else)."""
    on = raw.max(2) > 20
    return on.reshape(GRID, CELL, GRID, CELL).mean((1, 3)).astype(np.float32)


def stage_maps(args):
    torch, model = torch_setup(args)
    t0 = time.time()
    out = out_dir(args)
    parts = Path(args.parts) if args.parts else out / "parts_maps"
    parts.mkdir(parents=True, exist_ok=True)
    man, X = common(args)
    heads = np.load(out / "heads.npz")
    assert (heads["uid"] == man["uid"].astype(str).to_numpy()).all()
    W, b = heads["W"], heads["b"]
    hr = heads_for_rows(heads, man)
    paths = [str(Path(args.data) / p) for p in man["out_path"]]
    starts = list(range(0, len(paths), 256))
    todo = [s for s in starts if not (parts / f"part_{s:05d}.npz").exists()]
    print(f"{len(paths)} photos, {len(starts) - len(todo)}/{len(starts)} parts done", flush=True)
    for s in todo:
        if args.time_budget and time.time() - t0 > args.time_budget:
            break
        idx = np.arange(s, min(s + 256, len(paths)))
        maps = np.zeros((len(idx), 2, 3, GRID, GRID), np.float32)
        fov = np.zeros((len(idx), GRID, GRID), np.float32)
        dev = np.zeros(len(idx), np.float32)
        err = np.zeros(len(idx), np.float32)
        for a in range(0, len(idx), 16):
            batch = idx[a:a + 16]
            loaded = [load_224(paths[i]) for i in batch]
            F = last_layer(torch, model, np.stack([x for _, x in loaded]))           # n x 2048 x 7 x 7
            pooled = F.mean((2, 3))
            for j, i in enumerate(batch):
                r = i - s
                fov[r] = fov_cells(loaded[j][0])
                dev[r] = np.abs(pooled[j] - X[i]).max()
                Fi = F[j].reshape(2048, -1).astype(np.float64)
                for h in range(2):
                    M = (W[hr[i, h]] @ Fi) / (GRID * GRID)                            # 3 x 49
                    maps[r, h] = M.reshape(3, GRID, GRID)
                    err[r] = max(err[r], np.abs(M.sum(1) - X[i] @ W[hr[i, h]].T).max())    # against the saved features
        tmp = parts / f"part_{s:05d}.tmp.npz"
        np.savez(tmp, maps=maps, fov=fov, dev=dev, err=err)
        os.replace(tmp, parts / f"part_{s:05d}.npz")
        print(f"  part {s:05d}: largest feature difference {dev.max():.2e} ({time.time() - t0:.0f} s)", flush=True)
    missing = [s for s in starts if not (parts / f"part_{s:05d}.npz").exists()]
    if missing:
        print(f"time budget reached: {len(missing)} parts left; run the same command again")
        sys.exit(3)
    d = [np.load(parts / f"part_{s:05d}.npz") for s in starts]
    np.savez_compressed(out / "maps.npz", uid=np.array(man["uid"].astype(str).tolist()), heads=hr,
                        maps=np.concatenate([x["maps"] for x in d]), fov=np.concatenate([x["fov"] for x in d]),
                        feature_dev=np.concatenate([x["dev"] for x in d]), logit_err=np.concatenate([x["err"] for x in d]))
    print(f"wrote {out / 'maps.npz'} ({time.time() - t0:.0f} s)")


# --------------------------------------------------------------------------------------------
# stage 3: deletion test (PyTorch, photos)
# --------------------------------------------------------------------------------------------

def deletion_sample(man, n, seed):
    urg = np.flatnonzero(((man["dataset"] == "aptos") & (man["ref3"] == 2)).to_numpy())
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(urg, size=min(n, len(urg)), replace=False)) if n else urg


def stage_deletion(args):
    from PIL import Image, ImageFilter
    torch, model = torch_setup(args)
    t0 = time.time()
    out = out_dir(args)
    parts = Path(args.parts) if args.parts else out / "parts_deletion"
    parts.mkdir(parents=True, exist_ok=True)
    man, _ = common(args)
    heads = np.load(out / "heads.npz")
    W, b = heads["W"], heads["b"]
    mp = np.load(out / "maps.npz")
    maps, hr, fov = mp["maps"], mp["heads"], mp["fov"]
    sample = deletion_sample(man, args.n_eyes, args.seed)
    chunks = [sample[a:a + DEL_CHUNK] for a in range(0, len(sample), DEL_CHUNK)]
    print(f"{len(sample)} urgent eyes in {len(chunks)} chunks", flush=True)
    for ci, chunk in enumerate(chunks):
        f = parts / f"chunk_{ci:03d}.csv"
        if f.exists():
            continue
        if args.time_budget and time.time() - t0 > args.time_budget:
            break
        rows = []
        for i in chunk:
            im = Image.open(Path(args.data) / man["out_path"].iloc[i]).convert("RGB")
            im = im.resize((SIZE, SIZE), Image.BICUBIC, reducing_gap=3.0)
            raw = np.asarray(im, np.uint8)
            blur = np.asarray(im.filter(ImageFilter.GaussianBlur(radius=8)), np.uint8)
            E = (maps[i, 0, 2] - maps[i, 0, 0]).ravel()                      # urgent evidence, plain model
            rng = np.random.default_rng(args.seed + int(i))
            orders = {"evidence": np.argsort(-E, kind="stable")}
            orders.update({f"random{r}": rng.permutation(GRID * GRID) for r in range(DEL_RANDOM)})
            on = np.flatnonzero(fov[i].ravel() >= 0.5)                       # cells mostly on the retina
            off = np.setdiff1d(np.arange(GRID * GRID), on)
            orders.update({f"retina{r}": np.concatenate([rng.permutation(on), rng.permutation(off)])
                           for r in range(DEL_RANDOM)})
            imgs, keys = [raw], [("none", 0)]
            for oname, order in orders.items():
                for k in DEL_STEPS[1:]:
                    a = raw.copy()
                    for cell in order[:k]:
                        r0, c0 = (cell // GRID) * CELL, (cell % GRID) * CELL
                        a[r0:r0 + CELL, c0:c0 + CELL] = blur[r0:r0 + CELL, c0:c0 + CELL]
                    imgs.append(a)
                    keys.append((oname, k))
            F = last_layer(torch, model, to_input(imgs)).mean((2, 3)).astype(np.float64)
            L = F @ W[hr[i, 0]].T + b[hr[i, 0]]
            P = softmax(L)
            for (oname, k), l, p in zip(keys, L, P):
                for o in (orders if oname == "none" else [oname]):
                    rows.append({"uid": man["uid"].iloc[i], "order": o, "cells": k, "urgent_logodds": l[2] - l[0],
                                 "p_urgent": p[2]})
        pd.DataFrame(rows).to_csv(f, index=False, float_format="%.5f")
        print(f"  chunk {ci}: {len(chunk)} eyes ({time.time() - t0:.0f} s)", flush=True)
    missing = [ci for ci in range(len(chunks)) if not (parts / f"chunk_{ci:03d}.csv").exists()]
    if missing:
        print(f"time budget reached: {len(missing)} chunks left; run the same command again")
        sys.exit(3)
    pd.concat([pd.read_csv(parts / f"chunk_{ci:03d}.csv") for ci in range(len(chunks))]).to_csv(
        out / "deletion.csv", index=False, float_format="%.5f")
    print(f"wrote {out / 'deletion.csv'} ({time.time() - t0:.0f} s)")


def flip_sample(man, n, seed):
    rng = np.random.default_rng(seed)
    idx = []
    for d in ("aptos", "idrid", "messidor2"):
        s = np.flatnonzero((man["dataset"] == d).to_numpy())
        idx.append(np.sort(rng.choice(s, size=min(n, len(s)), replace=False)))
    return np.concatenate(idx)


def stage_flip(args):
    torch, model = torch_setup(args)
    t0 = time.time()
    out = out_dir(args)
    man, X = common(args)
    heads = np.load(out / "heads.npz")
    W, b = heads["W"], heads["b"]
    hr = np.load(out / "maps.npz")["heads"]
    sample = flip_sample(man, args.n_flip, args.seed)
    res = {k: np.zeros((len(sample), 3, GRID, GRID), np.float32) for k in ("maps_lr", "maps_ud")}
    res.update({k: np.zeros((len(sample), 3)) for k in ("logits", "logits_lr", "logits_ud")})
    for a in range(0, len(sample), 16):
        batch = sample[a:a + 16]
        raws = [load_224(Path(args.data) / man["out_path"].iloc[i])[0] for i in batch]
        for key, flip in (("lr", lambda r: r[:, ::-1]), ("ud", lambda r: r[::-1, :])):
            F = last_layer(torch, model, to_input([flip(r) for r in raws])).astype(np.float64)
            for j, i in enumerate(batch):
                h = hr[i, 0]
                M = (W[h] @ F[j].reshape(2048, -1)) / (GRID * GRID)
                res[f"maps_{key}"][a + j] = M.reshape(3, GRID, GRID)
                res[f"logits_{key}"][a + j] = M.sum(1) + b[h]
        for j, i in enumerate(batch):
            res["logits"][a + j] = X[i] @ W[hr[i, 0]].T + b[hr[i, 0]]
        if a % 160 == 0:
            print(f"  {a + len(batch)}/{len(sample)} photos ({time.time() - t0:.0f} s)", flush=True)
    np.savez_compressed(out / "flip.npz", idx=sample, uid=np.array(man["uid"].astype(str).iloc[sample].tolist()), **res)
    print(f"wrote {out / 'flip.npz'} ({time.time() - t0:.0f} s)")


# --------------------------------------------------------------------------------------------
# shared by examples and summary: C1 decisions of both models
# --------------------------------------------------------------------------------------------

def c1_decisions(R, man, alpha):
    """Out-of-fold C1 decisions on APTOS for E1's model and the camera-decorrelated one."""
    from c1_referral import classwise_thresholds, decide_from_sets
    y = man["ref3"].to_numpy(int)
    row = pd.Series(np.arange(len(man)), index=man["uid"])
    dec = {m: np.full(len(man), -1) for m in MODELS}
    for m, folder in (("plain", R / "e1" / "lr"), ("dec", R / "camera_fixes" / "weights")):
        for k in range(10):
            f = pd.read_csv(folder / f"fold{k}.csv.gz")
            idx = row[f["uid"]].to_numpy()
            P = f[["p0", "p1", "p2"]].to_numpy()
            cal, test = (f["role"] == "cal").to_numpy(), (f["role"] == "test").to_numpy()
            q = classwise_thresholds(P[cal], y[idx[cal]], alpha)
            dec[m][idx[test]] = decide_from_sets((1 - P[test]) <= q[None, :])
    return dec


# --------------------------------------------------------------------------------------------
# stage 4: example overlays (photos; never Messidor-2)
# --------------------------------------------------------------------------------------------

def stage_examples(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from PIL import Image
    from c1_referral import HOME, ROUTINE, URGENT, UNCERTAIN, camera_groups
    out, R = out_dir(args), Path(args.results)
    man, _ = common(args)
    y = man["ref3"].to_numpy(int)
    g = camera_groups(man).to_numpy()
    heads = np.load(out / "heads.npz")
    b = heads["b"]
    mp = np.load(out / "maps.npz")
    maps, hr = mp["maps"], mp["heads"]
    dec = c1_decisions(R, man, args.alpha)
    apt = (man["dataset"] == "aptos").to_numpy()
    rng = np.random.default_rng(args.seed)
    word = {HOME: "no referral", ROUTINE: "routine", URGENT: "urgent", UNCERTAIN: "grader"}

    def pick(mask, n):
        idx = np.flatnonzero(mask)
        return list(rng.choice(idx, size=min(n, len(idx)), replace=False)) if len(idx) else []

    rows = []
    rows += [(i, "Urgent eye, referred urgently") for i in pick(apt & (y == 2) & (dec["plain"] == URGENT), 2)]
    rows += [(i, "Urgent eye, not referred urgently") for i in pick(apt & (y == 2) & (dec["plain"] != URGENT), 2)]
    rows += [(i, "H camera, referable eye: sent home by\nE1's model, referred after the camera fix")
             for i in pick(apt & (y >= 1) & (g == "H") & (dec["plain"] == HOME) & (dec["dec"] != HOME), 2)]
    rows += [(i, "IDRiD, urgent eye") for i in pick((man["dataset"] == "idrid").to_numpy() & (y == 2), 1)]
    assert not any(man["dataset"].iloc[i] == "messidor2" for i, _ in rows)      # its licence forbids showing them
    E, P = {}, {}
    for i, _ in rows:
        c = 2 if y[i] == 2 else 1
        for h in range(2):
            E[(i, h)] = maps[i, h, c] - maps[i, h, 0]
            P[(i, h)] = softmax((maps[i, h].sum((1, 2)) + b[hr[i, h]])[None])[0]
    vals = np.concatenate([np.delete(v.ravel(), GRID + 1) for v in E.values()])
    vmax = float(np.quantile(np.abs(vals), 0.99))           # the fixed top-left hotspot (summary section 4) saturates
    fig, axes = plt.subplots(len(rows), 3, figsize=(9.0, 3.25 * len(rows)), dpi=150)
    for r, (i, title) in enumerate(rows):
        im = np.asarray(Image.open(Path(args.data) / man["out_path"].iloc[i]).convert("RGB"))
        cls = "urgent" if y[i] == 2 else "routine"
        sub = f"grade {int(man['grade'].iloc[i])}" + (f", camera group {g[i]}" if isinstance(g[i], str) else "")
        axes[r, 0].imshow(im)
        axes[r, 0].set_title(f"{title}\n{sub}", fontsize=7.5, loc="left")
        for h, name in enumerate(MODELS):
            up = np.asarray(Image.fromarray(E[(i, h)].astype(np.float32)).resize(im.shape[:2][::-1], Image.BILINEAR))
            rgba = cm.RdBu_r(np.clip(0.5 + up / (2 * vmax), 0, 1))
            rgba[..., 3] = np.clip(np.abs(up) / vmax, 0, 1) * 0.8           # strong evidence opaque, none clear
            ax = axes[r, 1 + h]
            ax.imshow(im)
            ax.imshow(rgba)
            d = f", C1: {word[int(dec[name][i])]}" if apt[i] else ""
            ax.set_title(f"{MODEL_NAMES[name]}\nP({cls}) {P[(i, h)][2 if cls == 'urgent' else 1]:.2f}{d}",
                         fontsize=7.5, loc="left")
        for ax in axes[r]:
            ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(f"Evidence for {'/'.join(['urgent', 'routine'])} against no referral (red: towards referral, blue: "
                 "against). The blue spot near the top left of some maps is the network's fixed hotspot.",
                 fontsize=8.5, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out / "examples.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame([{"uid": man["uid"].iloc[i], "dataset": man["dataset"].iloc[i], "grade": int(man["grade"].iloc[i]),
                   "camera_group": g[i], "why": t.replace("\n", " ")} for i, t in rows]).to_csv(out / "examples.csv", index=False)
    print(f"wrote {out / 'examples.png'} (photos: keep it on your computer)")


# --------------------------------------------------------------------------------------------
# stage 5: summary
# --------------------------------------------------------------------------------------------

def region_split(maps_c, fov, bias):
    """Log-odds = bias + inside + edge + outside, per photo."""
    inside, outside = fov >= FOV_IN, fov <= FOV_OUT
    edge = ~inside & ~outside
    s = lambda m: (maps_c * m).sum((1, 2))      # noqa: E731
    return {"bias": bias, "inside": s(inside), "edge": s(edge), "outside": s(outside),
            "n_inside": inside.sum((1, 2)), "n_edge": edge.sum((1, 2)), "n_outside": outside.sum((1, 2))}


def stage_summary(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from c1_referral import camera_groups
    t0 = time.time()
    out, R = out_dir(args), Path(args.results)
    man, _ = common(args)
    y, grade = man["ref3"].to_numpy(int), man["grade"].to_numpy(int)
    ds = man["dataset"].to_numpy()
    g = camera_groups(man).fillna("").to_numpy()
    heads = np.load(out / "heads.npz")
    b = heads["b"]
    mp = np.load(out / "maps.npz")
    maps, fov, hr = mp["maps"], mp["fov"], mp["heads"]
    ck = pd.read_csv(out / "heads_check.csv")

    # ---- regions: decompose the routine and urgent log-odds of every photo ----
    rows = []
    for h, name in enumerate(MODELS):
        for c, cname in ((1, "routine"), (2, "urgent")):
            Mc = maps[:, h, c] - maps[:, h, 0]
            bias = b[hr[:, h], c] - b[hr[:, h], 0]
            sp = region_split(Mc, fov, bias)
            rows.append(pd.DataFrame({"uid": man["uid"], "dataset": ds, "grade": grade, "group": g, "model": name,
                                      "contrast": cname, **sp,
                                      "total": sp["bias"] + sp["inside"] + sp["edge"] + sp["outside"]}))
    reg = pd.concat(rows, ignore_index=True)
    reg.to_csv(out / "regions.csv.gz", index=False, float_format="%.4f")

    def cohort(dname, grp=None):
        s = (reg["dataset"] == dname)
        return s & (reg["group"] == grp) if grp else s

    # ---- deletion ----
    dl = pd.read_csv(out / "deletion.csv") if (out / "deletion.csv").exists() else None

    L = ["# Additional work: where in the photo does the model find its evidence?", "",
         "Logistic regression is linear in ResNet-50's average-pooled features, so each logit splits exactly into a "
         "7 × 7 map over the photo (class activation mapping): cell (i, j) contributes w · F(i, j) / 49, where F(i, j) "
         "is the last convolutional layer at that cell. APTOS photos are explained by the E1 fold model that never "
         "trained on them (and by the camera-decorrelated model of the same fold); IDRiD and Messidor-2 photos by the "
         "all-APTOS models of E2. The maps below are for log-odds against no referral (routine − no referral, urgent "
         "− no referral).", "",
         "## 1. Checks", "",
         f"- Linear heads rebuilt from the saved settings: largest difference from the saved probabilities "
         f"{ck['head_vs_saved'].max():.1e} (E1 folds {ck[ck['head'].str.startswith('plain_f')]['head_vs_saved'].max():.1e}, "
         f"decorrelated folds {ck[ck['head'].str.startswith('dec_f')]['head_vs_saved'].max():.1e}, all-APTOS models "
         f"{ck[ck['head'].str.endswith('_all')]['head_vs_saved'].max():.1e}).",
         f"- Features recomputed from the photos against the saved features: largest difference {mp['feature_dev'].max():.1e}.",
         f"- Maps against the logits: the 49 cells plus the bias reproduce every logit computed from the saved features "
         f"to within {mp['logit_err'].max():.1e}.",
         "", "## 2. Where the evidence lies", "",
         f"Cells are inside the retina (≥ {100 * FOV_IN:.0f} % of the cell on the retina), on its edge, or outside it "
         "(black). Mean log-odds of grade-2 (routine) eyes for routine against no referral, split by region:", "",
         "| Photos | Model | Eyes | Log-odds | Bias | Inside | Edge | Outside | Cells inside / edge / outside |",
         "|---|---|---|---|---|---|---|---|---|"]
    cohorts = [("APTOS, H cameras", "aptos", "H"), ("APTOS, M cameras", "aptos", "M"), ("APTOS, S cameras", "aptos", "S"),
               ("IDRiD", "idrid", None), ("Messidor-2", "messidor2", None)]
    table = []
    for label, dname, grp in cohorts:
        for name in MODELS:
            s = cohort(dname, grp) & (reg["model"] == name) & (reg["contrast"] == "routine") & (reg["grade"] == 2)
            r = reg[s]
            table.append({"photos": label, "model": name, "eyes": len(r), **{c: r[c].mean() for c in
                          ("total", "bias", "inside", "edge", "outside", "n_inside", "n_edge", "n_outside")}})
            L.append(f"| {label} | {MODEL_NAMES[name]} | {len(r)} | {r['total'].mean():+.2f} | {r['bias'].mean():+.2f} | "
                     f"{r['inside'].mean():+.2f} | {r['edge'].mean():+.2f} | {r['outside'].mean():+.2f} | "
                     f"{r['n_inside'].mean():.0f} / {r['n_edge'].mean():.0f} / {r['n_outside'].mean():.0f} |")
    T = pd.DataFrame(table).set_index(["photos", "model"])
    T.to_csv(out / "regions_grade2.csv", float_format="%.4f")
    gap = {name: T.loc[("APTOS, M cameras", name)] - T.loc[("APTOS, H cameras", name)] for name in MODELS}
    L += ["", "The gap between grade-2 eyes on M and on H cameras (M minus H), by region:", "",
          "| Model | Log-odds gap | From inside cells | From edge cells | From outside cells | From the bias |",
          "|---|---|---|---|---|---|"]
    for name in MODELS:
        d = gap[name]
        L.append(f"| {MODEL_NAMES[name]} | {d['total']:+.2f} | {d['inside']:+.2f} | {d['edge']:+.2f} | "
                 f"{d['outside']:+.2f} | {d['bias']:+.2f} |")
    p = gap["plain"]
    share_in = p["inside"] / p["total"] if p["total"] else np.nan
    L += ["", f"For E1's model, {100 * share_in:.0f} % of the gap comes from cells inside the retina and "
          f"{100 * (p['edge'] + p['outside']) / p['total']:.0f} % from its edge and the black corners."
          if p["total"] else "", ""]

    # healthy eyes: does the camera move the log-odds of grade-0 eyes too?
    L += ["Mean log-odds of grade-0 eyes (routine against no referral), the same split:", "",
          "| Photos | Model | Eyes | Log-odds | Inside | Edge | Outside |", "|---|---|---|---|---|---|---|"]
    for label, dname, grp in cohorts:
        for name in MODELS:
            s = cohort(dname, grp) & (reg["model"] == name) & (reg["contrast"] == "routine") & (reg["grade"] == 0)
            r = reg[s]
            if len(r):
                L.append(f"| {label} | {MODEL_NAMES[name]} | {len(r)} | {r['total'].mean():+.2f} | "
                         f"{r['inside'].mean():+.2f} | {r['edge'].mean():+.2f} | {r['outside'].mean():+.2f} |")

    # ---- deletion ----
    dt = None
    if dl is not None:
        piv = dl.pivot_table(index=["uid", "cells"], columns="order", values="urgent_logodds").reset_index()
        ctrl = {"random": [c for c in piv.columns if str(c).startswith("random")],
                "retina": [c for c in piv.columns if str(c).startswith("retina")]}
        ctrl = {k: v for k, v in ctrl.items() if v}
        for k, cols in ctrl.items():
            piv[k] = piv[cols].mean(axis=1)
        base = piv[piv["cells"] == 0].set_index("uid")["evidence"]
        for k in ["evidence", *ctrl]:
            piv[f"drop_{k}"] = base.reindex(piv["uid"]).to_numpy() - piv[k]
        boot = np.random.default_rng(args.seed)
        uids = piv["uid"].unique()
        dt = []
        for k in DEL_STEPS:
            sk = piv[piv["cells"] == k].set_index("uid").loc[uids]
            r = {"cells": k, "drop_evidence": sk["drop_evidence"].mean(),
                 "p_urgent_evidence": dl[(dl["cells"] == k) & (dl["order"] == "evidence")].set_index("uid").loc[uids, "p_urgent"].mean()}
            for c in ctrl:
                diff = (sk["drop_evidence"] - sk[f"drop_{c}"]).to_numpy()
                bs_ = [diff[boot.integers(0, len(diff), len(diff))].mean() for _ in range(1000)]
                r.update({f"drop_{c}": sk[f"drop_{c}"].mean(), f"diff_{c}": diff.mean(),
                          f"lo_{c}": np.quantile(bs_, 0.025), f"hi_{c}": np.quantile(bs_, 0.975),
                          f"p_urgent_{c}": dl[(dl["cells"] == k) & dl["order"].str.startswith(c)].groupby("uid")["p_urgent"].mean().loc[uids].mean()})
            dt.append(r)
        dt = pd.DataFrame(dt)
        dt.to_csv(out / "deletion_summary.csv", index=False, float_format="%.4f")
        main_ctrl = "retina" if "retina" in ctrl else "random"
        L += ["", "## 3. Do the maps point at what matters? (deletion test)", "",
              f"{len(uids)} APTOS urgent eyes (random sample). The cells with the most urgent evidence are replaced by a "
              "blurred copy of the photo (Gaussian, radius 8 px at 224 px: lesions disappear, colour and brightness "
              f"stay), and the urgent log-odds recomputed. Controls ({DEL_RANDOM} random orders per eye each): random "
              "cells anywhere, and random cells on the retina first (blurring a black corner changes nothing). Mean over "
              "eyes; 95 % bootstrap CI of the difference from the evidence order.", "",
              "| Cells blurred (of 49) | Drop in urgent log-odds: evidence order | Random cells | Random retina cells | "
              "Evidence minus retina control (95 % CI) | Mean P(urgent): evidence / random / retina |", "|---|---|---|---|---|---|"]
        for _, r in dt.iterrows():
            L.append(f"| {int(r['cells'])} | {r['drop_evidence']:.2f} | {r.get('drop_random', np.nan):.2f} | "
                     f"{r.get('drop_retina', np.nan):.2f} | {r[f'diff_{main_ctrl}']:.2f} ({r[f'lo_{main_ctrl}']:.2f} to "
                     f"{r[f'hi_{main_ctrl}']:.2f}) | {r['p_urgent_evidence']:.2f} / {r.get('p_urgent_random', np.nan):.2f} / "
                     f"{r.get('p_urgent_retina', np.nan):.2f} |")

    # ---- flip test ----
    fl = np.load(out / "flip.npz") if (out / "flip.npz").exists() else None
    if fl is not None:
        idx = fl["idx"]
        orig = maps[idx, 0]                                          # frame coordinates, as the network sees them
        E = {"original": orig[:, 1] - orig[:, 0], "mirrored left-right": fl["maps_lr"][:, 1] - fl["maps_lr"][:, 0],
             "mirrored top-bottom": fl["maps_ud"][:, 1] - fl["maps_ud"][:, 0]}
        hot = (1, 1)
        Eall = maps[:, 0, 1] - maps[:, 0, 0]
        hot_sd, tot_sd = float(Eall[:, 1, 1].std()), float(Eall.sum((1, 2)).std())
        L += ["", "## 4. A fixed hotspot, and mirrored eyes", "",
              "One cell, in the second row and second column of the 7 × 7 grid (top left, just inside the retina), "
              "carries strong evidence against referral in every dataset, while the cells in the mirror positions do "
              f"not. To see whether it belongs to the eyes or to the network, {len(idx)} photos ({len(idx) // 3} per "
              "dataset) were mirrored and explained again. Mean evidence for routine against no referral, in the cell at "
              "each position of the frame:", "",
              "| Photos | Top left (row 2, col 2) | Top right (row 2, col 6) | Bottom left (row 6, col 2) | "
              "Bottom right (row 6, col 6) |", "|---|---|---|---|---|"]
        for k, v in E.items():
            L.append(f"| {k} | {v[:, 1, 1].mean():+.2f} | {v[:, 1, 5].mean():+.2f} | {v[:, 5, 1].mean():+.2f} | "
                     f"{v[:, 5, 5].mean():+.2f} |")
        stays = abs(E["mirrored left-right"][:, 1, 1].mean() - E["original"][:, 1, 1].mean()) < \
            abs(E["mirrored left-right"][:, 1, 5].mean() - E["original"][:, 1, 1].mean())
        # per photo: correlate the original map with the mirrored photo's map flipped back (the evidence follows the
        # eye) or as it is (it stays in the frame), inside the retina and on its edge; hotspot positions left out
        fv = fov[idx]
        corr = {}
        for key, back in (("mirrored left-right", lambda m: m[..., ::-1]), ("mirrored top-bottom", lambda m: m[..., ::-1, :])):
            for region, sel in (("inside the retina", lambda f: f >= FOV_IN), ("edge of the retina", lambda f: (f > FOV_OUT) & (f < FOV_IN))):
                fo, fr = [], []
                for o, m, f in zip(E["original"], E[key], fv):
                    keep = sel(f) & sel(back(f))
                    keep[1, 1] = keep[1, 5] = keep[5, 1] = keep[5, 5] = False
                    if keep.sum() >= 5:
                        fo.append(np.corrcoef(o[keep], back(m)[keep])[0, 1])
                        fr.append(np.corrcoef(o[keep], m[keep])[0, 1])
                corr[(key, region)] = (np.nanmedian(fo), np.nanmedian(fr))
        L += ["", ("The hotspot stays in the top-left cell of the frame when the eye is mirrored: it belongs to the "
                   "network, not to the eye. A plausible cause is the asymmetric zero padding of ResNet-50's strided "
                   "convolutions, which treats the top and left borders differently from the bottom and right ones "
                   "(Alsallakh et al., ICLR 2021); here the dark corner of the photo meets that border. It adds about "
                   f"the same amount to every photo (SD {hot_sd:.2f} over all {len(maps)} photos, against {tot_sd:.2f} for "
                   "the whole log-odds), so it acts mostly like a shift of the bias.")
                  if stays else "The hotspot moves with the eye when the photo is mirrored: it belongs to the eye.", "",
              "Does the rest of each map follow the eye? Median correlation, over the photos, between a photo's map and "
              "the map of its mirror image, flipped back (the evidence follows the eye) or as it is (it stays in the "
              "frame); hotspot positions left out:", "",
              "| Mirror | Cells | Flipped back (follows the eye) | As it is (stays in the frame) |", "|---|---|---|---|"]
        for (key, region), (fo, fr) in corr.items():
            L.append(f"| {key.replace('mirrored ', '')} | {region} | {fo:.2f} | {fr:.2f} |")
        lr_in, lr_edge = corr[("mirrored left-right", "inside the retina")], corr[("mirrored left-right", "edge of the retina")]
        L += ["", "The maps mix two parts. Inside the retina, under the left-right mirror (which moves the optic disc to "
              f"the other side), the evidence follows the eye ({lr_in[0]:.2f} against {lr_in[1]:.2f}), as evidence about "
              f"the retina should. On the edge of the retina it stays in the frame ({lr_edge[1]:.2f} against "
              f"{lr_edge[0]:.2f}): the outline of the retina, which mirroring does not change, drives it. Under the "
              "top-bottom mirror the frame-fixed part is larger even inside the retina. So a single photo's map should be "
              "read with care; the averages over many eyes (section 2) and the deletion test (section 3) are the "
              "reliable evidence.", ""]
        # predictions of mirrored eyes
        P0, Plr, Pud = softmax(fl["logits"]), softmax(fl["logits_lr"]), softmax(fl["logits_ud"])
        dsx = man["dataset"].to_numpy()[idx]
        L += ["How much do the predictions change when an eye is mirrored (left and right eyes are mirror images)?", "",
              "| Photos | Mirror | Mean change in P(referable) | Largest change | Most likely class changes |",
              "|---|---|---|---|---|"]
        for d, dn in (("aptos", "APTOS"), ("idrid", "IDRiD"), ("messidor2", "Messidor-2")):
            sel = dsx == d
            for name, Pm in (("left-right", Plr), ("top-bottom", Pud)):
                dref = np.abs(Pm[sel, 1:].sum(1) - P0[sel, 1:].sum(1))
                L.append(f"| {dn} ({sel.sum()}) | {name} | {dref.mean():.3f} | {dref.max():.2f} | "
                         f"{100 * np.mean(Pm[sel].argmax(1) != P0[sel].argmax(1)):.1f} % |")
        # C1 decisions of the mirrored APTOS photos, with each fold's thresholds
        from c1_referral import classwise_thresholds, decide_from_sets
        W_, b_ = heads["W"], heads["b"]
        Xf = np.load(Path(args.data) / "features" / "resnet50_224.npz")["X"]
        row = pd.Series(np.arange(len(man)), index=man["uid"])
        sel = np.flatnonzero(dsx == "aptos")
        changed = {"left-right": 0, "top-bottom": 0}
        for k in range(10):
            f = pd.read_csv(R / "e1" / "lr" / f"fold{k}.csv.gz")
            cal = row[f.loc[f["role"] == "cal", "uid"]].to_numpy()
            h = list(heads["names"]).index(f"plain_f{k}")
            q = classwise_thresholds(softmax(Xf[cal] @ W_[h].T + b_[h]), y[cal], args.alpha)
            mine = sel[heads["fold_of"][idx[sel]] == k]
            if not len(mine):
                continue
            d0 = decide_from_sets((1 - P0[mine]) <= q[None, :])
            for name, Pm in (("left-right", Plr), ("top-bottom", Pud)):
                changed[name] += int(np.sum(decide_from_sets((1 - Pm[mine]) <= q[None, :]) != d0))
        L += ["", f"C1's decision changes for {changed['left-right']} of the {len(sel)} APTOS eyes when they are mirrored "
              f"left-right, and for {changed['top-bottom']} when mirrored top-bottom.", ""]
        pd.DataFrame({"uid": fl["uid"], "dataset": dsx, "p_ref": P0[:, 1:].sum(1), "p_ref_lr": Plr[:, 1:].sum(1),
                      "p_ref_ud": Pud[:, 1:].sum(1)}).to_csv(out / "flip.csv", index=False, float_format="%.4f")
        plot_flip(E, out / "flip.png")

    L += ["", f"Made {time.strftime('%Y-%m-%d %H:%M')} by evidence_maps.py ({time.time() - t0:.0f} s).", ""]
    (out / "summary.md").write_text("\n".join(L))

    # ---- figures ----
    plot_mean_maps(maps, fov, man, g, out / "evidence_by_camera.png")
    if dt is not None:
        plot_deletion(dt, out / "deletion.png")
    print(f"wrote {out / 'summary.md'} ({time.time() - t0:.0f} s)")


def plot_mean_maps(maps, fov, man, g, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds, grade = man["dataset"].to_numpy(), man["grade"].to_numpy(int)
    cols = [("APTOS, H cameras", (ds == "aptos") & (g == "H")), ("APTOS, M cameras", (ds == "aptos") & (g == "M")),
            ("APTOS, S cameras", (ds == "aptos") & (g == "S")), ("IDRiD", ds == "idrid"), ("Messidor-2", ds == "messidor2")]
    M = {}
    for h in range(2):
        for j, (label, sel) in enumerate(cols):
            s = sel & (grade == 2)
            M[(h, j)] = (maps[s, h, 1] - maps[s, h, 0]).mean(0)
            M[("fov", j)] = fov[s].mean(0)
    vals = np.concatenate([np.delete(v.ravel(), GRID + 1) for k, v in M.items() if k[0] != "fov"])
    vmax = float(np.abs(vals).max())                     # the top-left hotspot (section 4) is clipped
    fig, axes = plt.subplots(2, len(cols), figsize=(2.3 * len(cols), 5.3), dpi=200)
    for h, name in enumerate(MODELS):
        for j, (label, sel) in enumerate(cols):
            ax = axes[h, j]
            im = ax.imshow(M[(h, j)], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.contour(M[("fov", j)], levels=[0.5], colors="#0b0b0b", linewidths=0.8)
            n = int(np.sum(sel & (grade == 2)))
            ax.set_title((f"{label}\n{n} eyes; " if h == 0 else "") + f"sum {M[(h, j)].sum():+.2f}",
                         fontsize=7.2, loc="left")
            ax.set_xticks([]), ax.set_yticks([])
        axes[h, 0].set_ylabel(MODEL_NAMES[name], fontsize=8)
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
    cb.set_label("evidence for routine vs no referral (log-odds per cell)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle("Mean evidence maps of grade-2 eyes: routine vs no referral (black line: edge of the retina;\n"
                 "the top-left hotspot is clipped, see section 4)", fontsize=9, x=0.01, ha="left")
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_deletion(dt, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=200)
    ax.plot(dt["cells"], dt["drop_evidence"], "-o", color="#d03b3b", ms=3.5, label="most urgent evidence first")
    if "drop_random" in dt:
        ax.plot(dt["cells"], dt["drop_random"], "-o", color="#a8a7a1", ms=3.5, label="random cells (control)")
    if "drop_retina" in dt:
        ax.plot(dt["cells"], dt["drop_retina"], "-o", color="#2a78d6", ms=3.5, label="random retina cells (control)")
    ax.set_xlabel("cells blurred (of 49)", fontsize=8)
    ax.set_ylabel("drop in urgent log-odds", fontsize=8)
    ax.set_title("Deletion test, APTOS urgent eyes", fontsize=9.5, loc="left")
    ax.grid(color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7.5)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_flip(E, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vmax = max(np.abs(v.mean(0)).max() for v in E.values())
    fig, axes = plt.subplots(1, len(E), figsize=(2.6 * len(E) + 1, 3.0), dpi=200)
    for ax, (k, v) in zip(axes, E.items()):
        im = ax.imshow(v.mean(0), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(k, fontsize=8, loc="left")
        ax.set_xticks([]), ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle("Mean evidence (routine vs no referral) in the frame, original and mirrored photos", fontsize=9,
                 x=0.01, ha="left")
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Additional work: exact evidence maps of the logistic regression.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("stage", choices=["heads", "maps", "deletion", "flip", "examples", "summary"])
    ap.add_argument("--data", default="../data_512")
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default="../results/evidence_maps")
    ap.add_argument("--weights", default="../models/resnet50_ram-a26f946b.pth")
    ap.add_argument("--parts", default=None, help="folder for the resumable parts (default: inside --out)")
    ap.add_argument("--n-eyes", type=int, default=150, help="deletion test: urgent eyes (0 = all)")
    ap.add_argument("--n-flip", type=int, default=200, help="flip test: photos per dataset")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--time-budget", type=float, default=0, help="maps, deletion: stop after this many seconds")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    {"heads": stage_heads, "maps": stage_maps, "deletion": stage_deletion, "flip": stage_flip,
     "examples": stage_examples, "summary": stage_summary}[args.stage](args)


if __name__ == "__main__":
    main()
