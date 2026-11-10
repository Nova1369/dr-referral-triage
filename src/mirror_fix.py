#!/usr/bin/env python3
"""
mirror_fix.py - additional work: mirror-averaged features.

Left and right eyes are mirror images of each other, but ResNet-50's features are not mirror-invariant: mirroring a
photo changes C1's decision for about one APTOS eye in eight (evidence_maps.py, flip test). Averaging the features of a
photo and of its mirror image makes any model on them exactly mirror-invariant, at no training cost.

Compared, in E1's 10 outer folds (each fold's chosen setting of E1's logistic regression, refitted; C1 fitted on the
fold's calibration photos):
  plain      the features of the photo, as in E1
  averaged   (features of the photo + features of its mirror image) / 2
For both: AUROCs, C1's targets, and how often C1's decision changes when the test photo is mirrored (the averaged
features of a photo and of its mirror image are the same, so for them it cannot change).
New clinics: logistic regression (C = 0.003, no PCA, as in E2) trained on all of APTOS; AUROC, and C2 refitted on k
labelled local photos at IDRiD (k = 50, 100) and Messidor-2 (k = 200, 400), 200 draws as in E2.

Needs features/resnet50_224_hflip.npz:
  python extract_variants.py --data ../data_512 --weights ../models/resnet50_ram-a26f946b.pth --variant hflip

Outputs (in --out, default ../results/mirror): summary.md, metrics.csv, c2.csv

Usage:
  python mirror_fix.py --data ../data_512 --results ../results
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ablations import C_FIXED, K_DRAWS, auroc, fit_lr, load_features, proba, rule_metrics, same_camera_auroc
from conformal import HOME, classwise_thresholds, decide_from_sets
from e2_new_clinics import Pool

NAMES = {"plain": "Features of the photo (E1)", "averaged": "Mirror-averaged features"}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Additional work: mirror-averaged features.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", default="../data_512")
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default="../results/mirror")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    t0 = time.time()
    data, R, out = Path(args.data), Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    a = args.alpha

    man = pd.read_csv(data / "manifest.csv")
    y = man["ref3"].to_numpy(int)
    camera = (man["orig_w"].astype(str) + "x" + man["orig_h"].astype(str)).to_numpy()
    row = pd.Series(np.arange(len(man)), index=man["uid"])
    apt = np.flatnonzero((man["dataset"] == "aptos").to_numpy())
    X = load_features(data, "resnet50_224.npz", man)
    Xm = load_features(data, "resnet50_224_hflip.npz", man)
    if Xm is None or np.isnan(Xm).any():
        raise SystemExit("features/resnet50_224_hflip.npz is missing or incomplete; run extract_variants.py --variant hflip")
    feats = {"plain": X, "averaged": (X + Xm) / 2}
    best = pd.read_csv(R / "e1" / "best_params.csv")
    best = best[best["model"] == "lr"].set_index("fold")["best"].map(json.loads)
    folds = {}
    for k in range(10):
        f = pd.read_csv(R / "e1" / "lr" / f"fold{k}.csv.gz")
        folds[k] = {r: row[f.loc[f["role"] == r, "uid"]].to_numpy() for r in ("test", "cal")}
        folds[k]["saved"] = f.set_index(row[f["uid"]].to_numpy())[["p0", "p1", "p2"]]

    # ---- APTOS: E1's folds and settings ----
    rows, max_dev = [], 0.0
    for name, F in feats.items():
        P_oof, dec, dec_m = np.zeros((len(man), 3)), np.full(len(man), -1), np.full(len(man), -1)
        for k, f in folds.items():
            fit = np.setdiff1d(apt, np.concatenate([f["test"], f["cal"]]))
            n_pca, C = best[k]["pca"], best[k]["C"]
            sc = StandardScaler().fit(F[fit])
            pca = PCA(n_components=min(2048, len(fit), F.shape[1]), svd_solver="full").fit(sc.transform(F[fit]))
            tr = lambda Z: pca.transform(sc.transform(Z))[:, :n_pca]       # noqa: E731
            m = fit_lr(tr(F[fit]), y[fit], C)
            Pc, Pt = proba(m, tr(F[f["cal"]])), proba(m, tr(F[f["test"]]))
            if name == "plain":
                max_dev = max(max_dev, float(np.abs(Pt - f["saved"].loc[f["test"]].to_numpy()).max()))
            q = classwise_thresholds(Pc, y[f["cal"]], a)
            P_oof[f["test"]] = Pt
            dec[f["test"]] = decide_from_sets((1 - Pt) <= q[None, :])
            # the test photo mirrored: the plain model sees the mirror image's features
            Fm = Xm if name == "plain" else F
            Ptm = proba(m, tr(Fm[f["test"]]))
            dec_m[f["test"]] = decide_from_sets((1 - Ptm) <= q[None, :])
        ya, Pa, da = y[apt], P_oof[apt], dec[apt]
        rows.append({"features": name, "auroc_ref": auroc(ya >= 1, Pa[:, 1:].sum(1)), "auroc_urg": auroc(ya == 2, Pa[:, 2]),
                     "same_camera_auroc_ref": same_camera_auroc(ya >= 1, Pa[:, 1:].sum(1), camera[apt]),
                     **{f"c1_{k_}": v for k_, v in rule_metrics(da, ya).items()},
                     "mirror_changes": float(np.mean(dec_m[apt] != da)),
                     "mirror_changes_ref": float(np.mean(dec_m[apt][ya >= 1] != da[ya >= 1])),
                     # referable eyes sent home in at least one of the two orientations
                     "mirror_ref_home": float(np.mean(((dec_m[apt] == HOME) | (da == HOME))[ya >= 1]))})
        print(f"{name}: APTOS done ({time.time() - t0:.0f} s)", flush=True)
    met = pd.DataFrame(rows)

    # ---- new clinics: all-APTOS model, then C2 ----
    group = pd.factorize(man["dataset"] + "/" + man["group_id"].astype(str))[0]
    rng = np.random.default_rng(args.seed)
    pools = {ds: Pool(np.flatnonzero((man["dataset"] == ds).to_numpy()),
                      np.flatnonzero((man["dataset"] == ds).to_numpy()), group) for ds in K_DRAWS}
    draws = {(ds, k, b): pools[ds].draw(k, rng) for ds in K_DRAWS for k in K_DRAWS[ds] for b in range(args.draws)}
    c2_rows = []
    for name, F in feats.items():
        sc = StandardScaler().fit(F[apt])
        m = fit_lr(sc.transform(F[apt]), y[apt], C_FIXED)
        P = proba(m, sc.transform(F))
        for ds in K_DRAWS:
            s = np.flatnonzero((man["dataset"] == ds).to_numpy())
            met.loc[met["features"] == name, f"{ds}_auroc_ref"] = auroc(y[s] >= 1, P[s, 1:].sum(1))
            met.loc[met["features"] == name, f"{ds}_auroc_urg"] = auroc(y[s] == 2, P[s, 2])
        for (ds, k, b), (cal, test) in draws.items():
            q = classwise_thresholds(P[cal], y[cal], a)
            d = decide_from_sets((1 - P[test]) <= q[None, :])
            c2_rows.append({"features": name, "dataset": ds, "k": k, "draw": b, **rule_metrics(d, y[test])})
    c2 = pd.DataFrame(c2_rows).groupby(["features", "dataset", "k"]).mean(numeric_only=True).drop(columns="draw").reset_index()
    met.to_csv(out / "metrics.csv", index=False, float_format="%.5f")
    c2.to_csv(out / "c2.csv", index=False, float_format="%.5f")
    write_summary(out, args, met, c2, max_dev, time.time() - t0)
    print(f"wrote {out / 'summary.md'} ({time.time() - t0:.0f} s)")


def write_summary(out, args, met, c2, max_dev, elapsed):
    M = met.set_index("features")
    L = ["# Additional work: mirror-averaged features", "",
         "Left and right eyes are mirror images, but ResNet-50's features are not mirror-invariant. Averaging the "
         "features of a photo and of its mirror image makes the model give a photo and its mirror image the same answer. "
         "APTOS: E1's outer folds, each fold's chosen setting of E1's logistic regression refitted (on the photo's own "
         f"features it reproduces E1's saved predictions to within {max_dev:.0e}), C1 at α = {args.alpha:g} on the fold's "
         "calibration photos, pooled over the 10 held-out folds. In %.", "",
         "## 1. APTOS", "",
         "| Features | AUROC referable | Same-camera AUROC referable | AUROC urgent | C1: referable sent home | "
         "C1: urgent referred urgently | C1: healthy sent home | Referred per 1,000 | C1's decision changes when the "
         "photo is mirrored | ... for referable eyes |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("plain", "averaged"):
        r = M.loc[name]
        L.append(f"| {NAMES[name]} | {r['auroc_ref']:.3f} | {r['same_camera_auroc_ref']:.3f} | {r['auroc_urg']:.3f} | "
                 f"{100 * r['c1_ref_home']:.1f} | {100 * r['c1_urgent_urgent']:.1f} | {100 * r['c1_healthy_home']:.1f} | "
                 f"{1000 * r['c1_referred']:.0f} | {100 * r['mirror_changes']:.1f} | {100 * r['mirror_changes_ref']:.1f} |")
    p = M.loc["plain"]
    L += ["", f"With the photo's own features, {100 * p['mirror_ref_home']:.1f} % of referable eyes are sent home in at "
          f"least one of the two orientations, against {100 * p['c1_ref_home']:.1f} % in the original one.", "",
          "## 2. New clinics", "",
          "All-APTOS logistic regression (C = 0.003, no PCA). C2: thresholds refitted on k labelled local photos, "
          f"{args.draws} draws by whole groups, as in E2. In %.", "",
          "| Features | IDRiD AUROC referable | Messidor-2 AUROC referable | Messidor-2 AUROC urgent |", "|---|---|---|---|"]
    for name in ("plain", "averaged"):
        r = M.loc[name]
        L.append(f"| {NAMES[name]} | {r['idrid_auroc_ref']:.3f} | {r['messidor2_auroc_ref']:.3f} | "
                 f"{r['messidor2_auroc_urg']:.3f} |")
    L += ["", "| Features | Clinic | k | Referable sent home | Urgent referred urgently | Healthy sent home | Uncertain |",
          "|---|---|---|---|---|---|---|"]
    order = {"plain": 0, "averaged": 1, "idrid": 0, "messidor2": 1}
    c2 = c2.sort_values(["dataset", "k", "features"], key=lambda c: c.map(order) if c.name != "k" else c)
    for _, r in c2.iterrows():
        L.append(f"| {NAMES[r['features']]} | {'IDRiD' if r['dataset'] == 'idrid' else 'Messidor-2'} | {int(r['k'])} | "
                 f"{100 * r['ref_home']:.1f} | {100 * r['urgent_urgent']:.1f} | {100 * r['healthy_home']:.1f} | "
                 f"{100 * r['uncertain']:.1f} |")
    L += ["", f"Made {time.strftime('%Y-%m-%d %H:%M')} by mirror_fix.py ({elapsed:.0f} s).", ""]
    (out / "summary.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
