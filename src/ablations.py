#!/usr/bin/env python3
"""
ablations.py - the planned ablations and robustness tests, for logistic regression and the C1/C2 rules.

1. Image ablations. Logistic regression refitted in every E1 outer fold with E1's most frequent setting
   (standardised features, no PCA reduction, C = 0.003), on three feature sets:
     224 px (as in E1) | 512 px (small lesions kept) | Graham-filtered photos at 224 px
   APTOS: AUROCs and C1 on each fold's calibration photos. New clinics: the same setting trained on all of APTOS,
   then C2 refitted on k labelled local photos (200 draws, by whole groups, as in E2).
2. PCA size: from E1's own tuning logs (inner-fold scores of every setting it tried).
3. Robustness. E1's logistic regression, refitted in every fold with the setting E1 chose there (its predictions
   on the original photos reproduce E1's saved ones), applied to altered test photos:
     blur (radius 3 px at 512 px) | dark (half brightness) | JPEG quality 20
   - with C1's thresholds from the original calibration photos: the photos got worse after calibration;
   - with C1 recalibrated on equally altered calibration photos: a clinic that calibrates on its own photos.

Needs the feature files of extract_features.py (--size 512) and extract_variants.py in data_512/features.

Outputs (in --out, default ../results/ablations): summary.md, image_ablation.csv, robustness.csv, pca_size.csv,
ablations.png

Usage:
  python ablations.py --data ../data_512 --results ../results
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from conformal import HOME, ROUTINE, URGENT, UNCERTAIN, classwise_thresholds, decide_from_sets
from c1_referral import camera_groups
from e2_new_clinics import Pool

FEATURES = {"224": "resnet50_224.npz", "512": "resnet50_512.npz", "graham": "resnet50_224_graham.npz"}
FEATURE_NAMES = {"224": "224 px (E1)", "512": "512 px", "graham": "Graham filter, 224 px"}
VARIANTS = {"blur": "resnet50_224_blur.npz", "dark": "resnet50_224_dark.npz", "jpeg": "resnet50_224_jpeg.npz"}
VARIANT_NAMES = {"none": "original photos", "blur": "blurred (radius 3 px)", "dark": "half brightness",
                 "jpeg": "JPEG quality 20"}
C_FIXED = 0.003
K_DRAWS = {"idrid": [50, 100], "messidor2": [200, 400]}


def fit_lr(Z, y, C):
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LogisticRegression(C=C, max_iter=5000).fit(Z, y)


def proba(m, Z):
    P = np.zeros((len(Z), 3))
    P[:, m.classes_] = m.predict_proba(Z)
    return P


def auroc(pos, score):
    from scipy.stats import rankdata
    n1 = pos.sum()
    n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def same_camera_auroc(pos, score, camera):
    num = den = 0.0
    for c in np.unique(camera):
        s = camera == c
        n1, n0 = int(pos[s].sum()), int((~pos[s]).sum())
        if n1 and n0:
            num += auroc(pos[s], score[s]) * n1 * n0
            den += n1 * n0
    return num / den


def rule_metrics(d, y):
    return {"ref_home": np.mean(d[y >= 1] == HOME), "urgent_urgent": np.mean(d[y == 2] == URGENT),
            "healthy_home": np.mean(d[y == 0] == HOME), "uncertain": np.mean(d == UNCERTAIN),
            "referred": np.mean((d == ROUTINE) | (d == URGENT))}


def load_features(data, name, man):
    path = data / "features" / name
    if not path.exists():
        return None
    with np.load(path) as z:
        X, uid = z["X"], z["uid"].astype(str)
    out = np.full((len(man), X.shape[1]), np.nan, np.float32)
    pos = pd.Index(man["uid"].astype(str)).get_indexer(uid)
    assert (pos >= 0).all(), f"{name}: unknown photos"
    out[pos] = X
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ablations and robustness tests.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", default="../data_512")
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default="../results/ablations")
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
    folds = {}
    for k in range(10):
        f = pd.read_csv(R / "e1" / "lr" / f"fold{k}.csv.gz")
        folds[k] = {r: row[f.loc[f["role"] == r, "uid"]].to_numpy() for r in ("test", "cal", "external")}
        folds[k]["p_saved"] = f.set_index(row[f["uid"]].to_numpy())[["p0", "p1", "p2"]]
    best = pd.read_csv(R / "e1" / "best_params.csv")
    best = best[best["model"] == "lr"].set_index("fold")["best"].map(json.loads)

    # ---------------- 1. image ablations ----------------
    img_rows, c2_rows = [], []
    rng = np.random.default_rng(args.seed)
    group = pd.factorize(man["dataset"] + "/" + man["group_id"].astype(str))[0]
    pools = {ds: Pool(np.flatnonzero((man["dataset"] == ds).to_numpy()),
                      np.flatnonzero((man["dataset"] == ds).to_numpy()), group) for ds in K_DRAWS}
    draws = {(ds, k, b): pools[ds].draw(k, rng) for ds in K_DRAWS for k in K_DRAWS[ds] for b in range(args.draws)}
    for key, fname in FEATURES.items():
        X = load_features(data, fname, man)
        if X is None:
            print(f"{fname} not found; skipped")
            continue
        P_oof, dec = np.zeros((len(man), 3)), np.full(len(man), -1)
        ext = []
        for k, f in folds.items():
            fit = np.setdiff1d(apt, np.concatenate([f["test"], f["cal"]]))
            sc = StandardScaler().fit(X[fit])
            m = fit_lr(sc.transform(X[fit]), y[fit], C_FIXED)
            Pc, Pt = proba(m, sc.transform(X[f["cal"]])), proba(m, sc.transform(X[f["test"]]))
            q = classwise_thresholds(Pc, y[f["cal"]], a)
            P_oof[f["test"]] = Pt
            dec[f["test"]] = decide_from_sets((1 - Pt) <= q[None, :])
            for ds in ("idrid", "messidor2"):
                s = np.flatnonzero((man["dataset"] == ds).to_numpy())
                Ps = proba(m, sc.transform(X[s]))
                ext.append({"dataset": ds, "auroc_ref": auroc(y[s] >= 1, Ps[:, 1:].sum(1)),
                            "auroc_urg": auroc(y[s] == 2, Ps[:, 2])})
        ya, Pa = y[apt], P_oof[apt]
        r = {"features": key, "auroc_ref": auroc(ya >= 1, Pa[:, 1:].sum(1)), "auroc_urg": auroc(ya == 2, Pa[:, 2]),
             "same_camera_auroc_ref": same_camera_auroc(ya >= 1, Pa[:, 1:].sum(1), camera[apt]),
             **{f"c1_{m_}": v for m_, v in rule_metrics(dec[apt], ya).items()}}
        e = pd.DataFrame(ext).groupby("dataset")[["auroc_ref", "auroc_urg"]].mean()
        for ds in e.index:
            r[f"{ds}_auroc_ref"], r[f"{ds}_auroc_urg"] = e.loc[ds, "auroc_ref"], e.loc[ds, "auroc_urg"]
        img_rows.append(r)
        # all-APTOS model, then C2 at the new clinics
        sc = StandardScaler().fit(X[apt])
        m = fit_lr(sc.transform(X[apt]), y[apt], C_FIXED)
        P_all = proba(m, sc.transform(np.nan_to_num(X)))
        for (ds, k, b), (cal, test) in draws.items():
            q = classwise_thresholds(P_all[cal], y[cal], a)
            d = decide_from_sets((1 - P_all[test]) <= q[None, :])
            c2_rows.append({"features": key, "dataset": ds, "k": k, "draw": b, **rule_metrics(d, y[test])})
        print(f"features {key}: done ({time.time() - t0:.0f} s)")
    img = pd.DataFrame(img_rows)
    c2 = pd.DataFrame(c2_rows).groupby(["features", "dataset", "k"]).mean(numeric_only=True).drop(columns="draw")
    img.to_csv(out / "image_ablation.csv", index=False, float_format="%.5f")
    c2.reset_index().to_csv(out / "image_ablation_c2.csv", index=False, float_format="%.5f")

    # ---------------- 2. PCA size (E1 tuning logs) ----------------
    pca_rows = []
    for k in range(10):
        t = pd.read_csv(R / "e1" / "lr" / f"fold{k}_tuning.csv")
        cfg = t["config"].map(json.loads)
        t["pca"], t["C"] = cfg.map(lambda c: c["pca"]), cfg.map(lambda c: c["C"])
        for pca, g in t.groupby("pca"):
            bestrow = g.loc[g["score"].idxmax()]
            pca_rows.append({"fold": k, "pca": pca, "score": bestrow["score"], "auroc_ref": bestrow["auroc_referable"],
                             "auroc_urg": bestrow["auroc_urgent"], "C": bestrow["C"]})
    pca = pd.DataFrame(pca_rows)
    pca.to_csv(out / "pca_size.csv", index=False, float_format="%.5f")

    # ---------------- 3. robustness ----------------
    X0 = load_features(data, FEATURES["224"], man)
    alt = {v: load_features(data, fn, man) for v, fn in VARIANTS.items()}
    rob_rows, max_dev = [], 0.0
    for k, f in folds.items():
        fit = np.setdiff1d(apt, np.concatenate([f["test"], f["cal"]]))
        n_pca, C = best[k]["pca"], best[k]["C"]
        sc = StandardScaler().fit(X0[fit])
        pca_k = PCA(n_components=min(2048, len(fit), X0.shape[1]), svd_solver="full").fit(sc.transform(X0[fit]))
        tr = lambda X: pca_k.transform(sc.transform(X))[:, :n_pca]       # noqa: E731
        m = fit_lr(tr(X0[fit]), y[fit], C)
        Pc0, Pt0 = proba(m, tr(X0[f["cal"]])), proba(m, tr(X0[f["test"]]))
        saved = f["p_saved"].loc[f["test"]].to_numpy()
        max_dev = max(max_dev, float(np.abs(Pt0 - saved).max()))
        q0 = classwise_thresholds(Pc0, y[f["cal"]], a)
        yt = y[f["test"]]
        for v in ["none"] + [v for v in VARIANTS if alt[v] is not None]:
            Pt = Pt0 if v == "none" else proba(m, tr(alt[v][f["test"]]))
            Pc = Pc0 if v == "none" else proba(m, tr(alt[v][f["cal"]]))
            q1 = classwise_thresholds(Pc, y[f["cal"]], a)
            for thr_name, q in (("original calibration", q0), ("recalibrated on altered photos", q1)):
                d = decide_from_sets((1 - Pt) <= q[None, :])
                rob_rows.append({"fold": k, "variant": v, "thresholds": thr_name, "n": len(yt),
                                 "n_ref": int((yt >= 1).sum()), "n_urg": int((yt == 2).sum()), "n_non": int((yt == 0).sum()),
                                 **{f"{m_}_k": v_ for m_, v_ in rule_metrics(d, yt).items()},
                                 "pos_ref": (yt >= 1), "pos_urg": (yt == 2), "dec": d, "score_ref": Pt[:, 1:].sum(1),
                                 "score_urg": Pt[:, 2]})
    print(f"robustness done; largest difference from E1's saved predictions {max_dev:.2e} ({time.time() - t0:.0f} s)")
    rob = []
    df_rob = pd.DataFrame(rob_rows)
    for (v, thr_name), g in df_rob.groupby(["variant", "thresholds"], sort=False):
        d = np.concatenate(g["dec"].to_list())
        pr, pu = np.concatenate(g["pos_ref"].to_list()), np.concatenate(g["pos_urg"].to_list())
        yy = np.where(pu, 2, np.where(pr, 1, 0))
        rob.append({"variant": v, "thresholds": thr_name, **rule_metrics(d, yy),
                    "auroc_ref": auroc(pr, np.concatenate(g["score_ref"].to_list())),
                    "auroc_urg": auroc(pu, np.concatenate(g["score_urg"].to_list()))})
    rob = pd.DataFrame(rob)
    rob.to_csv(out / "robustness.csv", index=False, float_format="%.5f")

    write_summary(out, args, img, c2, pca, rob, max_dev, time.time() - t0)
    plot(img, c2, rob, out / "ablations.png")
    print(f"wrote {out / 'summary.md'} ({time.time() - t0:.0f} s)")


def write_summary(out, args, img, c2, pca, rob, max_dev, elapsed):
    a = args.alpha
    L = ["# Ablations and robustness", "",
         "## 1. Image ablations: resolution and colour normalisation", "",
         f"Logistic regression with E1's most frequent setting (standardised features, no PCA reduction, C = {C_FIXED}), "
         "refitted in every E1 outer fold; C1 (α = "
         f"{a:g}) fitted on each fold's calibration photos. New clinics: AUROC of the 10 fold models (mean).", "",
         "| Features | AUROC referable | Same-camera AUROC referable | AUROC urgent | C1: referable sent home | "
         "C1: urgent referred urgently | C1: healthy sent home | IDRiD AUROC referable | Messidor-2 AUROC referable | "
         "Messidor-2 AUROC urgent |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in img.iterrows():
        L.append(f"| {FEATURE_NAMES[r['features']]} | {r['auroc_ref']:.3f} | {r['same_camera_auroc_ref']:.3f} | "
                 f"{r['auroc_urg']:.3f} | {100 * r['c1_ref_home']:.1f} | {100 * r['c1_urgent_urgent']:.1f} | "
                 f"{100 * r['c1_healthy_home']:.1f} | {r['idrid_auroc_ref']:.3f} | {r['messidor2_auroc_ref']:.3f} | "
                 f"{r['messidor2_auroc_urg']:.3f} |")
    L += ["", f"C2 at the new clinics with each feature set (model trained on all of APTOS; {args.draws} draws of k "
          "labelled local photos, as in E2), in %:", "",
          "| Features | Clinic | k | Referable sent home | Urgent referred urgently | Healthy sent home | Uncertain |",
          "|---|---|---|---|---|---|---|"]
    for (feat, ds, k), r in c2.iterrows():
        L.append(f"| {FEATURE_NAMES[feat]} | {'IDRiD' if ds == 'idrid' else 'Messidor-2'} | {k} | "
                 f"{100 * r['ref_home']:.1f} | {100 * r['urgent_urgent']:.1f} | {100 * r['healthy_home']:.1f} | "
                 f"{100 * r['uncertain']:.1f} |")
    p = pca.groupby("pca")[["score", "auroc_ref", "auroc_urg"]].mean()
    L += ["", "## 2. PCA size (logistic regression, E1's inner folds)", "",
          "Best inner-fold score over C for each number of components, mean over the 10 outer folds (score: mean of "
          "the referable and urgent AUROCs).", "", "| PCA components | Score | AUROC referable | AUROC urgent |",
          "|---|---|---|---|"]
    for n, r in p.iterrows():
        L.append(f"| {n} | {r['score']:.4f} | {r['auroc_ref']:.4f} | {r['auroc_urg']:.4f} |")
    L += ["", "## 3. Robustness: worse photos at the clinic", "",
          "E1's logistic regression (each fold's chosen setting; its predictions on the original photos match E1's "
          f"saved ones to within {max_dev:.0e}). Only the test photos (and, for recalibration, the calibration photos) "
          "are altered; pooled over the 10 held-out folds, in %.", "",
          "| Test photos | C1 thresholds from | AUROC referable | AUROC urgent | Referable sent home | Urgent referred "
          "urgently | Healthy sent home | Uncertain |", "|---|---|---|---|---|---|---|---|"]
    for _, r in rob.iterrows():
        if r["variant"] == "none" and r["thresholds"] != "original calibration":
            continue
        L.append(f"| {VARIANT_NAMES[r['variant']]} | {r['thresholds']} | {r['auroc_ref']:.3f} | {r['auroc_urg']:.3f} | "
                 f"{100 * r['ref_home']:.1f} | {100 * r['urgent_urgent']:.1f} | {100 * r['healthy_home']:.1f} | "
                 f"{100 * r['uncertain']:.1f} |")
    L += ["", f"Made {time.strftime('%Y-%m-%d %H:%M')} by ablations.py ({elapsed:.0f} s).", ""]
    (out / "summary.md").write_text("\n".join(L))


def plot(img, c2, rob, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid_c = "#0b0b0b", "#52514e", "#e4e3df"
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), dpi=200)
    ax = axes[0]
    r = rob[rob["variant"] != "none"]
    variants = list(dict.fromkeys(r["variant"]))
    x = np.arange(len(variants))
    base = rob[(rob["variant"] == "none")].iloc[0]
    for j, (thr, col) in enumerate((("original calibration", "#a8a7a1"), ("recalibrated on altered photos", "#2a78d6"))):
        v = [100 * r[(r["variant"] == vv) & (r["thresholds"] == thr)]["urgent_urgent"].iloc[0] for vv in variants]
        ax.bar(x + (j - 0.5) * 0.36, v, width=0.34, color=col, label=f"thresholds: {thr}")
        for xi, vi in zip(x + (j - 0.5) * 0.36, v):
            ax.text(xi, vi + 1, f"{vi:.0f}", ha="center", fontsize=7, color=ink2)
    ax.axhline(100 * base["urgent_urgent"], color=ink2, lw=1, ls=(0, (1.5, 1.5)), label="original photos")
    ax.axhline(90, color=ink, lw=0.9, ls=(0, (4, 3)), label="target 90 %")
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_NAMES[v] for v in variants], fontsize=7.5)
    ax.set_ylim(0, 108)
    ax.set_title("C1: urgent eyes referred urgently (%), altered test photos", fontsize=9.5, color=ink, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax = axes[1]
    feats = list(img["features"])
    xx = np.arange(len(feats))
    for j, (col_key, name, col) in enumerate((("auroc_ref", "APTOS", "#2a78d6"), ("idrid_auroc_ref", "IDRiD", "#eb6834"),
                                             ("messidor2_auroc_ref", "Messidor-2", "#1baf7a"))):
        v = img[col_key].to_numpy()
        ax.bar(xx + (j - 1) * 0.26, v, width=0.24, color=col, label=name)
        for xi, vi in zip(xx + (j - 1) * 0.26, v):
            ax.text(xi, vi + 0.003, f"{vi:.3f}", ha="center", fontsize=6.3, color=ink2)
    ax.set_xticks(xx)
    ax.set_xticklabels([FEATURE_NAMES[f] for f in feats], fontsize=7.5)
    ax.set_ylim(0.6, 1.0)
    ax.set_title("AUROC referable by feature set", fontsize=9.5, color=ink, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=3)
    for ax in axes:
        ax.grid(axis="y", color=grid_c, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(ink2)
        ax.tick_params(colors=ink2, labelsize=7.5, length=0)
    fig.suptitle("Ablations and robustness (logistic regression, C1 at α = 0.1)", fontsize=10.5, color=ink, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
