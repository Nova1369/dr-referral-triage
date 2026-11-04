#!/usr/bin/env python3
"""
camera_fixes.py - additional work: two fixes for the APTOS camera shortcut, tuned and tested under
the E1 protocol, and what they do for the referral rule C1.

Models (logistic regression on standardised ResNet-50 features; C tuned in 5 grouped inner folds, by the mean
of the referable and urgent AUROCs, as in E1):
  plain     no fix: E1's logistic regression (its saved predictions)
  weights   camera-decorrelated weights w = P(class) / P(class | camera group), from the camera study
  erasure   linear concept erasure (LEACE, Belrose et al. 2023). The features are moved so that no linear
            function of them can tell the camera groups apart. The eraser is fitted on the grade-0 eyes of
            the training photos only (camera style without disease), so disease signal that happens to
            differ between cameras is not erased with it.
  erasure_grade
            the same eraser estimated within each DR grade, on all training photos: every direction in which
            the camera groups differ inside a grade is removed, so camera style is erased in sick eyes too,
            while the differences between grades (disease) are kept.

The summary also explains why the healthy-eye eraser does not help: camera probes on healthy and on referable
eyes before and after each eraser, the risk the models give grade-2 eyes from each camera group, and the size
of the weights.

Protocol: E1's outer folds and its calibration photos (read from results/e1/lr), so every APTOS photo is
tested once. C1 is fitted on each fold's calibration photos with the raw probabilities, as in results/c1.

Outputs (in --out, default ../results/camera_fixes):
  summary.md
  metrics.csv
  <fix>/fold<k>.csv.gz       predictions for test, calibration and external photos
  camera_fixes.png

Usage:
  python camera_fixes.py --data ../data_512 --results ../results
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from c1_referral import HOME, ROUTINE, URGENT, camera_groups, classwise_thresholds, decide_from_sets
from dr_preprocess import assign_folds

C_GRID = [3e-4, 1e-3, 3e-3, 1e-2]
FIXES = ["plain", "weights", "erasure", "erasure_grade"]
FIX_NAMES = {"plain": "No fix (E1)", "weights": "Camera-decorrelated weights",
             "erasure": "Camera erasure fitted on healthy eyes (LEACE)",
             "erasure_grade": "Camera erasure within each grade (LEACE)"}


# --------------------------------------------------------------------------------------------
# the two fixes
# --------------------------------------------------------------------------------------------

def decorrelating_weights(y, g):
    p_y = np.bincount(y, minlength=3) / len(y)
    w = np.ones(len(y))
    for grp in np.unique(g):
        s = g == grp
        p_yg = np.bincount(y[s], minlength=3) / s.sum()
        for c in range(3):
            w[s & (y == c)] = p_y[c] / p_yg[c]
    return w / w.mean()


def fit_leace(Z, groups):
    """LEACE eraser for the concept `groups` (one-hot), fitted on the rows of Z.
    r(x) = x - W+ P W (x - mu): W whitens (Ledoit-Wolf covariance), P projects onto the span of
    W Cov(x, z). Afterwards every camera group has the same mean of every linear feature."""
    mu = Z.mean(0)
    Zc = Z - mu
    S = LedoitWolf().fit(Zc).covariance_
    evals, evecs = np.linalg.eigh(S)
    evals = np.clip(evals, 1e-10, None)
    W = (evecs * evals ** -0.5) @ evecs.T
    W_inv = (evecs * evals ** 0.5) @ evecs.T
    G = np.stack([groups == g for g in np.unique(groups)], 1).astype(float)
    Sxz = Zc.T @ (G - G.mean(0)) / len(Z)
    U, sv, _ = np.linalg.svd(W @ Sxz, full_matrices=False)
    U = U[:, sv > 1e-8 * sv.max()]
    M = W_inv @ (U @ U.T) @ W
    return mu, M


def fit_leace_within(Z, groups, strata):
    """LEACE eraser for `groups`, estimated within each stratum (here the DR grade). The concept is one indicator
    per (group, stratum), centred within its stratum, so afterwards every camera group has the same mean of every
    linear feature inside each grade; the means of the grades themselves are not touched."""
    mu = Z.mean(0)
    Zc = Z - mu
    S = LedoitWolf().fit(Zc).covariance_
    evals, evecs = np.linalg.eigh(S)
    evals = np.clip(evals, 1e-10, None)
    W = (evecs * evals ** -0.5) @ evecs.T
    W_inv = (evecs * evals ** 0.5) @ evecs.T
    cols = []
    for st in np.unique(strata):
        m = strata == st
        present = np.unique(groups[m])
        if len(present) < 2:
            continue
        for grp in present:
            c = np.zeros(len(Z))
            c[m] = (groups[m] == grp) - np.mean(groups[m] == grp)
            cols.append(c)
    G = np.stack(cols, 1)
    Sxz = Zc.T @ G / len(Z)
    U, sv, _ = np.linalg.svd(W @ Sxz, full_matrices=False)
    U = U[:, sv > 1e-8 * sv.max()]
    M = W_inv @ (U @ U.T) @ W
    return mu, M


def apply_leace(Z, eraser):
    mu, M = eraser
    return Z - (Z - mu) @ M.T


def fit_lr(Z, y, w, C):
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LogisticRegression(C=C, max_iter=5000).fit(Z, y, sample_weight=w)


def prepare(fix, X_fit, y_fit, g_fit, grade_fit):
    """Scaler (and eraser) fitted on the training rows only; returns transform and sample weights."""
    scaler = StandardScaler().fit(X_fit)
    Z_fit = scaler.transform(X_fit)
    w, eraser = None, None
    if fix == "weights":
        w = decorrelating_weights(y_fit, g_fit)
    if fix == "erasure":
        healthy = grade_fit == 0
        eraser = fit_leace(Z_fit[healthy], g_fit[healthy])
    if fix == "erasure_grade":
        eraser = fit_leace_within(Z_fit, g_fit, grade_fit)

    def transform(X):
        Z = scaler.transform(X)
        return apply_leace(Z, eraser) if eraser is not None else Z
    return transform, w


def auroc(pos, score):
    from scipy.stats import rankdata
    n1 = pos.sum()
    n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def inner_task(fix, X, y, g, grade, tr, va, C_grid):
    transform, w = prepare(fix, X[tr], y[tr], g[tr], grade[tr])
    Ztr, Zva = transform(X[tr]), transform(X[va])
    out = []
    for C in C_grid:
        m = fit_lr(Ztr, y[tr], w, C)
        P = np.zeros((len(va), 3))
        P[:, m.classes_] = m.predict_proba(Zva)
        out.append(P)
    return va, out


def outer_fold(fix, k, X, y, g, grade, man, fit_idx, other_idx, seed, jobs_inner):
    inner = assign_folds(man.iloc[fit_idx].reset_index(drop=True), 5, seed + 1000 + k)[0].to_numpy()
    res = [inner_task(fix, X, y, g, grade, fit_idx[inner != j], fit_idx[inner == j], C_GRID) for j in range(5)]
    oof = {C: np.zeros((len(y), 3)) for C in C_GRID}
    for va, Ps in res:
        for C, P in zip(C_GRID, Ps):
            oof[C][va] = P
    score = {C: np.mean([auroc(y[fit_idx] >= 1, oof[C][fit_idx, 1:].sum(1)),
                         auroc(y[fit_idx] == 2, oof[C][fit_idx, 2])]) for C in C_GRID}
    best = max(score, key=score.get)
    transform, w = prepare(fix, X[fit_idx], y[fit_idx], g[fit_idx], grade[fit_idx])
    m = fit_lr(transform(X[fit_idx]), y[fit_idx], w, best)
    P = np.zeros((len(other_idx), 3))
    P[:, m.classes_] = m.predict_proba(transform(X[other_idx]))
    return fix, k, best, score[best], P


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Additional work: camera-shortcut fixes under the E1 protocol.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", default="../data_512")
    ap.add_argument("--features", default="features/resnet50_224.npz")
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default="../results/camera_fixes")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R = Path(args.results)

    man = pd.read_csv(Path(args.data) / "manifest.csv")
    with np.load(Path(args.data) / args.features) as z:
        X, uid = z["X"], z["uid"]
    assert (uid == man["uid"].astype(str).to_numpy()).all()
    y = man["ref3"].to_numpy(int)
    grade = man["grade"].to_numpy(int)
    g = camera_groups(man).fillna("other").to_numpy()
    camera = (man["orig_w"].astype(str) + "x" + man["orig_h"].astype(str)).to_numpy()
    row = pd.Series(np.arange(len(man)), index=man["uid"])

    # E1's outer folds and calibration photos
    folds = {}
    for k in range(10):
        f = pd.read_csv(R / "e1" / "lr" / f"fold{k}.csv.gz")
        folds[k] = {r: row[f.loc[f["role"] == r, "uid"]].to_numpy() for r in ("test", "cal", "external")}
        folds[k]["plain"] = f[["p0", "p1", "p2"]].to_numpy()
        folds[k]["order"] = row[f["uid"]].to_numpy()
    apt = np.flatnonzero((man["dataset"] == "aptos").to_numpy())

    preds = {"plain": {k: (folds[k]["order"], folds[k]["plain"]) for k in folds}}
    tasks = []
    for fix in FIXES[1:]:
        for k in folds:
            used = np.concatenate([folds[k]["test"], folds[k]["cal"]])
            fit_idx = np.setdiff1d(apt, used)
            other = folds[k]["order"]
            tasks.append(delayed(outer_fold)(fix, k, X, y, g, grade, man, fit_idx, other, args.seed, 1))
    chosen = []
    res = Parallel(n_jobs=args.jobs, verbose=10)(tasks)
    for fix, k, best, sc, P in res:
        preds.setdefault(fix, {})[k] = (folds[k]["order"], P)
        chosen.append({"fix": fix, "fold": k, "C": best, "inner_score": sc})
    print(f"models fitted ({time.time() - t0:.0f} s)")
    for fix in FIXES[1:]:
        (out / fix).mkdir(exist_ok=True)
        for k, (order, P) in preds[fix].items():
            role = np.full(len(order), "external", dtype=object)
            role[np.isin(order, folds[k]["test"])] = "test"
            role[np.isin(order, folds[k]["cal"])] = "cal"
            pd.DataFrame({"uid": man["uid"].to_numpy()[order], "role": role, "p0": P[:, 0], "p1": P[:, 1],
                          "p2": P[:, 2]}).to_csv(out / fix / f"fold{k}.csv.gz", index=False, float_format="%.6g")

    # ---- evaluation ----
    rows, group_rows = [], []
    for fix in FIXES:
        P_test = np.zeros((len(man), 3))
        dec_test = np.full(len(man), -1)
        ext = {"idrid": [], "messidor2": []}
        for k, (order, P) in preds[fix].items():
            pos = {u: i for i, u in enumerate(order)}
            cal = np.array([pos[i] for i in folds[k]["cal"]])
            te = np.array([pos[i] for i in folds[k]["test"]])
            q = classwise_thresholds(P[cal], y[order[cal]], args.alpha)
            P_test[order[te]] = P[te]
            dec_test[order[te]] = decide_from_sets((1 - P[te]) <= q[None, :])
            for ds in ext:
                s = np.flatnonzero(man["dataset"].to_numpy()[order] == ds)
                d = decide_from_sets((1 - P[s]) <= q[None, :])
                yy = y[order[s]]
                ext[ds].append({"auroc_ref": auroc(yy >= 1, P[s, 1:].sum(1)), "auroc_urg": auroc(yy == 2, P[s, 2]),
                                "ref_home": np.mean(d[yy >= 1] == HOME), "urgent_urgent": np.mean(d[yy == 2] == URGENT),
                                "healthy_home": np.mean(d[yy == 0] == HOME)})
        a = apt
        ya, Pa, da, ga, ca, gra = y[a], P_test[a], dec_test[a], g[a], camera[a], grade[a]
        within_num = within_den = 0.0
        for c in np.unique(ca):
            s = ca == c
            n1, n0 = int((ya[s] >= 1).sum()), int((ya[s] == 0).sum())
            if n1 and n0:
                within_num += auroc(ya[s] >= 1, Pa[s, 1:].sum(1)) * n1 * n0
                within_den += n1 * n0
        am = Pa.argmax(1)
        r = {"fix": fix, "auroc_ref": auroc(ya >= 1, Pa[:, 1:].sum(1)), "auroc_urg": auroc(ya == 2, Pa[:, 2]),
             "same_camera_auroc_ref": within_num / within_den,
             "c1_ref_home": np.mean(da[ya >= 1] == HOME), "c1_urgent_urgent": np.mean(da[ya == 2] == URGENT),
             "c1_healthy_home": np.mean(da[ya == 0] == HOME),
             "c1_referred": np.mean((da == ROUTINE) | (da == URGENT))}
        for ds, lst in ext.items():
            for key in lst[0]:
                r[f"{ds}_{key}"] = np.mean([e[key] for e in lst])
        rows.append(r)
        for grp in ("H", "M", "S"):
            s = ga == grp
            group_rows.append({"fix": fix, "group": grp, "eyes": int(s.sum()), "referable": int((ya[s] >= 1).sum()),
                               "argmax_ref_home": np.mean(am[s & (ya >= 1)] == 0),
                               "argmax_healthy_referred": np.mean(am[s & (ya == 0)] >= 1),
                               "c1_ref_home": np.mean(da[s & (ya >= 1)] == HOME),
                               "c1_urgent_urgent": np.mean(da[s & (ya == 2)] == URGENT),
                               "c1_healthy_home": np.mean(da[s & (ya == 0)] == HOME),
                               "grade2_eyes": int(np.sum(s & (gra == 2))),
                               "grade2_median_p_ref": float(np.median(Pa[s & (gra == 2), 1:].sum(1)))})
    met = pd.DataFrame(rows)
    grp = pd.DataFrame(group_rows)
    met.to_csv(out / "metrics.csv", index=False, float_format="%.5f")
    grp.to_csv(out / "by_camera_group.csv", index=False, float_format="%.5f")
    pd.DataFrame(chosen).to_csv(out / "chosen_C.csv", index=False)

    # erasure check: can the camera still be read from healthy and from referable eyes after each eraser?
    check = erasure_check(X, y, g, grade, man, apt, args.seed)
    check.to_csv(out / "erasure_check.csv", index=False, float_format="%.4f")
    cells = calibration_cells(folds, y, g)
    weights = weight_table(y[apt], g[apt])
    write_summary(out, args, met, grp, pd.DataFrame(chosen), check, cells, weights, time.time() - t0)
    plot(met, grp, out / "camera_fixes.png")
    print(f"wrote {out / 'summary.md'} ({time.time() - t0:.0f} s)")


def erasure_check(X, y, g, grade, man, apt, seed):
    """Camera probes (H against M cameras, AUROC) on APTOS healthy (grade-0) eyes and on referable eyes, before
    and after each eraser. 5 grouped folds; scaler and eraser fitted on the training part of each probe fold,
    as in the models. The mostly-sick cameras have almost no grade-0 eyes, so the probes separate H from M."""
    ma = man.iloc[apt].reset_index(drop=True)
    fold = assign_folds(ma, 5, seed + 7)[0].to_numpy()
    Xa, ga, ya, gra = X[apt], g[apt], y[apt], grade[apt]
    eyes = {"healthy (grade 0)": gra == 0, "referable (grades 2-4)": ya >= 1}
    scores = {(e, w): ([], []) for e in eyes for w in ("none", "erasure", "erasure_grade")}
    for f in range(5):
        tr, te = fold != f, fold == f
        sc = StandardScaler().fit(Xa[tr])
        Z = {"none": (sc.transform(Xa[tr]), sc.transform(Xa[te]))}
        h = tr & (gra == 0)
        for name, er in (("erasure", fit_leace(sc.transform(Xa[h]), ga[h])),
                         ("erasure_grade", fit_leace_within(Z["none"][0], ga[tr], gra[tr]))):
            Z[name] = (apply_leace(Z["none"][0], er), apply_leace(Z["none"][1], er))
        for e, sel in eyes.items():
            s_tr = sel[tr] & np.isin(ga[tr], ["H", "M"])
            s_te = sel[te] & np.isin(ga[te], ["H", "M"])
            for w, (A, B) in Z.items():
                with threadpool_limits(limits=1), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = LogisticRegression(C=0.01, max_iter=5000, class_weight="balanced").fit(
                        A[s_tr], ga[tr][s_tr] == "H")
                scores[(e, w)][0].append(ga[te][s_te] == "H")
                scores[(e, w)][1].append(m.decision_function(B[s_te]))
    rows = []
    for (e, w), (lab, sco) in scores.items():
        lab, sco = np.concatenate(lab), np.concatenate(sco)
        rows.append({"eyes": e, "eraser": w, "n": len(lab), "n_H": int(lab.sum()), "auroc": auroc(lab, sco)})
    return pd.DataFrame(rows)


def weight_table(y, g):
    """Mean decorrelating weight (normalised to mean 1) and number of eyes per camera group and class."""
    w = decorrelating_weights(y, g)
    rows = []
    for grp in ("H", "M", "S"):
        for c in range(3):
            s = (g == grp) & (y == c)
            rows.append({"group": grp, "class": c, "eyes": int(s.sum()), "weight": float(w[s].mean()) if s.any() else np.nan})
    return pd.DataFrame(rows)


def calibration_cells(folds, y, g):
    """Calibration photos per class and camera group, mean over folds: can each camera have its own threshold?"""
    rows = []
    for k, f in folds.items():
        cal = f["cal"]
        for grp in ("H", "M", "S"):
            for c in range(3):
                rows.append({"fold": k, "group": grp, "class": c, "n": int(np.sum((g[cal] == grp) & (y[cal] == c)))})
    return pd.DataFrame(rows).groupby(["group", "class"])["n"].mean().unstack()


def write_summary(out, args, met, grp, chosen, check, cells, weights, elapsed):
    M = met.set_index("fix")
    G = grp.set_index(["fix", "group"])
    L = ["# Additional work: fixing the camera shortcut, tuned under the E1 protocol", "",
         "Logistic regression on standardised ResNet-50 features; C tuned in 5 grouped inner folds (mean of the two "
         "AUROCs), E1's outer folds and calibration photos. C1 (one conformal threshold per class, α = "
         f"{args.alpha:g}) is fitted on each fold's calibration photos with the raw probabilities.", "",
         "- **Weights:** w = P(class) / P(class | camera group): in the weighted training data the camera says "
         "nothing about the class.",
         "- **Erasure fitted on healthy eyes (LEACE):** the standardised features are moved so that every camera "
         "group has the same mean of every linear feature. The eraser is fitted on the grade-0 training eyes only, so "
         "it removes camera style and keeps disease.",
         "- **Erasure within each grade (LEACE):** the same kind of eraser, estimated within each DR grade on all "
         "training eyes, so the camera groups have the same means inside every grade, sick eyes included; the "
         "differences between grades are kept.",
         "",
         "Camera probes (H against M cameras, AUROC; 0.5 = the camera cannot be read; 5 grouped folds, eraser fitted "
         "on the training part):", "",
         "| Eyes | Eyes in the probe (H) | No erasure | Erasure fitted on healthy eyes | Erasure within each grade |",
         "|---|---|---|---|---|"]
    ck = check.set_index(["eyes", "eraser"])
    for e in check["eyes"].unique():
        L.append(f"| {e} | {ck.loc[(e, 'none'), 'n']} ({ck.loc[(e, 'none'), 'n_H']}) | "
                 + " | ".join(f"{ck.loc[(e, w), 'auroc']:.3f}" for w in ("none", "erasure", "erasure_grade")) + " |")
    L += [
         "", "C chosen per fold: " + "; ".join(f"{FIX_NAMES[f]}: " + ", ".join(f"{c:g}" for c in chosen[chosen['fix'] == f]['C'])
                                               for f in FIXES[1:]), "",
         "## 1. APTOS, out-of-fold", "",
         "| Model | AUROC referable | Same-camera AUROC referable | AUROC urgent | C1: referable sent home | "
         "C1: urgent referred urgently | C1: healthy sent home | C1: referred per 1,000 |", "|---|---|---|---|---|---|---|---|"]
    for f in FIXES:
        r = M.loc[f]
        L.append(f"| {FIX_NAMES[f]} | {r['auroc_ref']:.3f} | {r['same_camera_auroc_ref']:.3f} | {r['auroc_urg']:.3f} | "
                 f"{100 * r['c1_ref_home']:.1f} | {100 * r['c1_urgent_urgent']:.1f} | {100 * r['c1_healthy_home']:.1f} | "
                 f"{1000 * r['c1_referred']:.0f} |")
    L += ["", "## 2. By camera group (APTOS)", "",
          "Most likely class (as in the camera study) and the C1 rule, in %.", "",
          "| Model | Camera group | Referable eyes | Most likely class: referable sent home | Most likely class: healthy "
          "referred | C1: referable sent home | C1: urgent referred urgently | C1: healthy sent home |",
          "|---|---|---|---|---|---|---|---|"]
    for f in FIXES:
        for gname in ("H", "M", "S"):
            r = G.loc[(f, gname)]
            L.append(f"| {FIX_NAMES[f]} | {gname} | {r['referable']} | {100 * r['argmax_ref_home']:.1f} | "
                     f"{100 * r['argmax_healthy_referred']:.1f} | {100 * r['c1_ref_home']:.1f} | "
                     f"{100 * r['c1_urgent_urgent']:.1f} | {100 * r['c1_healthy_home']:.1f} |")
    L += ["", "## 3. The new clinics (the 10 fold models, APTOS thresholds, no recalibration)", "",
          "| Model | IDRiD AUROC referable | IDRiD: referable sent home / urgent referred urgently / healthy sent home | "
          "Messidor-2 AUROC referable | Messidor-2: referable sent home / urgent referred urgently / healthy sent home |",
          "|---|---|---|---|---|"]
    for f in FIXES:
        r = M.loc[f]
        L.append(f"| {FIX_NAMES[f]} | {r['idrid_auroc_ref']:.3f} | {100 * r['idrid_ref_home']:.1f} / "
                 f"{100 * r['idrid_urgent_urgent']:.1f} / {100 * r['idrid_healthy_home']:.1f} | "
                 f"{r['messidor2_auroc_ref']:.3f} | {100 * r['messidor2_ref_home']:.1f} / "
                 f"{100 * r['messidor2_urgent_urgent']:.1f} / {100 * r['messidor2_healthy_home']:.1f} |")
    L += ["", "## 4. Why not one threshold per camera group (Mondrian by camera)?", "",
          "Calibration photos per class and camera group, mean over the 10 folds. A class needs at least "
          f"{int(np.ceil(1 / args.alpha - 1))} calibration photos for a finite threshold.", "",
          "| Camera group | No referral | Routine | Urgent |", "|---|---|---|---|"]
    for gname in ("H", "M", "S"):
        L.append(f"| {gname} | " + " | ".join(f"{cells.loc[gname, c]:.1f}" for c in range(3)) + " |")
    L += ["", f"The mostly-healthy cameras have about {cells.loc['H', 1]:.0f} routine and {cells.loc['H', 2]:.0f} urgent "
          f"calibration photos per fold, and the mostly-sick cameras about {cells.loc['S', 0]:.0f} healthy ones: "
          "per-camera thresholds would put those classes in every set for those cameras and refer everyone. APTOS "
          "does not have enough sick eyes from the H cameras (55 in all) to give those cameras their own guarantee, "
          "so the fix has to be in the model, not in the calibration.", "",
          "## 5. Why the healthy-eye erasure does not help", "",
          "Median P(referable) that each model gives the grade-2 (routine) eyes of each camera group, out of fold:", "",
          "| Model | " + " | ".join(f"{gname} cameras ({int(G.loc[('plain', gname), 'grade2_eyes'])} eyes)"
                                    for gname in ("H", "M", "S")) + " |", "|---|---|---|---|"]
    for f in FIXES:
        L.append(f"| {FIX_NAMES[f]} | " + " | ".join(f"{G.loc[(f, gname), 'grade2_median_p_ref']:.2f}"
                                                   for gname in ("H", "M", "S")) + " |")
    W = weights.set_index(["group", "class"])
    L += ["", "Decorrelating weights (mean 1 over APTOS) and APTOS eyes per camera group and class:", "",
          "| Camera group | No referral | Routine | Urgent |", "|---|---|---|---|"]
    for gname in ("H", "M", "S"):
        L.append(f"| {gname} | " + " | ".join(f"{W.loc[(gname, c), 'weight']:.2f} ({W.loc[(gname, c), 'eyes']})"
                                             for c in range(3)) + " |")
    hc, rf = "healthy (grade 0)", "referable (grades 2-4)"
    L += ["", f"The plain model treats grade-2 eyes from the mostly-healthy cameras as healthy (median P(referable) "
          f"{G.loc[('plain', 'H'), 'grade2_median_p_ref']:.2f}, against {G.loc[('plain', 'M'), 'grade2_median_p_ref']:.2f} "
          "on M cameras): in its training data almost every H-camera photo is healthy. The eraser fitted on healthy eyes "
          "removes the camera only where it was estimated. Afterwards the camera can no longer be read from healthy "
          f"eyes (probe AUROC {ck.loc[(hc, 'erasure'), 'auroc']:.2f}), but it can still be read from referable eyes "
          f"({ck.loc[(rf, 'erasure'), 'auroc']:.3f}, against {ck.loc[(rf, 'none'), 'auroc']:.3f} before), so the model "
          "can still learn that sick-looking H-camera photos are rare. Erasing within each grade brings the "
          f"referable-eye probe to {ck.loc[(rf, 'erasure_grade'), 'auroc']:.3f}. The weights change the rarity itself: "
          f"the routine and urgent H-camera eyes count {W.loc[('H', 1), 'weight']:.0f} and {W.loc[('H', 2), 'weight']:.0f} "
          "times the average photo.", "",
          f"Made {time.strftime('%Y-%m-%d %H:%M')} by camera_fixes.py ({elapsed:.0f} s).", ""]
    (out / "summary.md").write_text("\n".join(L))


def plot(met, grp, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid_c = "#0b0b0b", "#52514e", "#e4e3df"
    cols = {"plain": "#a8a7a1", "weights": "#2a78d6", "erasure": "#eb6834", "erasure_grade": "#1baf7a"}
    nb, bw = len(FIXES), 0.2
    off = lambda j: (j - (nb - 1) / 2) * (bw + 0.01)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), dpi=200)
    groups = ["H", "M", "S"]
    x = np.arange(3)
    for ax, (key, title) in zip(axes[:2], (("c1_ref_home", "C1: referable eyes sent home (%)"),
                                           ("c1_healthy_home", "C1: healthy eyes sent home (%)"))):
        for j, f in enumerate(FIXES):
            v = [100 * grp[(grp["fix"] == f) & (grp["group"] == gname)][key].iloc[0] for gname in groups]
            ax.bar(x + off(j), v, width=bw, color=cols[f], label=FIX_NAMES[f])
            for xi, vi in zip(x + off(j), v):
                ax.text(xi, vi + 1, f"{vi:.0f}", ha="center", fontsize=6.2, color=ink2)
        ax.set_xticks(x)
        ax.set_xticklabels(["H: mostly healthy", "M: mixed", "S: mostly sick"], fontsize=7.5)
        ax.set_title(title, fontsize=9.5, color=ink, loc="left")
        ax.set_ylim(0, 105)
        if key == "c1_ref_home":
            ax.axhline(10, color=ink, lw=0.9, ls=(0, (4, 3)))
            ax.text(2.45, 11.5, "target ≤ 10 %", ha="right", fontsize=7, color=ink)
    ax = axes[2]
    m = met.set_index("fix")
    labels = ["APTOS\nall pairs", "APTOS\nsame camera", "IDRiD", "Messidor-2"]
    keys = ["auroc_ref", "same_camera_auroc_ref", "idrid_auroc_ref", "messidor2_auroc_ref"]
    xx = np.arange(len(keys))
    for j, f in enumerate(FIXES):
        v = [m.loc[f, k] for k in keys]
        ax.bar(xx + off(j), v, width=bw, color=cols[f])
        for xi, vi in zip(xx + off(j), v):
            ax.text(xi, vi + 0.004, f"{vi:.3f}", ha="center", fontsize=5.8, color=ink2, rotation=90)
    ax.set_xticks(xx)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0.6, 1.02)
    ax.set_title("AUROC referable", fontsize=9.5, color=ink, loc="left")
    for ax in axes:
        ax.grid(axis="y", color=grid_c, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(ink2)
        ax.tick_params(colors=ink2, labelsize=7.5, length=0)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, fontsize=7.5, loc="lower center", ncol=len(FIXES),
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Additional work: camera-shortcut fixes, tuned under the E1 protocol (APTOS out-of-fold; new "
                 "clinics without recalibration)", fontsize=10.5, color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
