#!/usr/bin/env python3
"""
camera_shortcut.py - additional study: the APTOS camera shortcut, and what it does at new clinics.

Question. APTOS photos come from several cameras (photo sizes) that saw very different patients: some
cameras photographed almost only healthy eyes, others almost only sick eyes. Do the models learn the
camera instead of the retina, and does that explain who is missed at IDRiD and Messidor-2?

Part 1  How much camera information is in the features?
        APTOS cameras are grouped by their share of referable eyes: H (<= 10 %), M (mixed), S (>= 80 %).
        Probes are trained on grade-0 eyes only (no DR at all), so disease cannot help them:
          - the camera from the ResNet-50 features, and from the quality columns alone (the B2b inputs)
          - the dataset (APTOS / IDRiD / Messidor-2) from the ResNet-50 features
Part 2  Who is missed?
        - APTOS, E1 out-of-fold predictions: sensitivity and specificity per camera group and grade
        - a camera-style score h(x) = P(mostly-healthy camera | x), learnt from APTOS grade-0 eyes only
          (out-of-fold for APTOS), applied to IDRiD and Messidor-2
        - IDRiD and Messidor-2, the 10 E1 fold models averaged: misses of referable eyes and E0
          standard-conformal coverage, by tertile of h within each dataset
Part 3  First look at two fixes, against all of APTOS:
          - camera-decorrelated weights: every photo weighted by P(class) / P(class | camera group)
          - training only on the mixed cameras (M), with random subsets of the same size and class mix
            as its control
        Tested on IDRiD and Messidor-2, and (weights) by 10-fold cross-validation on APTOS. Same model
        everywhere: standardised features + logistic regression, C = 0.003 (E1's most frequent setting).

Inputs: manifest.csv, the ResNet-50 features and the E1 predictions of run_models.py (lr, svm, b2a).
Outputs (in --out, default ../results/camera_shortcut): summary.md, *.csv, aptos_misses.png (APTOS misses
by camera group), external_h.png (camera style and misses: APTOS camera groups vs the other clinics),
fix_preview.png (camera-decorrelated weights against plain training)

Usage:
  python camera_shortcut.py --data ../data_512 --results ../results/e1
"""
from __future__ import annotations

import argparse
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

QUAL_COLS = ["q_valid_frac", "q_mean", "q_std", "q_r", "q_g", "q_b", "q_dark", "q_sat", "q_lapvar", "q_entropy"]
GROUPS = {"H": "mostly-healthy cameras (≤ 10 % referable)", "M": "mixed cameras",
          "S": "mostly-sick cameras (≥ 80 % referable)"}
DATASETS = {"aptos": "APTOS", "idrid": "IDRiD", "messidor2": "Messidor-2"}
C_FIX = 0.003


# ------------------------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------------------------

def auroc(pos, score):
    n1 = int(pos.sum())
    n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def logreg(C=0.01, balanced=True):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000,
                                                              class_weight="balanced" if balanced else None))


def group_boot(groups, n_boot, seed):
    """Yield index arrays of a bootstrap that resamples whole groups."""
    g = pd.factorize(groups)[0]
    G = g.max() + 1
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, G, G), minlength=G)[g]
        yield np.repeat(np.arange(len(g)), w)


def ci_text(v, lo, hi, pct=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    f = 100 if pct else 1
    d = 1 if pct else 3
    return f"{f * v:.{d}f} ({f * lo:.{d}f}–{f * hi:.{d}f})"


def load_preds(results: Path, key: str):
    """Out-of-fold APTOS test predictions, and per-fold calibration and external predictions."""
    frames = []
    for f in sorted(results.joinpath(key).glob("fold*.csv.gz")):
        if f.name[4:].split(".")[0].isdigit():
            frames.append(pd.read_csv(f).assign(fold=int(f.name[4:].split(".")[0])))
    if not frames:
        raise SystemExit(f"no predictions in {results / key}; run run_models.py first")
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Additional study: the APTOS camera shortcut.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", default="../data_512")
    ap.add_argument("--features", default="features/resnet50_224.npz", help="relative to --data")
    ap.add_argument("--results", default="../results/e1", help="E1 predictions (run_models.py)")
    ap.add_argument("--out", default="../results/camera_shortcut")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=20, help="random subsets in part 3")
    ap.add_argument("--alpha", type=float, default=0.1, help="for the E0 coverage in part 2")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore", category=UserWarning)
    t0 = time.time()
    out, res = Path(args.out), Path(args.results)
    out.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(Path(args.data) / "manifest.csv")
    with np.load(Path(args.data) / args.features) as z:
        X, uid = z["X"], z["uid"]
    assert (uid == man["uid"].astype(str).to_numpy()).all(), "features do not match manifest.csv"
    man["camera"] = man["orig_w"].astype(str) + "x" + man["orig_h"].astype(str)
    is_apt = (man["dataset"] == "aptos").to_numpy()
    lines = ["# Additional study: the APTOS camera shortcut", "",
             "APTOS photos come from several cameras (identified here by photo size) that saw very different "
             "patients. This study asks whether the models learn the camera instead of the retina, and whether "
             "that explains who is missed at IDRiD and Messidor-2.", ""]

    # ============================ Part 1: cameras and probes ============================
    apt = man[is_apt]
    cams = apt.groupby("camera").agg(n=("uid", "size"), referable=("ref3", lambda v: (v >= 1).mean()),
                                     urgent=("ref3", lambda v: (v == 2).mean()),
                                     grade0=("grade", lambda v: int((v == 0).sum())))
    cams["group"] = np.where(cams["referable"] <= 0.10, "H", np.where(cams["referable"] >= 0.80, "S", "M"))
    cams = cams.sort_values("referable")
    cams.to_csv(out / "camera_groups.csv", float_format="%.4f")
    man["cam_group"] = man["camera"].map(cams["group"]).where(is_apt)
    grp_n = man[is_apt].groupby("cam_group").agg(n=("uid", "size"), ref=("ref3", lambda v: int((v >= 1).sum())))
    lines += ["## 1. Cameras and how much the features know about them", "",
              "APTOS cameras grouped by their share of referable eyes:", "",
              "| Group | Cameras (photo sizes) | Photos | Referable photos |", "|---|---|---|---|"]
    for g in "HMS":
        sizes = ", ".join(cams.index[cams["group"] == g])
        lines.append(f"| {GROUPS[g]} | {sizes} | {grp_n.loc[g, 'n']:,} | {grp_n.loc[g, 'ref']:,} "
                     f"({grp_n.loc[g, 'ref'] / grp_n.loc[g, 'n']:.0%}) |")

    fold = man["fold"].to_numpy()
    g0 = (man["grade"] == 0).to_numpy()
    probe_rows = []

    # (a) camera from ResNet features, APTOS grade-0 eyes, cameras with >= 20 such eyes
    big = cams.index[cams["grade0"] >= 20]
    sel = is_apt & g0 & man["camera"].isin(big).to_numpy()
    y_cam = man["camera"].to_numpy()
    pred_f, pred_q = np.empty(len(man), object), np.empty(len(man), object)
    for k in range(10):
        tr, te = sel & (fold != k), sel & (fold == k)
        pred_f[te] = logreg().fit(X[tr], y_cam[tr]).predict(X[te])
        hgb = HistGradientBoostingClassifier(random_state=args.seed, class_weight="balanced")
        Q = man[QUAL_COLS].to_numpy(float)
        pred_q[te] = hgb.fit(Q[tr], y_cam[tr]).predict(Q[te])
    ba_f = balanced_accuracy_score(y_cam[sel], pred_f[sel].astype(str))
    ba_q = balanced_accuracy_score(y_cam[sel], pred_q[sel].astype(str))
    probe_rows += [{"probe": "camera from ResNet-50 features", "eyes": "APTOS grade 0", "classes": len(big),
                    "n": int(sel.sum()), "balanced_accuracy": ba_f, "chance": 1 / len(big)},
                   {"probe": "camera from quality columns (B2b inputs)", "eyes": "APTOS grade 0",
                    "classes": len(big), "n": int(sel.sum()), "balanced_accuracy": ba_q, "chance": 1 / len(big)}]

    # (b) dataset from ResNet features, grade-0 eyes of all three datasets (grouped 5-fold)
    sel_d = g0
    y_ds = man["dataset"].to_numpy()
    pred_d = np.empty(len(man), object)
    idx_d = np.flatnonzero(sel_d)
    for tr_i, te_i in GroupKFold(5).split(idx_d, groups=man["group_id"].to_numpy()[idx_d]):
        tr, te = idx_d[tr_i], idx_d[te_i]
        pred_d[te] = logreg().fit(X[tr], y_ds[tr]).predict(X[te])
    ba_d = balanced_accuracy_score(y_ds[sel_d], pred_d[sel_d].astype(str))
    probe_rows.append({"probe": "dataset from ResNet-50 features", "eyes": "grade 0, all datasets", "classes": 3,
                       "n": int(sel_d.sum()), "balanced_accuracy": ba_d, "chance": 1 / 3})

    # (c) camera-style score h = P(mostly-healthy camera | x), trained on APTOS grade-0 eyes only
    is_h = (man["cam_group"] == "H").to_numpy()
    train_h = is_apt & g0
    h = np.full(len(man), np.nan)
    for k in range(10):
        tr = train_h & (fold != k)
        h[is_apt & (fold == k)] = logreg().fit(X[tr], is_h[tr]).predict_proba(X[is_apt & (fold == k)])[:, 1]
    full_h = logreg().fit(X[train_h], is_h[train_h])
    h[~is_apt] = full_h.predict_proba(X[~is_apt])[:, 1]
    man["h"] = h
    auc_h = auroc(is_h[train_h], h[train_h])
    probe_rows.append({"probe": "mostly-healthy camera vs other cameras (score h)", "eyes": "APTOS grade 0",
                       "classes": 2, "n": int(train_h.sum()), "auroc": auc_h, "chance": 0.5})
    probes = pd.DataFrame(probe_rows)
    probes.to_csv(out / "probes.csv", index=False, float_format="%.4f")
    lines += ["", "Probes trained and tested on **grade-0 eyes only** (no diabetic retinopathy at all), so disease "
              "cannot help them; out-of-fold (APTOS outer folds; grouped 5-fold for the dataset probe):", "",
              "| What is predicted | From | Eyes | Classes | Result | Chance |", "|---|---|---|---|---|---|",
              f"| Camera (photo size) | ResNet-50 features | APTOS grade 0 ({int(sel.sum()):,}) | {len(big)} | "
              f"balanced accuracy {ba_f:.1%} | {1 / len(big):.0%} |",
              f"| Camera (photo size) | quality columns only | APTOS grade 0 ({int(sel.sum()):,}) | {len(big)} | "
              f"balanced accuracy {ba_q:.1%} | {1 / len(big):.0%} |",
              f"| Mostly-healthy camera or not (score h) | ResNet-50 features | APTOS grade 0 ({int(train_h.sum()):,}) | 2 | "
              f"AUROC {auc_h:.3f} | 0.5 |",
              f"| Dataset (clinic) | ResNet-50 features | grade 0, all datasets ({int(sel_d.sum()):,}) | 3 | "
              f"balanced accuracy {ba_d:.1%} | 33% |", ""]

    # ============================ Part 2: who is missed ============================
    y = man["ref3"].to_numpy()
    grade = man["grade"].to_numpy()
    preds = {k: load_preds(res, k) for k in ("lr", "svm", "b2a")}
    pos_of = pd.Series(np.arange(len(man)), index=man["uid"].astype(str))

    def oof(key):
        d = preds[key]
        t = d[d["role"] == "test"]
        P = np.full((len(man), 3), np.nan)
        P[pos_of[t["uid"]].to_numpy()] = t[["p0", "p1", "p2"]].to_numpy()
        return P

    def external_mean(key):
        d = preds[key]
        e = d[d["role"] == "external"].groupby("uid")[["p0", "p1", "p2"]].mean()
        P = np.full((len(man), 3), np.nan)
        P[pos_of[e.index].to_numpy()] = e.to_numpy()
        return P

    # --- 2a: APTOS, per camera group ---
    rows = []
    apt_idx = np.flatnonzero(is_apt)
    cg = man["cam_group"].to_numpy()

    def group_metrics(idx, pred):
        out_ = {}
        for g in "HMS":
            s = idx[cg[idx] == g]
            ref, urg = s[y[s] >= 1], s[y[s] == 2]
            non = s[y[s] == 0]
            out_[g] = {"miss_referable": np.mean(pred[ref] == 0) if len(ref) else np.nan,
                       "urgent_detected": np.mean(pred[urg] == 2) if len(urg) else np.nan,
                       "false_referral": np.mean(pred[non] >= 1) if len(non) else np.nan,
                       "miss_grade2": np.mean(pred[s[grade[s] == 2]] == 0) if (grade[s] == 2).any() else np.nan,
                       "miss_grade34": np.mean(pred[s[grade[s] >= 3]] == 0) if (grade[s] >= 3).any() else np.nan}
        return out_

    boots = {}
    for key, name in (("lr", "Logistic regression"), ("svm", "SVM (RBF)"), ("b2a", "B2a Image size only")):
        P = oof(key)
        pred = np.full(len(man), -1)
        pred[apt_idx] = P[apt_idx].argmax(1)
        point = group_metrics(apt_idx, pred)
        bs = [group_metrics(apt_idx[b], pred) for b in group_boot(man["group_id"].to_numpy()[apt_idx],
                                                                    args.boot, args.seed)]
        boots[key] = bs
        for g in "HMS":
            for metric, v in point[g].items():
                vals = np.array([b[g][metric] for b in bs], float)
                lo, hi = np.nanpercentile(vals, [2.5, 97.5]) if not np.isnan(v) else (np.nan, np.nan)
                rows.append({"model": key, "name": name, "group": g, "metric": metric, "value": v, "lo": lo, "hi": hi})
    by_group = pd.DataFrame(rows)
    by_group.to_csv(out / "aptos_by_camera_group.csv", index=False, float_format="%.4f")
    # difference H - M in the miss rate of referable eyes (logistic regression), paired bootstrap
    d_hm = np.array([b["H"]["miss_referable"] - b["M"]["miss_referable"] for b in boots["lr"]])
    lr_pt = by_group[(by_group.model == "lr") & (by_group.metric == "miss_referable")].set_index("group")["value"]
    n_ref = {g: int(((cg == g) & (y >= 1)).sum()) for g in "HMS"}
    n_non = {g: int(((cg == g) & (y == 0)).sum()) for g in "HMS"}
    n_urg = {g: int(((cg == g) & (y == 2)).sum()) for g in "HMS"}
    # AUROC inside the 1050x1050 camera (the one H camera with referable eyes)
    P_lr = oof("lr")
    s1050 = is_apt & (man["camera"] == "1050x1050").to_numpy()
    auc1050 = auroc(y[s1050] >= 1, P_lr[s1050, 1:].sum(1))

    def cell(key, g, metric):
        r = by_group[(by_group.model == key) & (by_group.group == g) & (by_group.metric == metric)].iloc[0]
        return ci_text(r["value"], r["lo"], r["hi"])

    lines += ["## 2. Who is missed", "",
              "### 2a. APTOS, by camera group", "",
              "E1 out-of-fold predictions, most likely class (no tuned threshold, no conformal step). 95 % CIs "
              f"from {args.boot} group-bootstrap resamples. In %.", "",
              "| Model | Camera group | Referable eyes missed | Grade 2 missed | Grade 3–4 missed | Urgent eyes called urgent | Healthy eyes referred |",
              "|---|---|---|---|---|---|---|"]
    for key, name in (("lr", "Logistic regression"), ("svm", "SVM (RBF)"), ("b2a", "B2a Image size only")):
        for g in "HMS":
            lines.append(f"| {name} | {g} (referable {n_ref[g]}, urgent {n_urg[g]}, healthy {n_non[g]}) | "
                         f"{cell(key, g, 'miss_referable')} | {cell(key, g, 'miss_grade2')} | "
                         f"{cell(key, g, 'miss_grade34')} | {cell(key, g, 'urgent_detected')} | "
                         f"{cell(key, g, 'false_referral')} |")
    lines += ["", f"Logistic regression misses {100 * lr_pt['H']:.0f} % of the referable eyes photographed with the "
              f"mostly-healthy cameras, against {100 * lr_pt['M']:.0f} % on the mixed cameras "
              f"(difference {100 * (lr_pt['H'] - lr_pt['M']):+.0f} points, 95 % CI "
              f"{100 * np.percentile(d_hm, 2.5):+.0f} to {100 * np.percentile(d_hm, 97.5):+.0f}). Inside the "
              f"1050×1050 camera alone the referable AUROC is {auc1050:.3f}: the eyes are ranked well, but the "
              "decision is pulled towards \"healthy\" for that camera.", ""]

    # --- 2b: external sites, camera-style score h ---
    P_ext = external_mean("lr")
    P_all = np.where(is_apt[:, None], P_lr, P_ext)
    pred_all = np.where(np.isnan(P_all).any(1), -1, np.nan_to_num(P_all).argmax(1))
    # E0: standard conformal (LAC, 3 classes), coverage of every image in every fold, averaged
    d = preds["lr"]
    cover_sum = np.zeros(len(man))
    cover_n = np.zeros(len(man))
    for k, df in d.groupby("fold"):
        cal = df[df["role"] == "cal"]
        yc = y[pos_of[cal["uid"]].to_numpy()]
        s_cal = 1 - cal[["p0", "p1", "p2"]].to_numpy()[np.arange(len(cal)), yc]
        n = len(s_cal)
        q = np.sort(s_cal)[math.ceil((n + 1) * (1 - args.alpha)) - 1]
        ev = df[df["role"].isin(["external", "test"])]
        pos = pos_of[ev["uid"]].to_numpy()
        ok = (1 - ev[["p0", "p1", "p2"]].to_numpy())[np.arange(len(ev)), y[pos]] <= q
        cover_sum[pos] += ok
        cover_n[pos] += 1
    coverage = np.where(cover_n > 0, cover_sum / np.maximum(cover_n, 1), np.nan)

    ext_rows, h_rows = [], []
    for ds in ("aptos", "idrid", "messidor2"):
        s = (man["dataset"] == ds).to_numpy()
        hv = h[s]
        h_rows.append({"dataset": ds, "n": int(s.sum()), "h_median": np.median(hv),
                       "h_q25": np.percentile(hv, 25), "h_q75": np.percentile(hv, 75), "share_h_above_0.5": np.mean(hv > 0.5),
                       "sensitivity": np.mean(pred_all[s & (y >= 1)] >= 1), "specificity": np.mean(pred_all[s & (y == 0)] == 0),
                       "auroc_referable": auroc(y[s] >= 1, P_all[s, 1:].sum(1))})
        ref = np.flatnonzero(s & (y >= 1))
        cuts = np.percentile(h[ref], [100 / 3, 200 / 3])
        tert = np.digitize(h[ref], cuts)
        rho = spearmanr(h[ref], P_all[ref, 1:].sum(1)).statistic
        bs_rho = []
        for b in group_boot(man["group_id"].to_numpy()[ref], args.boot, args.seed):
            bs_rho.append(spearmanr(h[ref][b], P_all[ref][b, 1:].sum(1)).statistic)
        for t in range(3):
            r = ref[tert == t]
            r2 = r[grade[r] == 2]
            boot_miss = [np.mean(pred_all[r][b] == 0) for b in group_boot(man["group_id"].to_numpy()[r], args.boot,
                                                                          args.seed)]
            ext_rows.append({"dataset": ds, "h_tertile": t + 1, "h_range": f"{h[r].min():.2f}–{h[r].max():.2f}",
                             "referable_eyes": len(r), "missed": np.mean(pred_all[r] == 0),
                             "missed_lo": np.percentile(boot_miss, 2.5), "missed_hi": np.percentile(boot_miss, 97.5),
                             "missed_grade2": np.mean(pred_all[r2] == 0) if len(r2) else np.nan, "grade2_eyes": len(r2),
                             "mean_p_referable": P_all[r, 1:].sum(1).mean(),
                             "e0_coverage": np.nanmean(coverage[r]), "spearman_h_vs_p": rho,
                             "spearman_lo": np.percentile(bs_rho, 2.5), "spearman_hi": np.percentile(bs_rho, 97.5)})
    hsum = pd.DataFrame(h_rows)
    ext = pd.DataFrame(ext_rows)
    hsum.to_csv(out / "camera_style_by_dataset.csv", index=False, float_format="%.4f")
    ext.to_csv(out / "misses_by_camera_style.csv", index=False, float_format="%.4f")
    # the parallel: APTOS camera groups against the other clinics (logistic regression)
    par_rows = []
    for label, s in (("APTOS, H cameras", is_apt & (cg == "H")), ("APTOS, M cameras", is_apt & (cg == "M")),
                     ("APTOS, S cameras", is_apt & (cg == "S")),
                     ("IDRiD", (man["dataset"] == "idrid").to_numpy()),
                     ("Messidor-2", (man["dataset"] == "messidor2").to_numpy())):
        ref, non = s & (y >= 1), s & (y == 0)
        par_rows.append({"group": label, "photos": int(s.sum()), "h_median": np.median(h[s]),
                         "referable": int(ref.sum()), "mean_p_referable": P_all[ref, 1:].sum(1).mean(),
                         "missed": np.mean(pred_all[ref] == 0), "e0_coverage_referable": np.nanmean(coverage[ref]),
                         "healthy": int(non.sum()), "healthy_referred": np.mean(pred_all[non] >= 1) if non.any() else np.nan})
    par = pd.DataFrame(par_rows)
    par.to_csv(out / "aptos_groups_vs_clinics.csv", index=False, float_format="%.4f")

    lines += ["### 2b. Camera style at the other clinics", "",
              "h = probability that a photo comes from one of APTOS's mostly-healthy cameras, learnt from APTOS "
              "grade-0 eyes only (out-of-fold for APTOS). Predictions: logistic regression (APTOS: out-of-fold; "
              "IDRiD and Messidor-2: the 10 E1 fold models averaged), most likely class.", "",
              "| Dataset | Photos | Median h (IQR) | Photos with h > 0.5 | AUROC referable | Sensitivity | Specificity |",
              "|---|---|---|---|---|---|---|"]
    for _, r in hsum.iterrows():
        lines.append(f"| {DATASETS[r['dataset']]} | {r['n']:,} | {r['h_median']:.2f} ({r['h_q25']:.2f}–{r['h_q75']:.2f}) | "
                     f"{r['share_h_above_0.5']:.0%} | {r['auroc_referable']:.3f} | {r['sensitivity']:.0%} | "
                     f"{r['specificity']:.0%} |")
    lines += ["", "The same model on APTOS's camera groups and on the two other clinics. \"E0 coverage\": share of "
              "referable eyes whose true class is in the standard conformal set (90 % target, threshold from APTOS), "
              "mean over the 10 folds.", "",
              "| Photos from | Photos | Median h | Referable eyes | Mean P(referable) of referable eyes | Referable eyes missed | E0 coverage of referable eyes | Healthy eyes referred |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in par.iterrows():
        hr = "–" if np.isnan(r["healthy_referred"]) else f"{r['healthy_referred']:.0%}"
        lines.append(f"| {r['group']} | {r['photos']:,} | {r['h_median']:.2f} | {r['referable']:,} | "
                     f"{r['mean_p_referable']:.2f} | {r['missed']:.0%} | {r['e0_coverage_referable']:.0%} | {hr} |")
    lines += ["", "### 2c. Within each dataset", "",
              "Referable eyes only, split into thirds of h within each dataset. \"Missed\": called no referral.", "",
              "| Dataset | Third of h (range) | Referable eyes | Missed, % (95 % CI) | Grade-2 eyes missed | Mean P(referable) | E0 coverage |",
              "|---|---|---|---|---|---|---|"]
    for _, r in ext.iterrows():
        g2 = "–" if np.isnan(r["missed_grade2"]) else f"{100 * r['missed_grade2']:.0f} % of {int(r['grade2_eyes'])}"
        lines.append(f"| {DATASETS[r['dataset']]} | {int(r['h_tertile'])} ({r['h_range']}) | {int(r['referable_eyes'])} | "
                     f"{ci_text(r['missed'], r['missed_lo'], r['missed_hi'])} | {g2} | "
                     f"{r['mean_p_referable']:.2f} | {100 * r['e0_coverage']:.0f} % |")
    lines.append("")
    for ds in ("aptos", "idrid", "messidor2"):
        r = ext[ext["dataset"] == ds].iloc[0]
        lines.append(f"- {DATASETS[ds]}: Spearman correlation between h and P(referable) among referable eyes "
                     f"{r['spearman_h_vs_p']:+.2f} (95 % CI {r['spearman_lo']:+.2f} to {r['spearman_hi']:+.2f}).")
    lines += ["", "Within a dataset this test is weak: Messidor-2 has almost no spread in h (nearly every photo looks "
              "like a mostly-healthy APTOS camera), and IDRiD uses a single camera. Part 3 therefore tests the "
              "mechanism by changing the training data instead.", ""]

    # ============================ Part 3: preliminary fixes ============================
    yy = man["ref3"].to_numpy()
    m_idx = np.flatnonzero(is_apt & (man["cam_group"] == "M").to_numpy())
    all_idx = np.flatnonzero(is_apt)
    tests = {ds: np.flatnonzero((man["dataset"] == ds).to_numpy()) for ds in ("idrid", "messidor2")}
    counts = np.bincount(yy[m_idx], minlength=3)
    rng = np.random.default_rng(args.seed)

    def decorrelating_weights(idx):
        """w = P(y) / P(y | camera group): in the weighted data the label no longer depends on the camera."""
        g, yi = cg[idx], yy[idx]
        p_y = np.bincount(yi, minlength=3) / len(idx)
        w = np.ones(len(idx))
        for grp in "HMS":
            sg = g == grp
            p_yg = np.bincount(yi[sg], minlength=3) / max(sg.sum(), 1)
            for c in range(3):
                sc = sg & (yi == c)
                if sc.any():
                    w[sc] = p_y[c] / p_yg[c]
        return w / w.mean()

    def fit(idx, weighted=False):
        model = logreg(C=C_FIX, balanced=False)
        if weighted:
            return model.fit(X[idx], yy[idx], logisticregression__sample_weight=decorrelating_weights(idx))
        return model.fit(X[idx], yy[idx])

    def evaluate(model, label, n_train):
        r = {"training": label, "n_train": n_train}
        for ds, te in tests.items():
            P = model.predict_proba(X[te])
            pr = P.argmax(1)
            r[f"{ds}_auroc_referable"] = auroc(yy[te] >= 1, P[:, 1:].sum(1))
            r[f"{ds}_auroc_urgent"] = auroc(yy[te] == 2, P[:, 2])
            r[f"{ds}_sensitivity"] = np.mean(pr[yy[te] >= 1] >= 1)
            r[f"{ds}_specificity"] = np.mean(pr[yy[te] == 0] == 0)
        return r

    fix = [evaluate(fit(all_idx), "all of APTOS", len(all_idx)),
           evaluate(fit(all_idx, weighted=True), "all of APTOS, camera-decorrelated weights", len(all_idx)),
           evaluate(fit(m_idx), "mixed cameras only (M)", len(m_idx))]
    for i in range(args.repeats):
        draw = np.concatenate([rng.choice(all_idx[yy[all_idx] == c], counts[c], replace=False) for c in range(3)])
        fix.append(evaluate(fit(draw), f"random APTOS subset {i + 1}", len(draw)))
    fix = pd.DataFrame(fix)
    fix.to_csv(out / "fix_preview.csv", index=False, float_format="%.4f")
    rnd = fix[fix["training"].str.startswith("random")]

    # APTOS 10-fold cross-validation (E1 outer folds), unweighted against camera-decorrelated weights
    cam_codes = pd.factorize(man["camera"])[0]
    cv_rows = []
    for weighted in (False, True):
        P = np.full((len(man), 3), np.nan)
        for k in range(10):
            tr, te = np.flatnonzero(is_apt & (fold != k)), np.flatnonzero(is_apt & (fold == k))
            P[te] = fit(tr, weighted).predict_proba(X[te])
        a_idx = all_idx
        pr = P[a_idx].argmax(1)
        ya, ca, ga = yy[a_idx], cam_codes[a_idx], cg[a_idx]
        num = den = 0.0
        for c in np.unique(ca):
            sc = ca == c
            n1 = int((ya[sc] >= 1).sum())
            n0 = int(sc.sum()) - n1
            if n1 and n0:
                num += auroc(ya[sc] >= 1, P[a_idx][sc, 1:].sum(1)) * n1 * n0
                den += n1 * n0
        row = {"training": "camera-decorrelated weights" if weighted else "unweighted",
               "auroc_referable": auroc(ya >= 1, P[a_idx, 1:].sum(1)), "auroc_referable_same_camera": num / den,
               "auroc_urgent": auroc(ya == 2, P[a_idx, 2])}
        for grp in "HMS":
            row[f"missed_{grp}"] = np.mean(pr[(ga == grp) & (ya >= 1)] == 0)
            row[f"healthy_referred_{grp}"] = np.mean(pr[(ga == grp) & (ya == 0)] >= 1)
        cv_rows.append(row)
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(out / "fix_aptos_cv.csv", index=False, float_format="%.4f")

    lines += ["## 3. First look at two fixes", "",
              f"Same model everywhere: standardised ResNet-50 features, logistic regression, C = {C_FIX} (one fixed "
              "setting, not tuned per fold).", "",
              "- **Camera-decorrelated weights:** all APTOS photos, each weighted by P(class) / P(class | camera group), "
              "so that in the weighted training data the camera says nothing about the class. The overall class mix "
              "is unchanged.",
              f"- **Mixed cameras only:** the {len(m_idx):,} photos from the mixed cameras, where the camera says little "
              "about the class.",
              f"- **Random subsets** (control for the previous row): {args.repeats} random sets of APTOS photos with the same "
              f"size and class mix ({counts[0]} / {counts[1]} / {counts[2]}) from all cameras; mean (lowest–highest).", "",
              "### 3a. Tested on the other clinics", "",
              "| Training set | Photos | IDRiD AUROC referable | IDRiD sensitivity | Messidor-2 AUROC referable | Messidor-2 AUROC urgent | Messidor-2 sensitivity | Messidor-2 specificity |",
              "|---|---|---|---|---|---|---|---|"]

    def fmt_row(label, r):
        return (f"| {label} | {int(r['n_train']):,} | {r['idrid_auroc_referable']:.3f} | {r['idrid_sensitivity']:.0%} | "
                f"{r['messidor2_auroc_referable']:.3f} | {r['messidor2_auroc_urgent']:.3f} | "
                f"{r['messidor2_sensitivity']:.0%} | {r['messidor2_specificity']:.0%} |")
    lines.append(fmt_row("All of APTOS", fix.iloc[0]))
    lines.append(fmt_row("All of APTOS, camera-decorrelated weights", fix.iloc[1]))
    lines.append(fmt_row("Mixed cameras only", fix.iloc[2]))
    cols = ["idrid_auroc_referable", "idrid_sensitivity", "messidor2_auroc_referable", "messidor2_auroc_urgent",
            "messidor2_sensitivity", "messidor2_specificity"]
    mean, lo, hi = rnd[cols].mean(), rnd[cols].min(), rnd[cols].max()
    lines.append(f"| Random APTOS subsets, same size and class mix as the mixed cameras | {int(rnd['n_train'].iloc[0]):,} | "
                 f"{mean.iloc[0]:.3f} ({lo.iloc[0]:.3f}–{hi.iloc[0]:.3f}) | {mean.iloc[1]:.0%} ({lo.iloc[1]:.0%}–{hi.iloc[1]:.0%}) | "
                 f"{mean.iloc[2]:.3f} ({lo.iloc[2]:.3f}–{hi.iloc[2]:.3f}) | {mean.iloc[3]:.3f} ({lo.iloc[3]:.3f}–{hi.iloc[3]:.3f}) | "
                 f"{mean.iloc[4]:.0%} ({lo.iloc[4]:.0%}–{hi.iloc[4]:.0%}) | {mean.iloc[5]:.0%} ({lo.iloc[5]:.0%}–{hi.iloc[5]:.0%}) |")
    lines += ["", "Sensitivity and specificity use the most likely class, so they also depend on the class mix of the "
              "training set: compare the mixed-camera row with the random subsets, and the weighted row with all of APTOS.", "",
              "### 3b. APTOS, 10-fold cross-validation (E1 outer folds)", "",
              "| Training | AUROC referable | Same-camera AUROC referable | AUROC urgent | Referable missed: H / M / S cameras | Healthy referred: H / M / S cameras |",
              "|---|---|---|---|---|---|"]
    for _, r in cv.iterrows():
        lines.append(f"| {r['training']} | {r['auroc_referable']:.3f} | {r['auroc_referable_same_camera']:.3f} | "
                     f"{r['auroc_urgent']:.3f} | {r['missed_H']:.0%} / {r['missed_M']:.0%} / {r['missed_S']:.0%} | "
                     f"{r['healthy_referred_H']:.0%} / {r['healthy_referred_M']:.0%} / {r['healthy_referred_S']:.0%} |")
    lines += ["", f"Made {time.strftime('%Y-%m-%d %H:%M')} by camera_shortcut.py ({time.time() - t0:.0f} s)."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")

    plot_aptos(by_group, n_ref, n_non, out / "aptos_misses.png")
    plot_external(man, ext, par, out / "external_h.png")
    plot_fix(fix, cv, out / "fix_preview.png")
    print(f"wrote {out / 'summary.md'} ({time.time() - t0:.0f} s)")


# ------------------------------------------------------------------------------------------------
# figures
# ------------------------------------------------------------------------------------------------

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8, length=0)


def plot_aptos(by_group, n_ref, n_non, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), dpi=200, sharey=True)
    models = [("lr", "Logistic regression", BLUE), ("b2a", "B2a Image size only", ORANGE)]
    panels = [("miss_referable", "Referable eyes called “no referral”", n_ref),
              ("false_referral", "Healthy eyes referred", n_non)]
    for ax, (metric, title, counts) in zip(axes, panels):
        for j, (key, name, col) in enumerate(models):
            sub = by_group[(by_group.model == key) & (by_group.metric == metric)].set_index("group").loc[list("HMS")]
            x = np.arange(3) + (j - 0.5) * 0.3
            ax.bar(x, 100 * sub["value"], width=0.28, color=col, label=name)
            ax.vlines(x, 100 * sub["lo"], 100 * sub["hi"], color=INK2, lw=1)
            for xi, v, top in zip(x, sub["value"], sub["hi"]):
                ax.text(xi, 100 * max(v, top) + 2, f"{100 * v:.0f}", ha="center", fontsize=7.5, color=INK2)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{g}\n(n = {counts[g]})" for g in "HMS"], fontsize=8, color=INK)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_ylim(0, 110)
        _style(ax)
    axes[0].set_ylabel("% of eyes (95 % CI)", fontsize=8.5, color=INK2)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("APTOS by camera group  (H: mostly-healthy cameras, M: mixed, S: mostly-sick)", fontsize=10.5,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_external(man, ext, par, path):
    """Left: camera style (h) of APTOS's camera groups and of the other clinics. Right: referable eyes missed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = [("aptos", "H", "APTOS, H cameras"), ("aptos", "M", "APTOS, M cameras"), ("aptos", "S", "APTOS, S cameras"),
              ("idrid", None, "IDRiD"), ("messidor2", None, "Messidor-2")]
    colors = [BLUE, BLUE, BLUE, ORANGE, AQUA]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=200, sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
    data = []
    for ds, g, _ in groups:
        sel = (man["dataset"] == ds) & ((man["cam_group"] == g) if g else True)
        data.append(man.loc[sel, "h"].to_numpy())
    ax = axes[0]
    kw = dict(widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.4),
              whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    try:
        bp = ax.boxplot(data, orientation="horizontal", **kw)
    except TypeError:                                  # matplotlib < 3.10
        bp = ax.boxplot(data, vert=False, **kw)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.55)
        patch.set_edgecolor(INK2)
    ax.set_yticks(range(1, len(groups) + 1))
    ax.set_yticklabels([g[2] for g in groups], fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("h = P(photo looks like a mostly-healthy APTOS camera)", fontsize=8.5, color=INK2)
    ax.set_title("Camera style", fontsize=10, color=INK, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax = axes[1]
    missed = par.set_index("group").loc[[g[2] for g in groups], "missed"].to_numpy()
    yy = np.arange(1, len(groups) + 1)
    ax.barh(yy, 100 * missed, height=0.55, color=colors, alpha=0.85)
    for yi, v in zip(yy, missed):
        ax.text(100 * v + 1.5, yi, f"{100 * v:.0f} %", va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Referable eyes called “no referral” (%)", fontsize=8.5, color=INK2)
    ax.set_title("Referable eyes missed (logistic regression)", fontsize=10, color=INK, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_fix(fix, cv, path):
    """Camera-decorrelated weights against plain training: decisions per camera group, and the other clinics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), dpi=200, gridspec_kw={"width_ratios": [1, 1, 0.85]})
    rows = [("unweighted", "Plain training", BLUE), ("camera-decorrelated weights", "Camera-decorrelated weights", AQUA)]
    for ax, (prefix, title) in zip(axes[:2], (("missed_", "APTOS: referable eyes missed"),
                                              ("healthy_referred_", "APTOS: healthy eyes referred"))):
        for j, (key, name, col) in enumerate(rows):
            r = cv[cv["training"] == key].iloc[0]
            vals = np.array([r[prefix + g] for g in "HMS"])
            x = np.arange(3) + (j - 0.5) * 0.32
            ax.bar(x, 100 * vals, width=0.3, color=col, label=name)
            for xi, v in zip(x, vals):
                ax.text(xi, 100 * v + 1.5, f"{100 * v:.0f}", ha="center", fontsize=7.5, color=INK2)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["H cameras", "M cameras", "S cameras"], fontsize=8, color=INK)
        ax.set_ylim(0, 85)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        _style(ax)
    axes[0].set_ylabel("% of eyes (10-fold cross-validation)", fontsize=8.5, color=INK2)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    ax = axes[2]
    for j, (row_i, name, col) in enumerate(((0, "Plain training", BLUE), (1, "Camera-decorrelated weights", AQUA))):
        r = fix.iloc[row_i]
        vals = [r["idrid_sensitivity"], r["messidor2_sensitivity"]]
        x = np.arange(2) + (j - 0.5) * 0.32
        ax.bar(x, [100 * v for v in vals], width=0.3, color=col)
        for xi, v in zip(x, vals):
            ax.text(xi, 100 * v + 1.5, f"{100 * v:.0f}", ha="center", fontsize=7.5, color=INK2)
    a0, a1 = fix.iloc[0], fix.iloc[1]
    ax.set_xticks(range(2))
    ax.set_xticklabels([f"IDRiD\nAUROC {a0['idrid_auroc_referable']:.3f} → {a1['idrid_auroc_referable']:.3f}",
                        f"Messidor-2\nAUROC {a0['messidor2_auroc_referable']:.3f} → {a1['messidor2_auroc_referable']:.3f}"],
                       fontsize=7.5, color=INK)
    ax.set_ylim(0, 85)
    ax.set_ylabel("Sensitivity, referable (%)", fontsize=8.5, color=INK2)
    ax.set_title("Other clinics", fontsize=10, color=INK, loc="left")
    _style(ax)
    fig.suptitle("Fix 1: camera-decorrelated weights (logistic regression, one fixed setting)", fontsize=10.5,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
