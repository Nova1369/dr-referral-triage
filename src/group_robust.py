#!/usr/bin/env python3
"""
group_robust.py - additional work: does the referral guarantee hold for every camera group, and which
method restores it? Our camera-decorrelated weights against the standard methods for spurious correlations and
against per-subgroup conformal calibration, under the E1 protocol.

Research question. C1 guarantees each CLASS 90 % coverage on average over cameras. APTOS's cameras saw very
different patients (camera group H, mostly healthy: <= 10 % referable; S, mostly sick: >= 80 %; M, mixed), and on
H cameras C1 sends home 47 % of referable eyes. Which training or calibration method brings every camera group
under the 10 % target, at what cost, and does it carry over to new clinics?

Methods. Logistic regression on standardised ResNet-50 features, C = 0.003 (E1's most frequent choice) for every
method, refitted in each of E1's 10 outer folds on E1's training photos; C1 fitted on the fold's calibration photos.
Training-time camera groups are recomputed from each fold's TRAINING photos only.
  erm          plain training
  decorr       camera-decorrelated weights w = P(y) / P(y | g)                (ours; camera_fixes.py)
  balanced     group-balanced weights w ~ 1 / n(y, g): every (class, camera group) cell weighs the same
               (the reweighting baseline of Sagawa et al. 2020; Idrissi et al. 2022)
  dfr          group-balanced subsampling of the training photos, logistic regression averaged over 10
               subsamples, C = 0.03 (deep feature reweighting, Kirichenko et al. 2023)
  groupdro     group DRO on the last layer (Sagawa et al. 2020): multiplicative weights over the 9 (class,
               camera group) cells, step 0.5, 20 rounds, coefficients averaged over the last 10
  jtt5, jtt20  just train twice (Liu et al. 2021), no group labels: photos the plain model gets wrong in 5
               inner folds are upweighted 5 or 20 times
  mondrian_g   plain model; C1 with one threshold per class AND camera group, from the fold's calibration photos
  mondrian_gx  plain model; one threshold per class and camera group from the out-of-fold scores of the other
               nine folds (cross-fitted, about 6 times more calibration photos per cell)
Read from camera_fixes.py (C tuned in inner folds): the two LEACE erasure variants, for reference.

Evaluation. APTOS out-of-fold (every photo decided once): AUROCs, C1's targets overall and per camera group
(analysis groups fixed: camera_groups() on all APTOS), worst-group rates with Wilson intervals, exact McNemar tests
on the H-camera referable eyes. New clinics: each training method refitted on all of APTOS; AUROC, APTOS
thresholds unchanged (B6), and C2 from k labelled local photos (200 draws, the same draws for every method).

Outputs (in --out, default ../results/group_robust): summary.md, metrics.csv, by_group.csv, mcnemar_h.csv,
mcnemar_healthy.csv (the cost in healthy eyes, tested against ours), clinics.csv, oof_<method>.csv.gz (every
method's out-of-fold probabilities and decisions), group_robust.png

Usage:
  python group_robust.py --data ../data_512 --results ../results
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from conformal import HOME, ROUTINE, URGENT, UNCERTAIN, classwise_thresholds, decide_from_sets
from c1_referral import camera_groups
from dr_preprocess import assign_folds
from e2_new_clinics import Pool

C_FIX, C_DFR = 0.003, 0.03
TRAINED = ["erm", "decorr", "balanced", "dfr", "groupdro", "jtt5", "jtt20"]
CALIBRATED = ["mondrian_g", "mondrian_gx"]
FROM_FILES = {"leace_healthy": "erasure", "leace_grade": "erasure_grade"}
NAMES = {"erm": "Plain training (ERM)", "decorr": "Camera-decorrelated weights (ours)",
         "balanced": "Group-balanced weights", "dfr": "DFR: group-balanced subsampling",
         "groupdro": "Group DRO", "jtt5": "JTT, errors x5", "jtt20": "JTT, errors x20",
         "mondrian_g": "Per-group thresholds (fold calibration)", "mondrian_gx": "Per-group thresholds (cross-fitted)",
         "leace_healthy": "LEACE fitted on healthy eyes", "leace_grade": "LEACE within each grade"}
GROUPS = ("H", "M", "S")
K_CLINIC = {"idrid": [50, 100], "messidor2": [200, 300]}


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------

def fit(Z, y, w=None, C=C_FIX, init=None):
    m = LogisticRegression(C=C, max_iter=5000, warm_start=init is not None)
    if init is not None:
        m.coef_, m.intercept_, m.classes_ = init.coef_.copy(), init.intercept_.copy(), init.classes_
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.fit(Z, y, sample_weight=w)


def averaged(models_or_params, classes, n_features):
    coef = np.mean([c for c, _ in models_or_params], axis=0)
    inter = np.mean([i for _, i in models_or_params], axis=0)
    m = LogisticRegression()
    m.coef_, m.intercept_, m.classes_, m.n_features_in_ = coef, inter, classes, n_features
    return m


def proba(m, Z):
    P = np.zeros((len(Z), 3))
    P[:, np.asarray(m.classes_, int)] = m.predict_proba(Z)
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


def wilson(k, n, z=1.959964):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def camera_groups_from(man, rows):
    """Camera groups from the referable shares of the given (training) rows only; NaN for unseen cameras."""
    size = (man["orig_w"].astype(str) + "x" + man["orig_h"].astype(str))
    share = man["referable"].astype(float).iloc[rows].groupby(size.iloc[rows].to_numpy()).mean()
    grp = pd.Series(np.where(share <= 0.10, "H", np.where(share >= 0.80, "S", "M")), index=share.index)
    return size.map(grp).to_numpy()


def cells(y, g):
    return np.array([f"{a}{b}" for a, b in zip(y, g)])


# --------------------------------------------------------------------------------------------
# the training methods
# --------------------------------------------------------------------------------------------

def train(method, Z, y, g, man_rows, seed):
    """Returns a fitted model (anything with predict_proba via proba())."""
    if method == "erm":
        return fit(Z, y)
    if method == "decorr":
        p_y = np.bincount(y, minlength=3) / len(y)
        w = np.ones(len(y))
        for grp in np.unique(g):
            s = g == grp
            p_yg = np.bincount(y[s], minlength=3) / s.sum()
            for c in range(3):
                w[s & (y == c)] = p_y[c] / p_yg[c] if p_yg[c] > 0 else 0.0
        return fit(Z, y, w / w.mean())
    cell = cells(y, g)
    if method == "balanced":
        _, inv, cnt = np.unique(cell, return_inverse=True, return_counts=True)
        w = 1.0 / cnt[inv]
        return fit(Z, y, w / w.mean())
    if method == "dfr":
        rng = np.random.default_rng(seed)
        uniq = np.unique(cell)
        m = min(int(np.sum(cell == c)) for c in uniq)
        params = []
        for _ in range(10):
            sel = np.concatenate([rng.choice(np.flatnonzero(cell == c), m, replace=False) for c in uniq])
            mm = fit(Z[sel], y[sel], C=C_DFR)
            params.append((mm.coef_, mm.intercept_))
        return averaged(params, np.arange(3), Z.shape[1])
    if method == "groupdro":
        uniq, ci, cnt = np.unique(cell, return_inverse=True, return_counts=True)
        q = np.ones(len(uniq)) / len(uniq)
        model, params = None, []
        for t in range(20):
            w = q[ci] / cnt[ci]
            model = fit(Z, y, w / w.mean(), init=model)
            P = proba(model, Z)
            ll = -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None))
            L = np.bincount(ci, weights=ll) / cnt
            q = q * np.exp(0.5 * L)
            q /= q.sum()
            if t >= 10:
                params.append((model.coef_.copy(), model.intercept_.copy()))
        return averaged(params, model.classes_, Z.shape[1])
    if method.startswith("jtt"):
        lam = float(method[3:])
        inner = assign_folds(man_rows.reset_index(drop=True), 5, seed)[0].to_numpy()
        oof = np.zeros((len(y), 3))
        for j in range(5):
            tr, va = inner != j, inner == j
            oof[va] = proba(fit(Z[tr], y[tr]), Z[va])
        err = oof.argmax(1) != y
        w = np.where(err, lam, 1.0)
        return fit(Z, y, w / w.mean())
    raise ValueError(method)


def fold_task(k, method, X, y, man, fit_idx, other_idx, seed):
    g_tr = camera_groups_from(man, fit_idx)[fit_idx]
    sc = StandardScaler().fit(X[fit_idx])
    m = train(method, sc.transform(X[fit_idx]), y[fit_idx], g_tr, man.iloc[fit_idx], seed + 1000 + k)
    return k, method, proba(m, sc.transform(X[other_idx])), g_tr


def all_aptos_task(method, X, y, man, apt, seed):
    g = camera_groups_from(man, apt)[apt]
    sc = StandardScaler().fit(X[apt])
    m = train(method, sc.transform(X[apt]), y[apt], g, man.iloc[apt], seed + 5000)
    return method, proba(m, sc.transform(X))


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Additional work: camera-group guarantees against strong baselines.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", default="../data_512")
    ap.add_argument("--features", default="features/resnet50_224.npz")
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default="../results/group_robust")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    t0 = time.time()
    out, R = Path(args.out), Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    a = args.alpha

    man = pd.read_csv(Path(args.data) / "manifest.csv")
    with np.load(Path(args.data) / args.features) as z:
        X, uid = z["X"], z["uid"]
    assert (uid == man["uid"].astype(str).to_numpy()).all()
    y = man["ref3"].to_numpy(int)
    g_all = camera_groups(man).fillna("").to_numpy()            # fixed analysis groups (reporting only)
    camera = (man["orig_w"].astype(str) + "x" + man["orig_h"].astype(str)).to_numpy()
    row = pd.Series(np.arange(len(man)), index=man["uid"])
    apt = np.flatnonzero((man["dataset"] == "aptos").to_numpy())

    folds = {}
    for k in range(10):
        f = pd.read_csv(R / "e1" / "lr" / f"fold{k}.csv.gz")
        folds[k] = {r: row[f.loc[f["role"] == r, "uid"]].to_numpy() for r in ("test", "cal")}
        folds[k]["order"] = row[f["uid"]].to_numpy()

    # ---- training methods in every fold ----
    tasks = []
    for k, f in folds.items():
        fit_idx = np.setdiff1d(apt, np.concatenate([f["test"], f["cal"]]))
        for method in TRAINED:
            tasks.append(delayed(fold_task)(k, method, X, y, man, fit_idx, f["order"], args.seed))
    res = Parallel(n_jobs=args.jobs, verbose=5)(tasks)
    P = {m: {} for m in TRAINED}
    group_mismatch = 0
    for k, method, Pk, g_tr in res:
        P[method][k] = Pk
        if method == "erm":
            fit_idx = np.setdiff1d(apt, np.concatenate([folds[k]["test"], folds[k]["cal"]]))
            group_mismatch += int(np.sum(g_tr != g_all[fit_idx]))
    print(f"fold models done ({time.time() - t0:.0f} s); training photos whose fold group differs from the "
          f"analysis group: {group_mismatch}", flush=True)
    for name, folder in FROM_FILES.items():
        P[name] = {}
        for k in folds:
            f = pd.read_csv(R / "camera_fixes" / folder / f"fold{k}.csv.gz")
            assert (row[f["uid"]].to_numpy() == folds[k]["order"]).all()
            P[name][k] = f[["p0", "p1", "p2"]].to_numpy()

    # ---- decisions: C1 per fold; per-group thresholds for the calibration baselines ----
    oof = {}
    dec = {}
    for method in TRAINED + list(FROM_FILES):
        Po, d = np.full((len(man), 3), np.nan), np.full(len(man), -1)
        for k, f in folds.items():
            pos = {u: i for i, u in enumerate(f["order"])}
            cal = np.array([pos[i] for i in f["cal"]])
            te = np.array([pos[i] for i in f["test"]])
            q = classwise_thresholds(P[method][k][cal], y[f["cal"]], a)
            Po[f["test"]] = P[method][k][te]
            d[f["test"]] = decide_from_sets((1 - P[method][k][te]) <= q[None, :])
        oof[method], dec[method] = Po, d
    # per-group thresholds (plain model)
    for method in CALIBRATED:
        d = np.full(len(man), -1)
        for k, f in folds.items():
            pos = {u: i for i, u in enumerate(f["order"])}
            te = np.array([pos[i] for i in f["test"]])
            fit_idx = np.setdiff1d(apt, np.concatenate([f["test"], f["cal"]]))
            g_fold = camera_groups_from(man, fit_idx)                # groups known at training time
            if method == "mondrian_g":
                cal = np.array([pos[i] for i in f["cal"]])
                Pc, yc, gc = P["erm"][k][cal], y[f["cal"]], g_fold[f["cal"]]
            else:                                                     # the other folds' out-of-fold scores
                others = np.concatenate([folds[j]["test"] for j in folds if j != k])
                Pc, yc, gc = oof["erm"][others], y[others], g_fold[others]
            Pt, gt = P["erm"][k][te], g_fold[f["test"]]
            dd = np.full(len(te), -1)
            for grp in GROUPS:
                s, st = gc == grp, gt == grp
                if st.any():
                    q = classwise_thresholds(Pc[s], yc[s], a)
                    dd[st] = decide_from_sets((1 - Pt[st]) <= q[None, :])
            rest = dd < 0                                             # a camera unseen in training: plain C1
            if rest.any():
                cal = np.array([pos[i] for i in f["cal"]])
                q = classwise_thresholds(P["erm"][k][cal], y[f["cal"]], a)
                dd[rest] = decide_from_sets((1 - Pt[rest]) <= q[None, :])
            d[f["test"]] = dd
        oof[method], dec[method] = oof["erm"], d
    print(f"decisions done ({time.time() - t0:.0f} s)", flush=True)

    # ---- APTOS metrics ----
    METHODS = TRAINED + CALIBRATED + list(FROM_FILES)
    ya, ga, ca = y[apt], g_all[apt], camera[apt]
    rows, grows = [], []
    for method in METHODS:
        Pa, da = oof[method][apt], dec[method][apt]
        ref, urg, non = ya >= 1, ya == 2, ya == 0
        r = {"method": method, "auroc_ref": auroc(ref, Pa[:, 1:].sum(1)), "auroc_urg": auroc(urg, Pa[:, 2]),
             "same_camera_auroc_ref": same_camera_auroc(ref, Pa[:, 1:].sum(1), ca),
             "ref_home": np.mean(da[ref] == HOME), "urgent_urgent": np.mean(da[urg] == URGENT),
             "healthy_home": np.mean(da[non] == HOME), "uncertain": np.mean(da == UNCERTAIN),
             "referred": np.mean((da == ROUTINE) | (da == URGENT))}
        worst_home, worst_urg = 0.0, 1.0
        for grp in GROUPS:
            s = ga == grp
            n_r, k_r = int((s & ref).sum()), int((s & ref & (da == HOME)).sum())
            n_u, k_u = int((s & urg).sum()), int((s & urg & (da == URGENT)).sum())
            n_h, k_h = int((s & non).sum()), int((s & non & (da == HOME)).sum())
            lo, hi = wilson(k_r, n_r)
            grows.append({"method": method, "group": grp, "referable": n_r, "ref_home_n": k_r, "ref_home": k_r / n_r,
                          "ref_home_lo": lo, "ref_home_hi": hi, "urgent": n_u, "urgent_urgent": k_u / n_u if n_u else np.nan,
                          "healthy": n_h, "healthy_home": k_h / n_h if n_h else np.nan})
            worst_home, worst_urg = max(worst_home, k_r / n_r), min(worst_urg, k_u / n_u if n_u else 1.0)
        r["worst_group_ref_home"], r["worst_group_urgent_urgent"] = worst_home, worst_urg
        rows.append(r)
    met = pd.DataFrame(rows)
    grp_df = pd.DataFrame(grows)
    # exact McNemar on the H-camera referable eyes (sent home or not), against ERM and against ours
    hr = (ga == "H") & (ya >= 1)
    tests = []
    for method in METHODS:
        for base in ("erm", "decorr"):
            if method == base:
                continue
            a_home, b_home = dec[method][apt][hr] == HOME, dec[base][apt][hr] == HOME
            n01, n10 = int(np.sum(~a_home & b_home)), int(np.sum(a_home & ~b_home))
            p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            tests.append({"method": method, "against": base, "fixed": n01, "broken": n10, "p": p})
    tests = pd.DataFrame(tests)
    # the cost, tested: exact McNemar on every healthy APTOS eye (sent home or not), each method against ours
    he = ya == 0
    htests = []
    for method in METHODS:
        if method == "decorr":
            continue
        m_home, o_home = dec[method][apt][he] == HOME, dec["decorr"][apt][he] == HOME
        only_m, only_o = int(np.sum(m_home & ~o_home)), int(np.sum(~m_home & o_home))
        p = binomtest(only_m, only_m + only_o, 0.5).pvalue if only_m + only_o else 1.0
        htests.append({"method": method, "healthy": int(he.sum()), "home_only_method": only_m,
                       "home_only_ours": only_o, "p": p})
    htests = pd.DataFrame(htests)
    htests.to_csv(out / "mcnemar_healthy.csv", index=False, float_format="%.5g")
    met.to_csv(out / "metrics.csv", index=False, float_format="%.5f")
    grp_df.to_csv(out / "by_group.csv", index=False, float_format="%.5f")
    tests.to_csv(out / "mcnemar_h.csv", index=False, float_format="%.5g")
    for method in METHODS:
        pd.DataFrame({"uid": man["uid"].to_numpy()[apt], "p0": oof[method][apt, 0], "p1": oof[method][apt, 1],
                      "p2": oof[method][apt, 2], "decision": dec[method][apt]}).to_csv(
            out / f"oof_{method}.csv.gz", index=False, float_format="%.6g")

    # ---- new clinics ----
    res = Parallel(n_jobs=args.jobs, verbose=0)(delayed(all_aptos_task)(m, X, y, man, apt, args.seed) for m in TRAINED)
    P_all = dict(res)
    group = pd.factorize(man["dataset"] + "/" + man["group_id"].astype(str))[0]
    rng = np.random.default_rng(args.seed)
    pools = {ds: Pool(np.flatnonzero((man["dataset"] == ds).to_numpy()), np.flatnonzero((man["dataset"] == ds).to_numpy()),
                      group) for ds in K_CLINIC}
    draws = {(ds, k, b): pools[ds].draw(k, rng) for ds in K_CLINIC for k in K_CLINIC[ds] for b in range(args.draws)}
    crow = []
    for method in TRAINED:
        Pm = P_all[method]
        q6 = classwise_thresholds(oof[method][apt], y[apt], a)          # B6: APTOS thresholds (out-of-fold)
        for ds in K_CLINIC:
            s = np.flatnonzero((man["dataset"] == ds).to_numpy())
            d6 = decide_from_sets((1 - Pm[s]) <= q6[None, :])
            ys = y[s]
            crow.append({"method": method, "dataset": ds, "rule": "B6", "k": 0,
                         "auroc_ref": auroc(ys >= 1, Pm[s, 1:].sum(1)), "auroc_urg": auroc(ys == 2, Pm[s, 2]),
                         "ref_home": np.mean(d6[ys >= 1] == HOME), "urgent_urgent": np.mean(d6[ys == 2] == URGENT),
                         "healthy_home": np.mean(d6[ys == 0] == HOME)})
            for k in K_CLINIC[ds]:
                ms = []
                for b in range(args.draws):
                    cal, test = draws[(ds, k, b)]
                    q = classwise_thresholds(Pm[cal], y[cal], a)
                    d = decide_from_sets((1 - Pm[test]) <= q[None, :])
                    yt = y[test]
                    ms.append((np.mean(d[yt >= 1] == HOME), np.mean(d[yt == 2] == URGENT), np.mean(d[yt == 0] == HOME),
                               float((np.mean(d[yt >= 1] == HOME) <= a) & (np.mean(d[yt == 2] == URGENT) >= 1 - a))))
                ms = np.array(ms)
                crow.append({"method": method, "dataset": ds, "rule": "C2", "k": k, "ref_home": ms[:, 0].mean(),
                             "urgent_urgent": ms[:, 1].mean(), "healthy_home": ms[:, 2].mean(), "targets_met": ms[:, 3].mean()})
    clin = pd.DataFrame(crow)
    clin.to_csv(out / "clinics.csv", index=False, float_format="%.5f")
    print(f"new clinics done ({time.time() - t0:.0f} s)", flush=True)

    write_summary(out, args, met, grp_df, tests, clin, group_mismatch, time.time() - t0, htests)
    plot(met, grp_df, out / "group_robust.png")
    print(f"wrote {out / 'summary.md'} ({time.time() - t0:.0f} s)")


# --------------------------------------------------------------------------------------------
# summary and figure
# --------------------------------------------------------------------------------------------

def write_summary(out, args, met, grp_df, tests, clin, group_mismatch, elapsed, htests=None):
    M = met.set_index("method")
    G = grp_df.set_index(["method", "group"])
    order = list(M.index)
    L = ["# Additional work: camera-group guarantees against strong baselines", "",
         "Logistic regression on standardised ResNet-50 features (C = 0.003 for every trained method; DFR 0.03), "
         "refitted in each of E1's 10 outer folds; C1 (α = 0.1) on each fold's calibration photos. Camera groups for "
         f"training are recomputed from each fold's training photos only ({group_mismatch} training photos, summed over "
         "the folds, fall in a different group than in the fixed analysis grouping). LEACE rows: camera_fixes.py, C "
         "tuned in inner folds. In %.", "",
         "## 1. Every camera group against the 10 % target", "",
         "| Method | Referable sent home: H (55 eyes) | M (717) | S (715) | Worst group | Urgent referred urgently: worst group | "
         "C1 overall: referable sent home / urgent referred urgently | Healthy sent home | Referred per 1,000 | "
         "AUROC referable | Same-camera AUROC |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in order:
        r = M.loc[m]
        h = G.loc[(m, "H")]
        cells_ = " | ".join(f"{100 * G.loc[(m, g_), 'ref_home']:.1f}" for g_ in ("M", "S"))
        L.append(f"| {NAMES[m]} | {100 * h['ref_home']:.1f} ({100 * h['ref_home_lo']:.0f}–{100 * h['ref_home_hi']:.0f}) | "
                 f"{cells_} | {100 * r['worst_group_ref_home']:.1f} | {100 * r['worst_group_urgent_urgent']:.1f} | "
                 f"{100 * r['ref_home']:.1f} / {100 * r['urgent_urgent']:.1f} | {100 * r['healthy_home']:.1f} | "
                 f"{1000 * r['referred']:.0f} | {r['auroc_ref']:.3f} | {r['same_camera_auroc_ref']:.3f} |")
    L += ["", "H-camera referable eyes: exact McNemar test on sent home or not, against plain training and against "
          "the camera-decorrelated weights (eyes the method saves / eyes it newly sends home):", "",
          "| Method | Against ERM: saved / lost, p | Against ours: saved / lost, p |", "|---|---|---|"]
    T = tests.set_index(["method", "against"])
    for m in order:
        parts = []
        for base in ("erm", "decorr"):
            if (m, base) in T.index:
                t = T.loc[(m, base)]
                parts.append(f"{int(t['fixed'])} / {int(t['broken'])}, p = {t['p']:.2g}")
            else:
                parts.append("–")
        L.append(f"| {NAMES[m]} | {parts[0]} | {parts[1]} |")
    L += ["", "## 2. Healthy eyes sent home per camera group (the cost)", "",
          "| Method | H | M | S |", "|---|---|---|---|"]
    for m in order:
        L.append(f"| {NAMES[m]} | " + " | ".join(f"{100 * G.loc[(m, g_), 'healthy_home']:.1f}" for g_ in GROUPS) + " |")
    if htests is not None and len(htests):
        n_h = int(htests["healthy"].iloc[0])
        L += ["", f"Every healthy APTOS eye ({n_h}): exact McNemar test on sent home or not, each method against the "
              "camera-decorrelated weights (eyes only the method sends home / eyes only ours sends home):", "",
              "| Method | Healthy sent home | Only the method / only ours | p |", "|---|---|---|---|"]
        HT = htests.set_index("method")
        for m in order:
            if m in HT.index:
                h = HT.loc[m]
                L.append(f"| {NAMES[m]} | {100 * M.loc[m, 'healthy_home']:.1f} | {int(h['home_only_method'])} / "
                         f"{int(h['home_only_ours'])} | {h['p']:.2g} |")
    L += ["", "## 3. New clinics", "",
          "Each trained method refitted on all of APTOS. B6: APTOS thresholds (from the method's out-of-fold APTOS "
          f"predictions) used unchanged; C2: thresholds refitted on k labelled local photos ({args.draws} draws, the "
          "same for every method). Referable sent home / urgent referred urgently / healthy sent home; last column: "
          "share of draws meeting both targets.", "",
          "| Method | Clinic | AUROC referable | B6 | C2, smaller k | C2, larger k |", "|---|---|---|---|---|---|"]
    C = clin
    for m in TRAINED:
        for ds, dn in (("idrid", "IDRiD"), ("messidor2", "Messidor-2")):
            b6 = C[(C["method"] == m) & (C["dataset"] == ds) & (C["rule"] == "B6")].iloc[0]
            c2 = C[(C["method"] == m) & (C["dataset"] == ds) & (C["rule"] == "C2")].sort_values("k")
            fmt = lambda r: (f"k = {int(r['k'])}: {100 * r['ref_home']:.1f} / {100 * r['urgent_urgent']:.1f} / "  # noqa: E731
                             f"{100 * r['healthy_home']:.1f} (both targets {100 * r['targets_met']:.0f} %)")
            L.append(f"| {NAMES[m]} | {dn} | {b6['auroc_ref']:.3f} | {100 * b6['ref_home']:.1f} / "
                     f"{100 * b6['urgent_urgent']:.1f} / {100 * b6['healthy_home']:.1f} | {fmt(c2.iloc[0])} | {fmt(c2.iloc[1])} |")
    L += ["", f"Made {time.strftime('%Y-%m-%d %H:%M')} by group_robust.py ({elapsed:.0f} s).", ""]
    (out / "summary.md").write_text("\n".join(L))


def plot(met, grp_df, path):
    """Trade-off: the worst camera group's referable eyes sent home (fairness) against healthy eyes sent home
    (usefulness), one point per method, with the 95 % Wilson interval of that worst group. Hollow markers: the
    worst group's urgent eyes referred urgently fall below 90 %. Labels sit at fixed places, clear of the bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    halo = [patheffects.withStroke(linewidth=3, foreground="white")]
    ink, ink2, grid_c, accent, other = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#7d7c77"
    short = {"erm": "Plain training (ERM)", "decorr": "Camera-decorrelated weights (ours)",
             "balanced": "Group-balanced weights", "dfr": "DFR", "groupdro": "Group DRO",
             "jtt5": "JTT, errors \u00d75", "jtt20": "JTT, errors \u00d720",
             "mondrian_g": "Per-group thresholds, fold calibration",
             "mondrian_gx": "Per-group thresholds, cross-fitted",
             "leace_healthy": "LEACE fitted on healthy eyes", "leace_grade": "LEACE within each grade"}
    # label position in data units: (x, y, ha), or "right" = just right of the interval's end, or "above"
    place = {"decorr": (10.6, 88.4, "left"), "erm": "above", "leace_healthy": (52.7, 94.2, "center"),
             "leace_grade": (50.4, 84.0, "left"), "jtt5": "right", "balanced": (26.0, 78.4, "left"),
             "mondrian_gx": (26.0, 73.0, "left"), "groupdro": "right", "jtt20": "right", "dfr": "right",
             "mondrian_g": "right"}
    leader = {"balanced", "mondrian_gx"}
    M = met.set_index("method")
    G = grp_df.set_index(["method", "group"])
    fig, ax = plt.subplots(figsize=(8.6, 5.8), dpi=200)
    ax.axvspan(0, 10, color=accent, alpha=0.06, lw=0)
    ax.axvline(10, color=ink, lw=0.9, ls=(0, (4, 3)))
    ax.text(9.6, 2.5, "every camera group\nmeets the 10 % target", ha="right", va="bottom", fontsize=7.5,
            color=ink2)
    ax.text(67.5, 2.5, "hollow marker: in its worst camera group, fewer than\n90 % of urgent eyes are referred "
            "urgently", ha="right", va="bottom", fontsize=7.3, color=ink2)
    for m in M.index:
        worst = max(GROUPS, key=lambda g_: G.loc[(m, g_), "ref_home"])
        x, lo, hi = (100 * G.loc[(m, worst), c] for c in ("ref_home", "ref_home_lo", "ref_home_hi"))
        yv = 100 * M.loc[m, "healthy_home"]
        urgent_ok = min(G.loc[(m, g_), "urgent_urgent"] for g_ in GROUPS) >= 0.9
        col = accent if m == "decorr" else (ink if m == "erm" else other)
        ax.errorbar([x], [yv], xerr=[[x - lo], [hi - x]], fmt="none", ecolor=col, elinewidth=0.8, capsize=2,
                    zorder=2)
        ax.plot([x], [yv], "o", ms=6.5 if m == "decorr" else 5, mec=col, mew=1.2,
                mfc=col if urgent_ok else "white", zorder=3)
        name = short[m] + (f" (worst group: {worst})" if worst != "H" else "")
        pos = place.get(m, "right")
        if pos == "right":
            lx, ly, ha = hi + 0.8, yv, "left"
        elif pos == "above":
            lx, ly, ha = x, yv + 3.2, "center"
        else:
            lx, ly, ha = pos
        ax.text(lx, ly, name, ha=ha, va="center", fontsize=7.3, color=col,
                fontweight="bold" if m == "decorr" else "normal", zorder=4, path_effects=halo)
        if m in leader:
            ax.plot([hi + 0.6, lx - 0.4], [yv, ly], color=col, lw=0.5, zorder=1)
        if m == "leace_healthy":       # label stands above the plain model's: a short line to its own point
            ax.plot([x, lx], [yv + 1.0, ly - 1.3], color=col, lw=0.5, zorder=1)
    ax.set_xlim(0, 68)
    ax.set_ylim(0, 100)
    ax.set_xlabel("worst camera group: referable eyes sent home (%), with 95 % Wilson interval", fontsize=8.5,
                  color=ink2)
    ax.set_ylabel("healthy eyes sent home (%), all cameras", fontsize=8.5, color=ink2)
    ax.grid(color=grid_c, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=ink2, labelsize=8, length=0)
    ax.set_title("Fairness against usefulness: C1 on APTOS, every photo decided out of fold (\u03b1 = 0.1); "
                 "upper left is better", fontsize=9.5, color=ink, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
