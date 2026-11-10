#!/usr/bin/env python3
"""
c1_referral.py - C1: a referral rule with one conformal threshold per class (Mondrian conformal
prediction [10]), compared with the simpler rules B3, B4 and B5, on the saved E1 predictions.

Nothing is retrained. In every E1 outer fold the thresholds are set on that fold's 15 % calibration
images (APTOS photos the model never trained on) and the rule is applied to the held-out APTOS fold, so
every APTOS photo is decided exactly once, by a rule that never saw it. The same thresholds are also
applied unchanged to IDRiD and Messidor-2: a new clinic without recalibration (B6), where C2 starts.

Input: the raw probabilities (p0-p2) by default; --probs calibrated uses the isotonic-calibrated ones (c0-c2).
The conformal step calibrates by itself. On APTOS both inputs give the same result (summary section 8), but at a
new clinic the isotonic map learnt on APTOS sets many sick eyes' probabilities to exactly 0, which ties them and
makes the thresholds useless (see e2_new_clinics.py). So C1 and C2 use the raw probabilities.
5-grade models are collapsed to the three classes first (0-1, 2, 3-4).

Rules (alpha = 0.1, i.e. 90 % targets, unless --alpha says otherwise):
  argmax  the most likely class, as in E1 (no guarantee)
  B3      two thresholds tuned on the calibration images [6, 17]: 90 % of their referable eyes have
          P(referable) >= t_ref and 90 % of their urgent eyes have P(urgent) >= t_urg.
          Urgent if P(urgent) >= t_urg, else routine if P(referable) >= t_ref, else no referral.
  B4      standard (marginal) split conformal prediction as in E0 / [8]: one LAC threshold for all
          classes; sets are turned into decisions with the C1 table below
  B5      "uncertain" (to a grader) if the top class probability is below 0.6, else the most likely class
  C1      class-wise (Mondrian) conformal prediction: one LAC threshold per class, each set only on the
          calibration images of that class. The set C(x) = {y : 1 - p(y|x) <= q_y} gives the decision:
              contains "urgent"                         -> urgent referral
              only "routine"                            -> routine referral
              only "no referral"                        -> no referral (rescreen)
              anything else ({no, routine} or empty)    -> uncertain: a human grader decides
          If calibration and test eyes are exchangeable, each class keeps P(y in C(x) | y) >= 1 - alpha,
          so at most alpha of the routine eyes and of the urgent eyes are sent home, and at least
          1 - alpha of the urgent eyes are referred urgently.
  C1-g    C1's sets with a different table (additional variant): only "urgent" -> urgent referral; "urgent" with
          any other class -> uncertain (grader). Fewer urgent referrals; the guarantee then counts urgent eyes
          referred urgently or sent to a grader
  C1-q    as C1, with one threshold per class and image-sharpness tertile (Mondrian by image quality;
          tertile edges from each fold's calibration images)

Outputs (in --out, default ../results/c1):
  summary.md                 all tables
  metrics.csv                every model, alpha and rule: pooled APTOS metrics with 95 % group-bootstrap CIs
  per_fold.csv               the same metrics for every outer fold, plus the external test sets
  thresholds.csv             the thresholds of every fold, rule and model
  decisions_<main>.csv.gz    every decision of the main model (APTOS test folds and the external sets)
  targets_by_rule.png        the two targets and the workload for every rule (main model)
  coverage_by_grade.png      coverage by DR grade: standard against class-wise conformal, and at new clinics
  workload_by_model.png      what C1 does with each course model

Usage:
  python c1_referral.py --data ../data_512 --results ../results/e1
"""
from __future__ import annotations

import argparse
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from conformal import (DECISION_NAMES, HOME, ROUTINE, UNCERTAIN, URGENT, classwise_thresholds,
                       decide_from_sets, marginal_threshold)

RULES = ["argmax", "B3", "B4", "B5", "C1", "C1-g", "C1-q"]
RULE_NAMES = {"argmax": "Most likely class", "B3": "B3 tuned thresholds", "B4": "B4 standard conformal",
              "B5": "B5 uncertain if top probability < 0.6", "C1": "C1 class-wise conformal",
              "C1-g": "C1-g as C1, sets with urgent and another class to a grader",
              "C1-q": "C1-q class-wise conformal by sharpness tertile"}
SET_RULES = ("B4", "C1", "C1-g", "C1-q")
MODEL_NAMES = {"b1": "B1 Most likely class", "b2": "B2 Image quality only", "b2a": "B2a Image size only",
               "b2b": "B2b Quality, without size", "lr-grade": "Logistic regression, 5 grades",
               "lr": "Logistic regression", "svm": "SVM (RBF)", "lda": "LDA", "mlp": "MLP",
               "gb": "Gradient boosting", "rf": "Random forest", "ada": "AdaBoost", "knn": "k-NN",
               "qda": "QDA", "gnb": "Naive Bayes", "cart": "Decision tree"}
EXTERNAL = [("idrid", "IDRiD, all 516"), ("idrid_test", "IDRiD, official test split (103)"),
            ("messidor2", "Messidor-2 (1,744)")]
B5_CUT = 0.6


# --------------------------------------------------------------------------------------------
# rules
# --------------------------------------------------------------------------------------------

def tuned_threshold(score_pos, sens):
    """Largest observed threshold t with at least `sens` of the positives scoring >= t."""
    s = np.sort(score_pos)
    m = int(math.floor(len(s) * (1 - sens) + 1e-9))
    return float(s[m])


def decide_grader(in_set):
    """C1-g: like decide_from_sets, but a set holding urgent and another class goes to a grader."""
    d = decide_from_sets(in_set)
    d[(d == URGENT) & (in_set.sum(1) > 1)] = UNCERTAIN
    return d


def tertile(values, edges):
    return np.digitize(values, edges)          # 0 = least sharp third, 2 = sharpest third


def fit_rules(P, y, sharp, alpha):
    """Everything each rule needs, fitted on one fold's calibration images."""
    pos_ref, pos_urg = y >= 1, y == 2
    edges = np.quantile(sharp, [1 / 3, 2 / 3])
    tq = tertile(sharp, edges)
    q_t = np.array([classwise_thresholds(P[tq == t], y[tq == t], alpha) for t in range(3)])
    return {"B3": (tuned_threshold(P[pos_ref, 1:].sum(1), 1 - alpha), tuned_threshold(P[pos_urg, 2], 1 - alpha)),
            "B4": marginal_threshold(P, y, alpha),
            "C1": classwise_thresholds(P, y, alpha),
            "C1-q": (edges, q_t),
            "n_class": np.bincount(y, minlength=3),
            "n_class_tertile": np.array([np.bincount(y[tq == t], minlength=3) for t in range(3)])}


def apply_rules(fit, P, sharp):
    """Decisions of every rule (and the sets of the set-valued rules) for the photos in P."""
    d, sets = {}, {}
    am = P.argmax(1)
    d["argmax"] = am.copy()
    t_ref, t_urg = fit["B3"]
    d["B3"] = np.where(P[:, 2] >= t_urg, URGENT, np.where(P[:, 1:].sum(1) >= t_ref, ROUTINE, HOME))
    d["B5"] = np.where(P.max(1) < B5_CUT, UNCERTAIN, am)
    sets["B4"] = (1.0 - P) <= fit["B4"]
    sets["C1"] = (1.0 - P) <= fit["C1"][None, :]
    edges, q_t = fit["C1-q"]
    sets["C1-q"] = (1.0 - P) <= q_t[tertile(sharp, edges)]
    sets["C1-g"] = sets["C1"]
    for r in SET_RULES:
        d[r] = decide_grader(sets[r]) if r == "C1-g" else decide_from_sets(sets[r])
    return d, sets


# --------------------------------------------------------------------------------------------
# metrics (weighted, so the same code serves the point estimate and the bootstrap)
# --------------------------------------------------------------------------------------------

def indicators(d, y, in_set=None):
    """Numerator / denominator indicator vectors of every metric."""
    ref, urg, rout, non = y >= 1, y == 2, y == 1, y == 0
    home, unc = d == HOME, d == UNCERTAIN
    is_urg, referred = d == URGENT, (d == ROUTINE) | (d == URGENT)
    one = np.ones(len(d), bool)
    m = {"ref_home": (ref & home, ref), "routine_home": (rout & home, rout), "urgent_home": (urg & home, urg),
         "urgent_urgent": (urg & is_urg, urg), "urgent_urgent_or_grader": (urg & (is_urg | unc), urg),
         "healthy_home": (non & home, non),
         "uncertain": (unc, one), "referred": (referred, one), "urgent_decisions": (is_urg, one),
         "home": (home, one), "urgent_ppv": (urg & is_urg, is_urg), "referred_ppv": (ref & referred, referred),
         "ref_uncertain": (ref & unc, ref)}
    if in_set is not None:
        covered = in_set[np.arange(len(y)), y]
        size = in_set.sum(1)
        for c in range(3):
            m[f"coverage_{c}"] = (covered & (y == c), y == c)
        m["set_size"] = (size.astype(float), one)
        m["empty"] = (size == 0, one)
    return m


def weighted(ind, W):
    """ind: {metric: (num, den)}; W: (B, n) weights -> {metric: (B,) values}."""
    out = {}
    for k, (num, den) in ind.items():
        n = W @ num.astype(float)
        dd = W @ den.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[k] = np.where(dd > 0, n / dd, np.nan)
    return out


def scale(metric, v):
    return 1000 * v if metric in ("referred", "urgent_decisions", "home") else (v if metric == "set_size" else 100 * v)


# --------------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------------

def camera_groups(man):
    """APTOS photo sizes grouped by their share of referable eyes, as in the camera study:
    H <= 10 %, S >= 80 %, M otherwise."""
    apt = man[man["dataset"] == "aptos"]
    size = apt["orig_w"].astype(str) + "x" + apt["orig_h"].astype(str)
    share = apt["referable"].astype(float).groupby(size).mean()
    group = pd.Series(np.where(share <= 0.10, "H", np.where(share >= 0.80, "S", "M")), index=share.index)
    return size.map(group).reindex(man.index)


def load_model(results: Path, key: str, man: pd.DataFrame, probs: str):
    frames = []
    for f in sorted((results / key).glob("fold*.csv.gz")):
        stem = f.name[4:].split(".")[0]
        if not stem.isdigit():
            continue
        df = pd.read_csv(f)
        df["fold_model"] = int(stem)
        frames.append(df)
    if len(frames) == 0:
        return None
    df = pd.concat(frames, ignore_index=True)
    col = "c" if probs == "calibrated" else "p"
    k = sum(c.startswith(col) and c[1:].isdigit() for c in df.columns)
    P = df[[f"{col}{i}" for i in range(k)]].to_numpy(float)
    if k == 5:                                   # 5 grades -> no referral (0-1), routine (2), urgent (3-4)
        P = np.column_stack([P[:, 0] + P[:, 1], P[:, 2], P[:, 3] + P[:, 4]])
    P = np.clip(P, 0, None)
    P = P / P.sum(1, keepdims=True)
    info = man.set_index("uid").loc[df["uid"], ["dataset", "split", "grade", "ref3", "q_lapvar", "camera_group"]]
    df = pd.concat([df[["uid", "role", "fold_model"]].reset_index(drop=True), info.reset_index(drop=True)], axis=1)
    df["set"] = np.where(df["role"] != "external", "aptos",
                         np.where(df["dataset"] == "messidor2", "messidor2", "idrid"))
    return df, P


def run_model(df, P, alpha):
    """Fit every rule per fold on its calibration images; decide its test fold and the external sets."""
    y = df["ref3"].to_numpy(int)
    sharp = df["q_lapvar"].to_numpy(float)
    rows, thresholds = [], []
    dec = {r: np.full(len(df), -1) for r in RULES}
    sets = {r: np.zeros((len(df), 3), bool) for r in SET_RULES}
    for fold, idx in df.groupby("fold_model").indices.items():
        cal = idx[df["role"].to_numpy()[idx] == "cal"]
        use = idx[df["role"].to_numpy()[idx] != "cal"]
        fit = fit_rules(P[cal], y[cal], sharp[cal], alpha)
        d, s = apply_rules(fit, P[use], sharp[use])
        for r in RULES:
            dec[r][use] = d[r]
        for r in SET_RULES:
            sets[r][use] = s[r]
        n0, n1, n2 = fit["n_class"]
        q = fit["C1"]
        thresholds.append({"fold": fold, "n_cal": len(cal), "n_cal_0": n0, "n_cal_1": n1, "n_cal_2": n2,
                           "B3_t_ref": fit["B3"][0], "B3_t_urg": fit["B3"][1], "B4_q": fit["B4"],
                           "C1_q0": q[0], "C1_q1": q[1], "C1_q2": q[2],
                           "C1q_edge1": fit["C1-q"][0][0], "C1q_edge2": fit["C1-q"][0][1],
                           "C1q_min_n_cell": int(fit["n_class_tertile"].min()),
                           "C1q_inf_cells": int(np.isinf(fit["C1-q"][1]).sum()),
                           "expected_cov_0": math.ceil((n0 + 1) * (1 - alpha)) / (n0 + 1),
                           "expected_cov_1": math.ceil((n1 + 1) * (1 - alpha)) / (n1 + 1),
                           "expected_cov_2": math.ceil((n2 + 1) * (1 - alpha)) / (n2 + 1),
                           "expected_cov_marginal": math.ceil((len(cal) + 1) * (1 - alpha)) / (len(cal) + 1)})
    return dec, sets, pd.DataFrame(thresholds)


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="C1: class-wise conformal referral rule against B3, B4, B5.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", default="../data_512")
    ap.add_argument("--results", default="../results/e1")
    ap.add_argument("--out", default="../results/c1")
    ap.add_argument("--models", default="lr,svm,lr-grade,lda,mlp,gb,ada,knn,rf,qda,gnb,cart,b2,b2a,b2b,b1")
    ap.add_argument("--main", default="lr", help="model used for the detailed tables and figures")
    ap.add_argument("--alphas", default="0.1,0.05")
    ap.add_argument("--probs", default="raw", choices=["raw", "calibrated"])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = Path(args.results)
    alphas = [float(a) for a in args.alphas.split(",")]

    man = pd.read_csv(Path(args.data) / "manifest.csv")
    man["camera_group"] = camera_groups(man)
    apt = man[(man["dataset"] == "aptos") & (man["fold"] >= 0)].reset_index(drop=True)
    groups = pd.factorize(apt["group_id"])[0]
    G = groups.max() + 1
    rng = np.random.default_rng(args.seed)                       # the E1 resampling scheme
    W = np.vstack([np.ones(len(apt))] +
                  [np.bincount(rng.integers(0, G, G), minlength=G)[groups] for _ in range(args.boot)])
    auroc = {}
    met_file = results / "metrics.csv"
    if met_file.exists():
        e1 = pd.read_csv(met_file)
        auroc = dict(zip(e1["model"], e1["auroc_referable"]))

    metric_rows, fold_rows, thr_rows = [], [], []
    detail = {}                                                  # main model: decisions, sets, frame

    def evaluate(key, label, probs, alpha_list):
        got = load_model(results, key, man, probs)
        if got is None:
            print(f"no predictions for {key}; skipped")
            return
        df, P = got
        test = (df["role"] == "test").to_numpy()
        pos = pd.Index(df.loc[test, "uid"]).get_indexer(apt["uid"])
        if (pos < 0).any() or test.sum() != len(apt):
            raise SystemExit(f"{key}: the test folds do not cover every APTOS photo exactly once")
        y_apt = df.loc[test, "ref3"].to_numpy(int)[pos]
        parts = {}
        not_cal = np.flatnonzero((df["role"] != "cal").to_numpy())
        for (fold, set_key), ii in df.iloc[not_cal].groupby(["fold_model", "set"]).indices.items():
            ii = not_cal[ii]
            parts[(fold, set_key)] = ii
            if set_key == "idrid":
                parts[(fold, "idrid_test")] = ii[(df["split"].to_numpy()[ii] == "test")]
        for alpha in alpha_list:
            dec, sets, thr = run_model(df, P, alpha)
            thr_rows.append(thr.assign(model=label, alpha=alpha))
            if label == args.main:
                detail[alpha] = (df, dec, sets, thr)
            for r in RULES:
                d_apt = dec[r][test][pos]
                s_apt = sets[r][test][pos] if r in SET_RULES else None
                vals = weighted(indicators(d_apt, y_apt, s_apt), W)
                row = {"model": label, "probs": probs, "alpha": alpha, "rule": r, "n": len(d_apt)}
                for k, v in vals.items():
                    row[k] = v[0]
                    with warnings.catch_warnings():              # e.g. no urgent decisions at all
                        warnings.simplefilter("ignore", RuntimeWarning)
                        lo, hi = np.nanpercentile(v[1:], [2.5, 97.5])
                    row[k + "_lo"], row[k + "_hi"] = lo, hi
                metric_rows.append(row)
                for (fold, name), ii in parts.items():          # every fold model, every test set
                    y = df["ref3"].to_numpy(int)[ii]
                    s = sets[r][ii] if r in SET_RULES else None
                    v = weighted(indicators(dec[r][ii], y, s), np.ones((1, len(ii))))
                    fold_rows.append({"model": label, "probs": probs, "alpha": alpha, "rule": r, "fold": fold,
                                      "set": name, "n": len(ii), **{k: x[0] for k, x in v.items()}})
        print(f"{label}: done ({time.time() - t0:.0f} s)")

    for key in [k.strip() for k in args.models.split(",") if k.strip()]:
        evaluate(key, key, args.probs, alphas)
    other = "raw" if args.probs == "calibrated" else "calibrated"
    for key in [k.strip() for k in args.models.split(",") if k.strip()]:  # robustness: the other probabilities
        evaluate(key, f"{key}:{other}", other, alphas[:1])
    if alphas[0] not in detail:
        raise SystemExit(f"no predictions for the main model {args.main}")

    met = pd.DataFrame(metric_rows)
    per_fold = pd.DataFrame(fold_rows)
    thr = pd.concat(thr_rows, ignore_index=True)
    met.to_csv(out / "metrics.csv", index=False, float_format="%.5f")
    per_fold.to_csv(out / "per_fold.csv", index=False, float_format="%.5f")
    thr.to_csv(out / "thresholds.csv", index=False, float_format="%.5f")

    # decisions of the main model, every rule, both alphas
    a0 = alphas[0]
    df, dec, sets, _ = detail[a0]
    keep = df["role"] != "cal"
    dd = df.loc[keep, ["uid", "role", "fold_model", "set", "split", "grade", "ref3", "camera_group"]].copy()
    for alpha in alphas:
        _, dec_a, sets_a, _ = detail[alpha]
        for r in RULES:
            dd[f"{r}_a{alpha:g}"] = pd.Series(dec_a[r][keep.to_numpy()]).map(DECISION_NAMES).to_numpy()
    dd.to_csv(out / f"decisions_{args.main}.csv.gz", index=False)

    write_summary(out, args, met, per_fold, thr, detail, man, apt, auroc, alphas, time.time() - t0)
    plot_targets(met, args.main, a0, out / "targets_by_rule.png")
    plot_coverage(per_fold, detail, man, args.main, a0, out / "coverage_by_grade.png")
    plot_models(met, auroc, a0, out / "workload_by_model.png")
    print(f"wrote {out / 'summary.md'} ({time.time() - t0:.0f} s)")


# --------------------------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------------------------

def pct(r, m, ci=True, d=1):
    v = r[m]
    if pd.isna(v):
        return "–"
    s = f"{scale(m, v):.{d}f}"
    if ci and f"{m}_lo" in r and not pd.isna(r[f"{m}_lo"]):
        s += f" ({scale(m, r[m + '_lo']):.{d}f}–{scale(m, r[m + '_hi']):.{d}f})"
    return s


def per1000(r, m):
    return f"{1000 * r[m]:.0f}"


def fold_range(per_fold, model, alpha, rule, metric, set_key="aptos"):
    f = per_fold[(per_fold["model"] == model) & (per_fold["alpha"] == alpha) & (per_fold["rule"] == rule)
                 & (per_fold["set"] == set_key)][metric]
    return f"{100 * f.min():.0f}–{100 * f.max():.0f}" if len(f) else "–"


def write_summary(out, args, met, per_fold, thr, detail, man, apt, auroc, alphas, elapsed):
    a0 = alphas[0]
    main = args.main
    other_kind = "calibrated" if args.probs == "raw" else "raw"
    M = met.set_index(["model", "alpha", "rule"])
    T = thr[(thr["model"] == main)]
    L = ["# C1: a referral rule with one conformal threshold per class", "",
         f"Every rule below is fitted, in each E1 outer fold, on that fold's calibration images (about "
         f"{T['n_cal'].mean():.0f} APTOS photos the model never trained on: {T['n_cal_0'].mean():.0f} / "
         f"{T['n_cal_1'].mean():.0f} / {T['n_cal_2'].mean():.0f} per class) and applied to the held-out fold, so "
         f"each of the {len(apt):,} APTOS photos is decided once by a rule that never saw it. Model inputs: the "
         f"{args.probs} probabilities saved by E1. 95 % CIs: {args.boot:,} bootstrap resamples of whole groups "
         "(the E1 resamples). Targets: at most 10 % of referable eyes sent home, at least 90 % of urgent "
         "(grade 3–4) eyes referred urgently.", "",
         "Rules: **argmax** the most likely class (as in E1). **B3** two thresholds tuned on the calibration "
         "images so that 90 % of their referable eyes clear the first (P(referable)) and 90 % of their urgent "
         "eyes the second (P(urgent)). **B4** standard conformal prediction, one threshold for all classes (E0). "
         f"**B5** uncertain if the top class probability is below {B5_CUT}, else the most likely class. "
         "**C1** one conformal threshold per class; the set gives the decision: contains urgent → urgent; only "
         "routine → routine; only no referral → no referral; anything else → uncertain (a grader decides). "
         "**C1-g** (additional variant) uses C1's sets but sends a set holding urgent and another class to a "
         "grader instead of referring it urgently; its urgent guarantee counts urgent eyes referred urgently or "
         "sent to a grader. **C1-q** as C1 with one threshold per class and sharpness tertile. B4, C1 and C1-q turn "
         "their sets into decisions with the same table.", ""]

    # 1. main comparison
    L += [f"## 1. The rules compared ({MODEL_NAMES.get(main, main)}, α = {a0:g})", "",
          "In %, pooled over the 10 held-out folds, with 95 % CIs. \"Uncertain\": share of all eyes sent to a "
          "grader. Referred: routine and urgent decisions per 1,000 eyes screened (APTOS prevalence, 40.6 % "
          "referable). " + ("The most likely class is the decision of E1." if args.probs == "raw" else
          "The most likely class is taken from the same probabilities as the other rules, so it differs slightly "
          "from E1, which used the raw probabilities (12.8 % and 46.3 %)."), "",
          "| Rule | Referable eyes sent home (≤ 10) | Urgent eyes referred urgently (≥ 90) | Urgent eyes referred "
          "urgently or to a grader | Routine eyes sent home | Urgent eyes sent home | Healthy eyes sent home | Uncertain | "
          "Referred per 1,000 | Urgent referrals per 1,000 | Urgent referrals that are urgent |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in RULES:
        x = M.loc[(main, a0, r)]
        L.append(f"| {RULE_NAMES[r]} | {pct(x, 'ref_home')} | {pct(x, 'urgent_urgent')} | "
                 f"{pct(x, 'urgent_urgent_or_grader', False)} | "
                 f"{pct(x, 'routine_home', False)} | {pct(x, 'urgent_home', False)} | {pct(x, 'healthy_home', False)} | "
                 f"{pct(x, 'uncertain', False)} | {per1000(x, 'referred')} | {per1000(x, 'urgent_decisions')} | "
                 f"{pct(x, 'urgent_ppv', False, 0)} |")
    L += ["", "Fold to fold (lowest–highest of the 10 held-out folds, about 150 referable and 49 urgent eyes each):",
          "", "| Rule | Referable eyes sent home | Urgent eyes referred urgently |", "|---|---|---|"]
    for r in RULES:
        L.append(f"| {RULE_NAMES[r]} | {fold_range(per_fold, main, a0, r, 'ref_home')} | "
                 f"{fold_range(per_fold, main, a0, r, 'urgent_urgent')} |")
    L.append("")

    # 2. coverage against theory
    L += ["## 2. Coverage of each class against conformal theory", "",
          "Share of eyes whose true class is in their set. \"Expected\": the exact average coverage that "
          "conformal theory gives for continuous scores, ⌈(n + 1)(1 − α)⌉ / (n + 1), averaged over the folds "
          "(n: calibration images of that class for C1, all calibration images for B4). B4 promises only the "
          "average over all eyes.", "",
          "| Rule | Class | Calibration images per fold | Expected | Observed (95 % CI) | Lowest–highest fold |",
          "|---|---|---|---|---|---|"]
    names = ["no referral", "routine", "urgent"]
    for r in ("B4", "C1", "C1-q"):
        x = M.loc[(main, a0, r)]
        for c in range(3):
            if r == "C1":
                n_c, exp = f"{T[T['alpha'] == a0][f'n_cal_{c}'].mean():.0f}", \
                    f"{100 * T[T['alpha'] == a0][f'expected_cov_{c}'].mean():.1f}"
            elif r == "B4":
                n_c, exp = f"{T[T['alpha'] == a0]['n_cal'].mean():.0f} (all classes)", \
                    f"{100 * T[T['alpha'] == a0]['expected_cov_marginal'].mean():.1f} (all eyes)"
            else:
                n_c, exp = "about a third of C1's", "≥ 90 (per tertile)"
            L.append(f"| {RULE_NAMES[r]} | {names[c]} | {n_c} | {exp} | {pct(x, f'coverage_{c}')} | "
                     f"{fold_range(per_fold, main, a0, r, f'coverage_{c}')} |")
    Ta = T[T["alpha"] == a0]
    n2 = Ta["n_cal_2"].to_numpy()
    b3_exp = np.mean((n2 - np.floor(n2 * a0 + 1e-9)) / (n2 + 1))
    x = M.loc[(main, a0, "B3")]
    L.append(f"| {RULE_NAMES['B3']} | urgent (P(urgent) at or above its threshold) | {n2.mean():.0f} | "
             f"{100 * b3_exp:.1f} | {pct(x, 'urgent_urgent')} | {fold_range(per_fold, main, a0, 'B3', 'urgent_urgent')} |")
    for r in ("B4", "C1", "C1-q"):
        x = M.loc[(main, a0, r)]
        L.append(f"| {RULE_NAMES[r]} | mean set size | | | {x['set_size']:.2f} classes; empty sets "
                 f"{100 * x['empty']:.1f} % | |")
    L += ["", "B3 and C1 set their urgent threshold on the same calibration images and differ by one image: C1 takes "
          "the ⌈(n + 1)(1 − α)⌉-th smallest score, B3 the plain empirical 90 % point, whose expected coverage "
          f"(n − ⌊nα⌋) / (n + 1) is {100 * b3_exp:.1f} % here, just under the target. With fewer calibration "
          "images (a new clinic, C2) the gap widens: for n = 20, B3 expects 85.7 % and C1 90.5 %.", ""]

    # 3. decisions by grade
    df, dec, sets, _ = detail[a0]
    test = (df["role"] == "test").to_numpy()
    g = df.loc[test, "grade"].to_numpy(int)
    L += ["## 3. What happens to each DR grade", "",
          f"Share of the eyes of each grade given each decision, in % ({MODEL_NAMES.get(main, main)}, α = {a0:g}, "
          "all 3,662 APTOS eyes).", ""]
    for r in ("B3", "C1"):
        L += [f"**{RULE_NAMES[r]}**", "", "| Grade | Eyes | No referral | Uncertain | Routine | Urgent |",
              "|---|---|---|---|---|---|"]
        d = dec[r][test]
        for gr in range(5):
            s = g == gr
            share = [100 * np.mean(d[s] == k) for k in (HOME, UNCERTAIN, ROUTINE, URGENT)]
            L.append(f"| {gr} | {s.sum()} | " + " | ".join(f"{v:.1f}" for v in share) + " |")
        L.append("")

    # 4. stricter target
    if len(alphas) > 1:
        a1 = alphas[1]
        L += [f"## 4. A stricter target: α = {a1:g} ({100 * (1 - a1):.0f} % targets)", "",
              f"Same rules with {100 * (1 - a1):.0f} % in place of 90 % (B5 has no target and is unchanged). Targets: "
              f"at most {100 * a1:.0f} % of referable eyes sent home, at least {100 * (1 - a1):.0f} % of urgent eyes "
              "referred urgently.", "",
              "| Rule | Referable eyes sent home | Urgent eyes referred urgently | Healthy eyes sent home | Uncertain | "
              "Referred per 1,000 | Urgent referrals per 1,000 |", "|---|---|---|---|---|---|---|"]
        for r in ("B3", "B4", "C1", "C1-q"):
            x = M.loc[(main, a1, r)]
            L.append(f"| {RULE_NAMES[r]} | {pct(x, 'ref_home')} | {pct(x, 'urgent_urgent')} | "
                     f"{pct(x, 'healthy_home', False)} | {pct(x, 'uncertain', False)} | {per1000(x, 'referred')} | "
                     f"{per1000(x, 'urgent_decisions')} |")
        L.append("")

    # 5. every model
    L += [f"## 5. C1 with every model (α = {a0:g})", "",
          "The guarantee does not depend on the model; what the model changes is how many eyes can safely be "
          f"sent home. Sorted by the referable AUROC of E1. Last column: healthy eyes sent home when the {other_kind} "
          "probabilities are used instead (the targets hold there too).", "",
          "| Model | AUROC referable (E1) | Referable eyes sent home | Urgent eyes referred urgently | Healthy eyes "
          f"sent home | Uncertain | Referred per 1,000 | Sent home per 1,000 | Healthy eyes sent home, {other_kind} |",
          "|---|---|---|---|---|---|---|---|---|"]
    models = [m for m in met["model"].unique() if ":" not in m]
    models.sort(key=lambda m: -auroc.get(m, 0))
    for m in models:
        x = M.loc[(m, a0, "C1")]
        z = M.loc[(f"{m}:{other_kind}", a0, "C1")] if (f"{m}:{other_kind}", a0, "C1") in M.index else None
        L.append(f"| {MODEL_NAMES.get(m, m)} | {auroc.get(m, float('nan')):.3f} | {pct(x, 'ref_home', False)} | "
                 f"{pct(x, 'urgent_urgent', False)} | {pct(x, 'healthy_home', False)} | {pct(x, 'uncertain', False)} | "
                 f"{per1000(x, 'referred')} | {per1000(x, 'home')} | "
                 f"{pct(z, 'healthy_home', False) if z is not None else '–'} |")
    L += ["", "Ties in the score cost efficiency, not validity: QDA's raw probabilities are pinned at 0 or 1 for most "
          "photos (77 % have a top probability above 0.99), so many scores tie and C1 has to refer more eyes; its "
          "calibrated probabilities are spread out and work better. For logistic regression both kinds agree.", ""]

    # 6. subgroups
    L += [f"## 6. Subgroups: camera group and image sharpness ({MODEL_NAMES.get(main, main)}, α = {a0:g})", "",
          "Camera groups as in the camera study (APTOS photo sizes with ≤ 10 % referable eyes: H; ≥ 80 %: S; "
          "the rest: M). Sharpness tertiles: thirds of the variance of the Laplacian, with edges set on each "
          "fold's calibration images. Small groups give rough numbers: the counts are shown.", "",
          "| Subgroup | Eyes (referable / urgent) | C1: referable sent home | C1: urgent referred urgently | "
          "C1: healthy sent home | C1-q: referable sent home | C1-q: urgent referred urgently | C1-q: healthy "
          "sent home |", "|---|---|---|---|---|---|---|---|"]
    cg = df.loc[test, "camera_group"].to_numpy()
    sharp = df.loc[test, "q_lapvar"].to_numpy(float)
    folds = df.loc[test, "fold_model"].to_numpy()
    terc = np.full(test.sum(), -1)
    thr_main = detail[a0][3]
    for _, row in thr_main.iterrows():
        s = folds == row["fold"]
        terc[s] = tertile(sharp[s], [row["C1q_edge1"], row["C1q_edge2"]])
    y = df.loc[test, "ref3"].to_numpy(int)
    subgroups = [(f"camera group {k}", cg == k) for k in ("H", "M", "S")] + \
                [(f"sharpness: {n} third", terc == t) for t, n in enumerate(("least sharp", "middle", "sharpest"))]

    def rate(d, num, den):
        return f"{100 * np.sum(num & d) / max(np.sum(den), 1):.1f}" if np.sum(den) else "–"
    for name, s in subgroups:
        cells = []
        for r in ("C1", "C1-q"):
            d = dec[r][test]
            cells += [rate(d[s] == HOME, y[s] >= 1, y[s] >= 1), rate(d[s] == URGENT, y[s] == 2, y[s] == 2),
                      rate(d[s] == HOME, y[s] == 0, y[s] == 0)]
        L.append(f"| {name} | {s.sum()} ({np.sum(y[s] >= 1)} / {np.sum(y[s] == 2)}) | " + " | ".join(cells) + " |")
    L += ["", f"C1-q: smallest calibration cell (class × tertile) {int(thr_main['C1q_min_n_cell'].min())} images; "
          f"cells with too few images for a finite threshold: {int(thr_main['C1q_inf_cells'].sum())} of "
          f"{9 * len(thr_main)} (that class is then in every set of its tertile).", ""]

    # 7. new clinics without recalibration
    L += [f"## 7. The same thresholds at new clinics, without recalibration ({MODEL_NAMES.get(main, main)}, "
          f"α = {a0:g})", "",
          "Each fold's APTOS thresholds applied unchanged to IDRiD and Messidor-2 (B6 in the proposal). Mean over "
          "the 10 fold models, lowest–highest in brackets. This is the starting point of C2, which refits the "
          "thresholds on a few labelled local images.", "",
          "| Test set | Rule | Referable eyes sent home | Urgent eyes referred urgently | Healthy eyes sent home | "
          "Uncertain | Coverage: no referral / routine / urgent |", "|---|---|---|---|---|---|---|"]
    pf = per_fold[(per_fold["model"] == main) & (per_fold["alpha"] == a0)]

    def agg(rule, set_key, m):
        v = pf[(pf["rule"] == rule) & (pf["set"] == set_key)][m]
        return f"{100 * v.mean():.1f} ({100 * v.min():.0f}–{100 * v.max():.0f})" if len(v) else "–"
    for set_key, set_name in [("aptos", "APTOS, held-out folds")] + EXTERNAL:
        for r in ("B3", "B4", "C1"):
            cov = " / ".join(f"{100 * pf[(pf['rule'] == r) & (pf['set'] == set_key)][f'coverage_{c}'].mean():.0f}"
                             for c in range(3)) if r in SET_RULES else "–"
            L.append(f"| {set_name} | {RULE_NAMES[r]} | {agg(r, set_key, 'ref_home')} | "
                     f"{agg(r, set_key, 'urgent_urgent')} | {agg(r, set_key, 'healthy_home')} | "
                     f"{agg(r, set_key, 'uncertain')} | {cov} |")
    L.append("")

    raw_key = [m for m in met["model"].unique() if m == f"{main}:{other_kind}"]
    if raw_key:
        rk = raw_key[0]
        other = rk.split(":")[1]
        L += [f"## 8. Robustness: {other} instead of {args.probs} probabilities ({MODEL_NAMES.get(main, main)}, "
              f"α = {a0:g})", "",
              "Same folds, calibration images and rules; only the input probabilities change.", "",
              f"| Rule | Referable eyes sent home: {args.probs} / {other} | Urgent eyes referred urgently: "
              f"{args.probs} / {other} | Healthy eyes sent home: {args.probs} / {other} | Uncertain: {args.probs} / "
              f"{other} | Referred per 1,000: {args.probs} / {other} |", "|---|---|---|---|---|---|"]
        for r in ("argmax", "B3", "B4", "B5", "C1"):
            x, z = M.loc[(main, a0, r)], M.loc[(rk, a0, r)]
            L.append(f"| {RULE_NAMES[r]} | {pct(x, 'ref_home', False)} / {pct(z, 'ref_home', False)} | "
                     f"{pct(x, 'urgent_urgent', False)} / {pct(z, 'urgent_urgent', False)} | "
                     f"{pct(x, 'healthy_home', False)} / {pct(z, 'healthy_home', False)} | "
                     f"{pct(x, 'uncertain', False)} / {pct(z, 'uncertain', False)} | "
                     f"{per1000(x, 'referred')} / {per1000(z, 'referred')} |")
        L.append("")

    L += ["## Notes", "",
          "- The guarantee is an average over calibration draws and eyes: single folds vary (section 1, fold "
          "range), and with about 66 urgent calibration images per fold the realised coverage of one draw has a "
          "standard deviation of about 3.5 points.",
          "- C1's per-class guarantee covers routine and urgent eyes separately; B3's tuned threshold targets the "
          "pooled referable eyes, so its guarantee depends on the mix of routine and urgent eyes it was tuned on.",
          "- B4, C1 and C1-q use the decision table of the proposal, so an eye whose set contains urgent is "
          "referred urgently even if other classes are in its set.",
          f"- Input probabilities: {args.probs} (section 8 repeats the main comparison with the other kind).",
          f"- Made {time.strftime('%Y-%m-%d %H:%M')} by c1_referral.py ({elapsed:.0f} s).", ""]
    (out / "summary.md").write_text("\n".join(L))


# --------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, GRAY = "#2a78d6", "#a8a7a1"


def _style(ax, grid_axis="x"):
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8, length=0)


def plot_targets(met, main, alpha, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = met[(met["model"] == main) & (met["alpha"] == alpha)].set_index("rule").loc[RULES[::-1]]
    short = {"argmax": "Most likely class", "B3": "B3 tuned thresholds", "B4": "B4 standard conformal",
             "B5": "B5 top probability < 0.6", "C1": "C1 class-wise conformal",
             "C1-g": "C1-g urgent + other class to grader", "C1-q": "C1-q by sharpness tertile"}
    labels = [short[r] for r in RULES][::-1]
    col = [BLUE if r.startswith("C1") else GRAY for r in rows.index]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), dpi=200, sharey=True,
                             gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for ax, m, target, title, side in ((axes[0], "ref_home", 100 * alpha,
                                        f"Referable eyes sent home (%), target ≤ {100 * alpha:.0f}", "max"),
                                       (axes[1], "urgent_urgent", 100 * (1 - alpha),
                                        f"Urgent eyes referred urgently (%), target ≥ {100 * (1 - alpha):.0f}", "min")):
        v = 100 * rows[m].to_numpy()
        lo, hi = 100 * rows[m + "_lo"].to_numpy(), 100 * rows[m + "_hi"].to_numpy()
        ax.barh(y, v, color=col, height=0.62)
        ax.errorbar(v, y, xerr=[v - lo, hi - v], fmt="none", ecolor=INK2, elinewidth=1, capsize=0)
        ax.axvline(target, color=INK, lw=1, ls=(0, (4, 3)))
        for yi, vi, hi_i in zip(y, v, hi):
            xt = (max(hi_i, target) if side == "max" else hi_i) + 1.5     # never on top of the target line
            ax.text(xt, yi, f"{vi:.1f}", va="center", fontsize=7.5, color=INK2)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_xlim(0, 105 if side == "min" else max(45, float(np.nanmax(hi)) + 8))
        _style(ax)
    ax = axes[2]
    parts = [("urgent_decisions", "urgent referral", "#104281"), ("referred_routine", "routine referral", BLUE),
             ("uncertain", "uncertain (grader)", "#86b6ef")]
    left = np.zeros(len(rows))
    for m, name, c in parts:
        v = (1000 * (rows["referred"] - rows["urgent_decisions"]) if m == "referred_routine"
             else 1000 * rows[m]).to_numpy()
        ax.barh(y, v, left=left, color=c, height=0.62, label=name, edgecolor="white", linewidth=1)
        left += v
    for yi, tot in zip(y, left):
        ax.text(tot + 10, yi, f"{tot:.0f}", va="center", fontsize=7.5, color=INK2)
    ax.set_xlim(0, 1080)
    ax.set_title("Eyes not sent home, per 1,000 screened", fontsize=9.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", bbox_to_anchor=(1.0, -0.42), ncol=3)
    _style(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8, color=INK)
    fig.suptitle(f"C1 and simpler rules on APTOS ({MODEL_NAMES.get(main, main).lower()}, 10 held-out folds, "
                 f"bars: 95 % CI)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_coverage(per_fold, detail, man, main, alpha, path):
    """Coverage by DR grade: B4 against C1 on APTOS, and C1 thresholds at the new clinics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df, dec, sets, _ = detail[alpha]
    use = df["role"] != "cal"
    rows = []
    for (fold, set_key), idx in df[use].groupby(["fold_model", "set"]).indices.items():
        idx = df.index[use][idx]
        g = df.loc[idx, "grade"].to_numpy(int)
        y = df.loc[idx, "ref3"].to_numpy(int)
        for r in ("B4", "C1"):
            cov = sets[r][idx][np.arange(len(idx)), y]
            for gr in range(5):
                s = g == gr
                if s.any():
                    rows.append({"fold": fold, "set": set_key, "rule": r, "grade": gr, "cov": cov[s].mean()})
    c = pd.DataFrame(rows).groupby(["set", "rule", "grade"])["cov"].agg(["mean", "min", "max"]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), dpi=200, sharey=True)
    panels = [(axes[0], "APTOS, held-out folds", [("aptos", "B4", GRAY, "B4 standard conformal (one threshold)"),
                                                  ("aptos", "C1", BLUE, "C1 class-wise conformal")]),
              (axes[1], "C1 thresholds from APTOS, no recalibration",
               [("aptos", "C1", BLUE, "APTOS"), ("idrid", "C1", "#eb6834", "IDRiD"),
                ("messidor2", "C1", "#1baf7a", "Messidor-2")])]
    for ax, title, series in panels:
        ax.axhline(100 * (1 - alpha), color=INK, lw=1, ls=(0, (4, 3)), label=f"target {100 * (1 - alpha):.0f} %")
        n = len(series)
        for j, (set_key, rule, colr, name) in enumerate(series):
            s = c[(c["set"] == set_key) & (c["rule"] == rule)]
            x = s["grade"].to_numpy() + (j - (n - 1) / 2) * 0.13
            ax.vlines(x, 100 * s["min"], 100 * s["max"], color=colr, lw=2)
            ax.plot(x, 100 * s["mean"], "-o", color=colr, lw=2, ms=6, mec="white", mew=1.2, label=name)
        ax.set_xticks(range(5))
        ax.set_xticklabels(["0\nnone", "1\nmild", "2\nmoderate", "3\nsevere", "4\nproliferative"], fontsize=7.5)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_ylim(0, 104)
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")
        _style(ax, "y")
        ax.spines["left"].set_visible(False)
    axes[0].set_ylabel("Eyes whose true class is in the set (%)\nmean and range over 10 folds", fontsize=8,
                       color=INK2)
    fig.suptitle(f"Coverage by DR grade ({MODEL_NAMES.get(main, main).lower()}, α = {alpha:g}; classes: grades "
                 "0–1, 2, 3–4)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_models(met, auroc, alpha, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = met[(met["alpha"] == alpha) & (met["rule"] == "C1") & ~met["model"].str.contains(":")].copy()
    m["auroc"] = m["model"].map(auroc)
    m = m.sort_values("auroc")
    y = np.arange(len(m))
    parts = [("home", "no referral", "#d6d5d0"), ("uncertain", "uncertain (grader)", "#86b6ef"),
             ("routine", "routine referral", BLUE), ("urgent_decisions", "urgent referral", "#104281")]
    fig, ax = plt.subplots(figsize=(8.2, 0.34 * len(m) + 1.4), dpi=200)
    left = np.zeros(len(m))
    for key, name, colr in parts:
        v = (1000 * (m["referred"] - m["urgent_decisions"]) if key == "routine" else 1000 * m[key]).to_numpy()
        ax.barh(y, v, left=left, color=colr, height=0.66, label=name, edgecolor="white", linewidth=1)
        left += v
    for yi, (_, r) in zip(y, m.iterrows()):
        ax.text(1008, yi, f"{100 * r['ref_home']:.1f} % / {100 * r['urgent_urgent']:.1f} %", va="center",
                fontsize=7, color=INK2)
    ax.text(1008, len(m) - 0.2, "referable sent home /\nurgent referred urgently", fontsize=7, color=INK2,
            va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{MODEL_NAMES.get(k, k)} ({a:.3f})" for k, a in zip(m["model"], m["auroc"])],
                       fontsize=7.5, color=INK)
    ax.set_xlim(0, 1000)
    ax.set_xlabel("Decisions per 1,000 APTOS eyes", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.11))
    ax.set_title(f"C1 (α = {alpha:g}) with every model, sorted by referable AUROC (in brackets)", fontsize=10,
                 color=INK, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
