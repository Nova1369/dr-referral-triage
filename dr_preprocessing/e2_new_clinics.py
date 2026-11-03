#!/usr/bin/env python3
"""
e2_new_clinics.py - C2 and E2: C1's three class-wise thresholds, refitted at a new clinic from k labelled
local photos.

The model never sees a photo from IDRiD or Messidor-2. It is a logistic regression trained on all 3,662
APTOS photos: standardised ResNet-50 features, C = 0.003 and no PCA reduction (E1's most frequent setting).
The conformal scores use its raw probabilities. The isotonic map of E1 (here fitted on 5-fold grouped
out-of-fold predictions) is learnt on APTOS and sets the low end of the scale to exactly 0. At Messidor-2
that erases the order among many sick eyes, so a conformal threshold can no longer tell them from healthy
eyes. Rule C2-cal shows this; the conformal step calibrates by itself.

For each new-clinic setting and each k, --draws times (default 200):
  1. draw k labelled photos of that clinic, by whole groups (copies of one photo, and the two eyes of a
     Messidor-2 exam, are drawn together);
  2. refit the rule on them;
  3. apply it to the clinic's other photos.

Settings:
  IDRiD              all 516 photos, random split
  IDRiD official     k drawn from the 413 training photos, tested on the 103 test photos
  Messidor-2         all 1,744 photos, random split

Rules (alpha = 0.1):
  B6    no recalibration: C1's three thresholds set on APTOS (out-of-fold predictions of all 3,662 photos)
  C2    C1 with its three thresholds refitted on the k local photos. A class with fewer than 1/alpha - 1 local
        photos (9 at alpha = 0.1) cannot get a finite threshold, so it is in every set: safe, uninformative
  B3-k  B3's two tuned thresholds refitted on the k local photos (plain empirical 90 % points)
  B4-k  standard conformal prediction (one threshold for all classes) refitted on the k local photos
  B7a   logistic regression retrained on the k local photos only (same penalty per photo); most likely class
  B7b   logistic regression retrained on APTOS plus the k local photos; C1 rule with the APTOS thresholds
  C2-cal, B6-cal  C2 and B6 on the isotonic-calibrated probabilities (for comparison)
Additional work, beyond the proposal:
  C2-PAC  per class, the threshold that keeps the clinic's own coverage >= 1 - alpha with probability
          >= 1 - delta (training-conditional conformal prediction [10]), not only on average over clinics.
          Its check uses the coverage over all the clinic's eyes of each class (the population the promise is
          about) as well as the coverage on the test photos
  dec     B6 and C2 with the camera-decorrelated model of the camera study

Outputs (in --out, default ../results/e2):
  summary.md
  draws.csv              every draw, k and rule
  curves.csv             means and 5-95 % ranges over draws
  aptos_model.csv.gz     the probabilities of both APTOS models for every photo
  e2_targets.png         C2 against B6, B3-k, B4-k
  e2_retrain.png         recalibrating against retraining
  e2_extensions.png           C2-PAC and the camera-decorrelated model

Usage:
  python e2_new_clinics.py --data ../data_512
"""
from __future__ import annotations

import argparse
import math
import time
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import beta as beta_dist
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from c1_referral import (HOME, ROUTINE, URGENT, UNCERTAIN, camera_groups, classwise_thresholds,
                         decide_from_sets, marginal_threshold, tuned_threshold)
from dr_preprocess import assign_folds
from run_models import isotonic_apply, isotonic_fit

C_LR = 0.003
SETTINGS = {"idrid": "IDRiD, all 516 photos (random split)",
            "idrid_split": "IDRiD official split (k from the 413 training photos, tested on the 103 test photos)",
            "messidor2": "Messidor-2, 1,744 photos (random split)"}
SHORT = {"idrid": "IDRiD", "idrid_split": "IDRiD official split", "messidor2": "Messidor-2"}
K_GRID = [10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400]
K_MAX = {"idrid": 300, "idrid_split": 300, "messidor2": 400}
K_RETRAIN = [20, 50, 100, 200, 400]
RULE_NAMES = {"B6-cal": "B6 on calibrated probabilities", "C2-cal": "C2 on calibrated probabilities",
              "B6": "B6 no recalibration (APTOS thresholds)", "C2": "C2 class-wise thresholds refitted locally",
              "B3-k": "B3-k tuned thresholds refitted locally", "B4-k": "B4-k standard conformal refitted locally",
              "B7a": "B7a retrained on the local photos only", "B7b": "B7b retrained on APTOS + local photos",
              "C2-PAC": "C2-PAC promise for the clinic's own draw",
              "B6-dec": "B6 with the camera-decorrelated model", "C2-dec": "C2 with the camera-decorrelated model"}


# --------------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------------

def fit_lr(Z, y, w=None, C=C_LR, init=None):
    model = LogisticRegression(C=C, max_iter=5000, warm_start=init is not None)
    if init is not None:
        model.coef_, model.intercept_, model.classes_ = init.coef_.copy(), init.intercept_.copy(), init.classes_
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(Z, y, sample_weight=w)


def proba3(model, Z):
    """Class probabilities with a column for every class, even one the model never saw."""
    P = np.zeros((len(Z), 3))
    P[:, np.asarray(model.classes_, int)] = model.predict_proba(Z)
    return P


def decorrelating_weights(y, g):
    """w = P(y) / P(y | camera group), as in the camera study: the camera says nothing about the class."""
    p_y = np.bincount(y, minlength=3) / len(y)
    w = np.ones(len(y))
    for grp in np.unique(g):
        s = g == grp
        p_yg = np.bincount(y[s], minlength=3) / s.sum()
        for c in range(3):
            w[s & (y == c)] = p_y[c] / p_yg[c]
    return w / w.mean()


def train_aptos_model(Z, y, groups_df, weights, seed):
    """All-APTOS model, its isotonic maps (5-fold grouped out-of-fold), and the raw and cross-fitted
    calibrated out-of-fold probabilities of every APTOS photo (the APTOS calibration scores of B6)."""
    folds = assign_folds(groups_df, 5, seed)[0].to_numpy()
    raw = np.zeros((len(y), 3))
    for j in range(5):
        tr, te = folds != j, folds == j
        m = fit_lr(Z[tr], y[tr], None if weights is None else weights[tr])
        raw[te] = proba3(m, Z[te])
    model = fit_lr(Z, y, weights)
    iso = isotonic_fit(raw, y)
    oof_cal = np.zeros_like(raw)
    for j in range(5):                          # isotonic map fitted without the fold it calibrates
        oof_cal[folds == j] = isotonic_apply(isotonic_fit(raw[folds != j], y[folds != j]), raw[folds == j])
    return model, iso, raw, oof_cal


# --------------------------------------------------------------------------------------------
# rules
# --------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def pac_index(n, alpha, delta):
    """Smallest j with P(Beta(j, n + 1 - j) >= 1 - alpha) >= 1 - delta (0 if none): the j-th smallest score
    then covers at least 1 - alpha of the clinic's eyes of that class with probability >= 1 - delta."""
    if n == 0:
        return 0
    j = np.arange(1, n + 1)
    ok = beta_dist.sf(1 - alpha, j, n + 1 - j) >= 1 - delta
    return int(j[ok][0]) if ok.any() else 0


def pac_thresholds(P, y, alpha, delta):
    q = np.full(3, np.inf)
    for c in range(3):
        s = np.sort(1.0 - P[y == c, c])
        j = pac_index(len(s), alpha, delta)
        if j:
            q[c] = s[j - 1]
    return q


def b3_thresholds(P, y, alpha):
    """B3's tuned thresholds; with no referable (urgent) photo to tune on, everyone is referred (urgently)."""
    ref, urg = y >= 1, y == 2
    t_ref = tuned_threshold(P[ref, 1:].sum(1), 1 - alpha) if ref.any() else 0.0
    t_urg = tuned_threshold(P[urg, 2], 1 - alpha) if urg.any() else 0.0
    return t_ref, t_urg


def b3_decide(P, t):
    t_ref, t_urg = t
    return np.where(P[:, 2] >= t_urg, URGENT, np.where(P[:, 1:].sum(1) >= t_ref, ROUTINE, HOME))


def set_decide(P, q):
    in_set = (1.0 - P) <= np.asarray(q)[None, :] if np.ndim(q) else (1.0 - P) <= q
    return decide_from_sets(in_set), in_set


def metrics(d, y, in_set=None):
    ref, urg, non = y >= 1, y == 2, y == 0
    m = {"ref_home": np.mean(d[ref] == HOME), "urgent_urgent": np.mean(d[urg] == URGENT),
         "healthy_home": np.mean(d[non] == HOME), "uncertain": np.mean(d == UNCERTAIN),
         "referred": np.mean((d == ROUTINE) | (d == URGENT)), "urgent_decisions": np.mean(d == URGENT),
         "nobody_home": float(not np.any(d == HOME))}
    if in_set is not None:
        cov = in_set[np.arange(len(y)), y]
        for c in range(3):
            m[f"coverage_{c}"] = cov[y == c].mean() if np.any(y == c) else np.nan
        m["set_size"] = in_set.sum(1).mean()
    return m


def auroc(pos, score):
    from scipy.stats import rankdata
    n1 = pos.sum()
    n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# --------------------------------------------------------------------------------------------
# draws
# --------------------------------------------------------------------------------------------

class Pool:
    """The photos of one setting that can be drawn for calibration, grouped, and its test photos."""

    def __init__(self, cal_idx, test_idx, group):
        self.test_idx = test_idx
        self.group = group
        g = group[cal_idx]
        uniq, inv = np.unique(g, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        bounds = np.cumsum(np.bincount(inv))
        self.members = np.split(cal_idx[order], bounds[:-1])
        self.sizes = np.array([len(x) for x in self.members])

    def draw(self, k, rng):
        chosen, total = [], 0
        for gi in rng.permutation(len(self.sizes)):
            s = self.sizes[gi]
            if total + s <= k:
                chosen.append(gi)
                total += s
                if total == k:
                    break
        cal = np.concatenate([self.members[gi] for gi in chosen])
        test = self.test_idx[~np.isin(self.group[self.test_idx], self.group[cal])]
        return cal, test


def retrain_task(setting, k, b, cal, test, Z, y, apt_idx, base, iso, q_aptos, n_apt):
    """B7a and B7b for one draw (run in parallel)."""
    out = []
    yc, yt = y[cal], y[test]
    # B7a: local photos only, the same penalty per photo as the APTOS model
    if len(np.unique(yc)) >= 2:
        m = fit_lr(Z[cal], yc, C=C_LR * n_apt / len(cal))
        P = proba3(m, Z[test])
    else:
        P = np.zeros((len(test), 3))
        P[:, yc[0]] = 1.0
    d = P.argmax(1)
    r = metrics(d, yt)
    r.update(auroc_ref=auroc(yt >= 1, P[:, 1:].sum(1)), auroc_urg=auroc(yt == 2, P[:, 2]))
    out.append({"setting": setting, "k": k, "draw": b, "rule": "B7a", **r})
    # B7b: APTOS + local photos, equal weight per photo, warm start; APTOS thresholds
    m = fit_lr(np.vstack([Z[apt_idx], Z[cal]]), np.concatenate([y[apt_idx], yc]), init=base)
    P = proba3(m, Z[test])
    d, s = set_decide(P, q_aptos)
    r = metrics(d, yt, s)
    r.update(auroc_ref=auroc(yt >= 1, P[:, 1:].sum(1)), auroc_urg=auroc(yt == 2, P[:, 2]))
    out.append({"setting": setting, "k": k, "draw": b, "rule": "B7b", **r})
    return out


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="C2/E2: class-wise thresholds refitted at new clinics.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", default="../data_512")
    ap.add_argument("--features", default="features/resnet50_224.npz", help="relative to --data")
    ap.add_argument("--out", default="../results/e2")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=0.1, help="C2-PAC: allowed chance of a clinic below target")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--retrain-draws", type=int, default=20, help="draws per k for B7a/B7b (slow)")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    alpha = args.alpha

    man = pd.read_csv(Path(args.data) / "manifest.csv")
    with np.load(Path(args.data) / args.features) as z:
        X, uid = z["X"], z["uid"]
    assert (uid == man["uid"].astype(str).to_numpy()).all(), "features do not match manifest.csv"
    y = man["ref3"].to_numpy(int)
    group = pd.factorize(man["dataset"] + "/" + man["group_id"].astype(str))[0]
    apt_idx = np.flatnonzero((man["dataset"] == "aptos").to_numpy())
    Z = StandardScaler().fit(X[apt_idx]).transform(X)

    # ---- the two APTOS models ----
    gdf = man.iloc[apt_idx].reset_index(drop=True)
    cam = camera_groups(man).to_numpy()
    models = {}
    for name, w in (("plain", None), ("dec", decorrelating_weights(y[apt_idx], cam[apt_idx]))):
        model, iso, oof_raw, oof_cal = train_aptos_model(Z[apt_idx], y[apt_idx], gdf, w, args.seed + 3000)
        P = proba3(model, Z)
        Pc = isotonic_apply(iso, P)
        P[apt_idx], Pc[apt_idx] = oof_raw, oof_cal        # APTOS photos: out-of-fold, never their own model
        models[name] = {"model": model, "iso": iso, "P": P, "P_cal": Pc,
                        "q_aptos": classwise_thresholds(oof_raw, y[apt_idx], alpha),
                        "q_aptos_cal": classwise_thresholds(oof_cal, y[apt_idx], alpha)}
        print(f"APTOS model ({name}) trained ({time.time() - t0:.0f} s)")
    pd.DataFrame({"uid": man["uid"], **{f"{n}_{kind}{c}": models[n][key][:, c] for n in models
                                        for kind, key in (("p", "P"), ("c", "P_cal")) for c in range(3)}}
                 ).to_csv(out / "aptos_model.csv.gz", index=False, float_format="%.6g")

    is_ds = {ds: (man["dataset"] == ds).to_numpy() for ds in ("idrid", "messidor2")}
    split = man["split"].to_numpy()
    pools = {"idrid": Pool(np.flatnonzero(is_ds["idrid"]), np.flatnonzero(is_ds["idrid"]), group),
             "idrid_split": Pool(np.flatnonzero(is_ds["idrid"] & (split == "train")),
                                 np.flatnonzero(is_ds["idrid"] & (split == "test")), group),
             "messidor2": Pool(np.flatnonzero(is_ds["messidor2"]), np.flatnonzero(is_ds["messidor2"]), group)}

    # model quality at each clinic (whole clinic)
    quality = []
    for ds in ("idrid", "messidor2"):
        s = is_ds[ds]
        for name in models:
            P = models[name]["P"][s]
            quality.append({"dataset": ds, "model": name, "n": int(s.sum()),
                            "share_0": np.mean(y[s] == 0), "share_1": np.mean(y[s] == 1), "share_2": np.mean(y[s] == 2),
                            "auroc_ref": auroc(y[s] >= 1, P[:, 1:].sum(1)), "auroc_urg": auroc(y[s] == 2, P[:, 2])})
    quality = pd.DataFrame(quality)

    # the clinic's own eyes of each class: the population that C2-PAC's promise is about
    # (on IDRiD's official split, its test photos)
    population = {s: (pool.test_idx if s == "idrid_split" else np.flatnonzero(is_ds[s])) for s, pool in pools.items()}
    by_class = {s: [idx[y[idx] == c] for c in range(3)] for s, idx in population.items()}

    def clinic_cov(P, q, setting):
        return {f"clinic_cov_{c}": float(np.mean(1.0 - P[by_class[setting][c], c] <= q[c])) for c in range(3)}

    # ---- the fast rules: every draw ----
    rows, retrain_tasks = [], []
    rng = np.random.default_rng(args.seed)
    for setting, pool in pools.items():
        for k in [k for k in K_GRID if k <= K_MAX[setting]]:
            for b in range(args.draws):
                cal, test = pool.draw(k, rng)
                yc, yt = y[cal], y[test]
                base = {"setting": setting, "k": k, "draw": b, "k_real": len(cal), "n_test": len(test),
                        **{f"n_cal_{c}": int(np.sum(yc == c)) for c in range(3)}}
                for name, tag in (("plain", ""), ("dec", "-dec")):
                    P = models[name]["P"]
                    d, s = set_decide(P[test], models[name]["q_aptos"])
                    rows.append({**base, "rule": "B6" + tag, **metrics(d, yt, s)})
                    q = classwise_thresholds(P[cal], yc, alpha)
                    d, s = set_decide(P[test], q)
                    rows.append({**base, "rule": "C2" + tag, **metrics(d, yt, s), **clinic_cov(P, q, setting),
                                 **{f"finite_{c}": float(np.isfinite(q[c])) for c in range(3)}})
                Pc = models["plain"]["P_cal"]
                d, s = set_decide(Pc[test], models["plain"]["q_aptos_cal"])
                rows.append({**base, "rule": "B6-cal", **metrics(d, yt, s)})
                q = classwise_thresholds(Pc[cal], yc, alpha)
                d, s = set_decide(Pc[test], q)
                rows.append({**base, "rule": "C2-cal", **metrics(d, yt, s),
                             **{f"finite_{c}": float(np.isfinite(q[c])) for c in range(3)}})
                P = models["plain"]["P"]
                q = pac_thresholds(P[cal], yc, alpha, args.delta)
                d, s = set_decide(P[test], q)
                rows.append({**base, "rule": "C2-PAC", **metrics(d, yt, s), **clinic_cov(P, q, setting),
                             **{f"finite_{c}": float(np.isfinite(q[c])) for c in range(3)}})
                d = b3_decide(P[test], b3_thresholds(P[cal], yc, alpha))
                rows.append({**base, "rule": "B3-k", **metrics(d, yt)})
                d, s = set_decide(P[test], marginal_threshold(P[cal], yc, alpha))
                rows.append({**base, "rule": "B4-k", **metrics(d, yt, s)})
                if k in K_RETRAIN and b < args.retrain_draws:
                    retrain_tasks.append((setting, k, b, cal, test))
        print(f"{setting}: threshold rules done ({time.time() - t0:.0f} s)")

    # ---- retraining (B7a, B7b): fewer draws, in parallel ----
    m = models["plain"]
    res = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(retrain_task)(s, k, b, cal, test, Z, y, apt_idx, m["model"], m["iso"], m["q_aptos"], len(apt_idx))
        for s, k, b, cal, test in retrain_tasks)
    c2_rows = {(x["setting"], x["k"], x["draw"]): x for x in rows if x["rule"] == "C2"}
    for r in res:
        for row in r:
            base = c2_rows[(row["setting"], row["k"], row["draw"])]
            rows.append({**{c: base[c] for c in ("k_real", "n_test", "n_cal_0", "n_cal_1", "n_cal_2")}, **row})
    print(f"retraining done: {len(retrain_tasks)} draws ({time.time() - t0:.0f} s)")

    draws = pd.DataFrame(rows)
    draws.to_csv(out / "draws.csv", index=False, float_format="%.5f")
    met = ["ref_home", "urgent_urgent", "healthy_home", "uncertain", "referred", "urgent_decisions", "nobody_home",
           "coverage_0", "coverage_1", "coverage_2", "clinic_cov_0", "clinic_cov_1", "clinic_cov_2",
           "set_size", "finite_0", "finite_1", "finite_2",
           "auroc_ref", "auroc_urg"]
    met = [c for c in met if c in draws]
    g = draws.groupby(["setting", "rule", "k"])
    curves = g[met].mean().add_suffix("_mean")
    curves = curves.join(g[met].quantile(0.05).add_suffix("_p05")).join(g[met].quantile(0.95).add_suffix("_p95"))
    curves["targets_met"] = g.apply(lambda x: np.mean((x["ref_home"] <= alpha) & (x["urgent_urgent"] >= 1 - alpha)),
                                    include_groups=False)
    curves["n_draws"] = g.size()
    curves = curves.reset_index()
    curves.to_csv(out / "curves.csv", index=False, float_format="%.5f")

    write_summary(out, args, draws, curves, quality, models, man, y, apt_idx, time.time() - t0)
    plot_targets(curves, alpha, out / "e2_targets.png")
    plot_retrain(curves, alpha, out / "e2_retrain.png")
    plot_extensions(curves, alpha, out / "e2_extensions.png")
    print(f"wrote {out / 'summary.md'} ({time.time() - t0:.0f} s)")


# --------------------------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------------------------

def smallest_k(c, cond):
    ks = c.loc[cond(c), "k"]
    return int(ks.min()) if len(ks) else None


def write_summary(out, args, draws, curves, quality, models, man, y, apt_idx, elapsed):
    a = args.alpha
    C = curves.set_index(["setting", "rule", "k"])

    def cell(setting, rule, k, m, pct=True, rng=True):
        try:
            r = C.loc[(setting, rule, k)]
        except KeyError:
            return "–"
        f = 100 if pct else 1000
        s = f"{f * r[m + '_mean']:.{1 if pct else 0}f}"
        if rng:
            s += f" ({f * r[m + '_p05']:.0f}–{f * r[m + '_p95']:.0f})"
        return s

    qa = models["plain"]["q_aptos"]
    L = ["# C2 and E2: refitting the three thresholds at a new clinic", "",
         "Model: logistic regression trained on all 3,662 APTOS photos (standardised ResNet-50 features, "
         "C = 0.003); the conformal scores use its raw probabilities (section 7 explains why). "
         "It never sees an IDRiD or Messidor-2 photo. For every k, "
         f"{args.draws} times: k labelled photos of the clinic are drawn by whole groups, the rule is refitted on "
         "them and applied to the clinic's other photos. Numbers are means over the draws, with the 5th–95th "
         "percentile of the draws in brackets: the spread a single clinic should expect. Targets: at most "
         f"{100 * a:.0f} % of referable eyes sent home, at least {100 * (1 - a):.0f} % of urgent eyes referred "
         "urgently.", "",
         f"B6 uses C1's thresholds from APTOS (the out-of-fold raw probabilities of all APTOS photos): "
         f"q = {qa[0]:.3f} / {qa[1]:.3f} / {qa[2]:.3f}, i.e. a class is in the set when its probability is at least "
         f"{1 - qa[0]:.3f} / {1 - qa[1]:.3f} / {1 - qa[2]:.3f}.", ""]

    # 1. the clinics
    L += ["## 1. The model at the new clinics", "",
          "| Clinic | Photos | No referral / routine / urgent | AUROC referable | AUROC urgent |", "|---|---|---|---|---|"]
    for _, r in quality[quality["model"] == "plain"].iterrows():
        L.append(f"| {SHORT[r['dataset']]} | {r['n']:,} | {100 * r['share_0']:.1f} / {100 * r['share_1']:.1f} / "
                 f"{100 * r['share_2']:.1f} % | {r['auroc_ref']:.3f} | {r['auroc_urg']:.3f} |")
    L.append("")

    # 2. main results per setting
    L += ["## 2. C2 against no recalibration (B6)", "",
          "Referable eyes sent home, urgent eyes referred urgently, healthy eyes sent home and uncertain in %; "
          "referred per 1,000 at the clinic's own prevalence. \"Urgent threshold finite\": share of draws with "
          f"at least {math.ceil(1 / a - 1)} urgent photos among the k, so that urgency is not simply assumed for "
          "everyone. With fewer, the targets are met only because every eye is referred (healthy eyes sent home: 0).",
          ""]
    for setting, title in SETTINGS.items():
        L += [f"### {title}", "",
              "| k | Rule | Referable eyes sent home | Urgent eyes referred urgently | Healthy eyes sent home | "
              "Uncertain | Referred per 1,000 | Urgent threshold finite | Both targets met (share of draws) |",
              "|---|---|---|---|---|---|---|---|---|"]
        ks = sorted(curves.loc[(curves["setting"] == setting) & (curves["rule"] == "C2"), "k"])
        r6 = C.loc[(setting, "B6", ks[-1])]
        L.append(f"| none | B6 | {cell(setting, 'B6', ks[-1], 'ref_home')} | {cell(setting, 'B6', ks[-1], 'urgent_urgent')} "
                 f"| {cell(setting, 'B6', ks[-1], 'healthy_home')} | {cell(setting, 'B6', ks[-1], 'uncertain', rng=False)} "
                 f"| {cell(setting, 'B6', ks[-1], 'referred', pct=False, rng=False)} | – | {100 * r6['targets_met']:.0f} % |")
        for k in ks:
            r = C.loc[(setting, "C2", k)]
            L.append(f"| {k} | C2 | {cell(setting, 'C2', k, 'ref_home')} | {cell(setting, 'C2', k, 'urgent_urgent')} | "
                     f"{cell(setting, 'C2', k, 'healthy_home')} | {cell(setting, 'C2', k, 'uncertain', rng=False)} | "
                     f"{cell(setting, 'C2', k, 'referred', pct=False, rng=False)} | {100 * r['finite_2_mean']:.0f} % | "
                     f"{100 * r['targets_met']:.0f} % |")
        L.append("")
        if setting == "idrid_split":
            L += [split_note(man, models["plain"]["P"], y), ""]

    # 3. how many local photos
    L += ["## 3. How many labelled local photos are needed?", "",
          f"A finite threshold for a class needs at least {math.ceil(1 / a - 1)} calibration photos of that class "
          f"(α = {a:g}); before that, the class is in every set. The urgent class is the rarest at both clinics, so it "
          "decides. Theory: binomial draws with the clinic's class shares. Observed: the first k at which at least 90 % "
          "of the draws have a finite urgent threshold, and what C2 does there and at the largest k.", "",
          "| Setting | Urgent share | Theory: k for 9 urgent photos on average / with 90 % chance | Observed k | "
          "At that k: referable sent home / urgent referred urgently / healthy sent home / uncertain | At the largest k |",
          "|---|---|---|---|---|---|"]
    for setting in SETTINGS:
        c = curves[(curves["setting"] == setting) & (curves["rule"] == "C2")].sort_values("k").set_index("k")
        ds = "idrid" if setting.startswith("idrid") else "messidor2"
        pool_mask = np.array(man["dataset"] == ds)
        if setting == "idrid_split":
            pool_mask = pool_mask & np.array(man["split"] == "train")
        share = np.mean(y[pool_mask] == 2)
        need = math.ceil(1 / a - 1)
        k_mean = math.ceil(need / share)
        k_90 = next(k for k in range(need, 5000) if binom_sf(need - 1, k, share) >= 0.9)
        ok = c.index[c["finite_2_mean"] >= 0.9]
        k_obs = int(ok.min()) if len(ok) else None

        def four(k):
            r = c.loc[k]
            return (f"{100 * r['ref_home_mean']:.1f} / {100 * r['urgent_urgent_mean']:.1f} / "
                    f"{100 * r['healthy_home_mean']:.1f} / {100 * r['uncertain_mean']:.1f}")
        L.append(f"| {SHORT[setting]} | {100 * share:.1f} % | {k_mean} / {k_90} | {k_obs if k_obs else 'not reached'} | "
                 f"{four(k_obs) if k_obs else '–'} | k = {int(c.index.max())}: {four(int(c.index.max()))} |")
    L += ["", "Answer for the report: at IDRiD about 40–50 labelled photos make the rule informative and keep both "
          "targets (random split); at Messidor-2 it takes about 200–300, because only 1 eye in 16 is urgent. "
          "More photos then tighten the thresholds only slowly: the model's ranking at the clinic limits how many "
          "healthy eyes can be sent home.", ""]

    # 4. conformal against tuning and against one threshold
    L += ["## 4. Why conformal, and why one threshold per class, at a new clinic", "",
          "Same draws. B3-k refits the tuned thresholds (plain empirical 90 % points, no finite-sample correction); "
          "B4-k refits standard conformal prediction (one threshold for all classes).", ""]
    for setting in SETTINGS:
        L += [f"**{SHORT[setting]}**", "",
              "| k | Urgent referred urgently: C2 / B3-k / B4-k | Referable sent home: C2 / B3-k / B4-k | "
              "Healthy sent home: C2 / B3-k / B4-k |", "|---|---|---|---|"]
        for k in [10, 20, 30, 50, 100, 200, 400]:
            if (setting, "C2", k) not in C.index:
                continue
            L.append(f"| {k} | " + " | ".join(
                " / ".join(cell(setting, r, k, m, rng=False) for r in ("C2", "B3-k", "B4-k"))
                for m in ("urgent_urgent", "ref_home", "healthy_home")) + " |")
        L.append("")
    L += ["B3-k's empirical threshold covers only n/(n + 1) of the positives it was tuned on, on average, and less "
          "when n is small: with one urgent calibration photo, half of the clinic's urgent eyes fall below it. C2's "
          "⌈(n + 1)(1 − α)⌉ rule refuses to promise anything until n is large enough.", ""]

    # 5. recalibrate or retrain
    L += ["## 5. Recalibrate or retrain? (B7)", "",
          f"B7a retrains logistic regression on the k local photos only (penalty scaled so each photo counts as in "
          f"the APTOS model; most likely class). B7b retrains it on APTOS plus the k local photos (equal weight per "
          f"photo) and keeps the APTOS thresholds. {args.retrain_draws} draws per k (the first draws of section 2). "
          "AUROC: referable, on the clinic's test photos.", "",
          "| Setting | k | Rule | Referable sent home | Urgent referred urgently | Healthy sent home | AUROC referable |",
          "|---|---|---|---|---|---|---|"]
    for setting in SETTINGS:
        for k in K_RETRAIN:
            if (setting, "B7a", k) not in C.index:
                continue
            for rule in ("C2", "B7a", "B7b"):
                au = C.loc[(setting, rule, k)].get("auroc_ref_mean", np.nan) if rule != "C2" else \
                    quality.set_index(["dataset", "model"]).loc[("idrid" if setting.startswith("idrid") else "messidor2",
                                                                "plain"), "auroc_ref"]
                L.append(f"| {SHORT[setting]} | {k} | {rule} | {cell(setting, rule, k, 'ref_home', rng=False)} | "
                         f"{cell(setting, rule, k, 'urgent_urgent', rng=False)} | "
                         f"{cell(setting, rule, k, 'healthy_home', rng=False)} | {au:.3f} |")
    L.append("")

    # 6. additional work
    L += ["## 6. Additional work (beyond the proposal)", "",
          f"### 6a. C2-PAC: a promise for each clinic, not only on average (δ = {args.delta:g})", "",
          "C2's guarantee is an average over clinics' calibration draws: a single clinic's coverage can fall below 90 %. "
          f"C2-PAC picks, per class, the threshold whose coverage is at least {100 * (1 - a):.0f} % with probability at "
          f"least {100 * (1 - args.delta):.0f} % for the clinic's own draw (the coverage of the j-th smallest of n "
          f"scores follows Beta(j, n + 1 − j)). It needs at least {_min_n_pac(a, args.delta)} photos of a class "
          "before that class can be left out of any set.", "",
          "Share of draws in which a class's coverage falls below 90 %, measured on the test photos and over all "
          "the clinic's eyes of that class (the population the promise is about; on IDRiD's official split, its test "
          "photos), C2 / C2-PAC:", "",
          "| Setting | k | Urgent, test photos | Urgent, all the clinic's urgent eyes | Routine, test photos | "
          "Routine, all the clinic's routine eyes | Healthy sent home: C2 / C2-PAC |", "|---|---|---|---|---|---|---|"]
    worst = {"C2": 0.0, "C2-PAC": 0.0}
    for setting in SETTINGS:
        for k in [50, 100, 200, 300, 400]:
            if (setting, "C2", k) not in C.index:
                continue
            sub = draws[(draws["setting"] == setting) & (draws["k"] == k)]
            below = {(r, col): np.mean(sub.loc[sub["rule"] == r, col] < 1 - a)
                     for r in ("C2", "C2-PAC") for col in ("coverage_1", "coverage_2", "clinic_cov_1", "clinic_cov_2")}
            if setting != "idrid_split":
                for r in worst:
                    worst[r] = max(worst[r], below[(r, "clinic_cov_1")], below[(r, "clinic_cov_2")])
            pair = lambda col: f"{100 * below[('C2', col)]:.0f} % / {100 * below[('C2-PAC', col)]:.0f} %"
            L.append(f"| {SHORT[setting]} | {k} | {pair('coverage_2')} | {pair('clinic_cov_2')} | {pair('coverage_1')} | "
                     f"{pair('clinic_cov_1')} | {cell(setting, 'C2', k, 'healthy_home', rng=False)} / "
                     f"{cell(setting, 'C2-PAC', k, 'healthy_home', rng=False)} |")
    sub_c2 = draws[(draws["setting"] == "idrid") & (draws["k"] == 200) & (draws["rule"] == "C2")]
    n_urg = int(np.sum(y[(man["dataset"] == "idrid").to_numpy()] == 2)) - sub_c2["n_cal_2"].mean()
    L += ["", f"The promise is about the clinic's eyes of each class as a whole. The test photos are what is left after "
          f"the draw: a finite sample (about {n_urg:.0f} urgent eyes at IDRiD for k = 200) whose measured coverage "
          "scatters around the clinic's, and leans low when the draw happened to take the easier eyes. Over all the "
          f"clinic's eyes (random splits), C2-PAC falls below 90 % in at most {100 * worst['C2-PAC']:.0f} % of draws "
          f"(promised: at most {100 * args.delta:.0f} %), C2 in up to {100 * worst['C2']:.0f} %. On IDRiD's official "
          "split the shift between its two parts (section 2) breaks the promise of both.", ""]
    L += ["", "### 6b. The camera-decorrelated model at new clinics (additional camera study)", "",
          "The same experiment with the logistic regression trained with camera-decorrelated weights "
          "(w = P(class) / P(class | camera group)).", "",
          "| Clinic | AUROC referable: plain / decorrelated | Rule | k | Referable sent home: plain / decorrelated | "
          "Urgent referred urgently: plain / decorrelated | Healthy sent home: plain / decorrelated |",
          "|---|---|---|---|---|---|---|"]
    qd = quality.set_index(["dataset", "model"])
    for setting in ("idrid", "messidor2"):
        au = f"{qd.loc[(setting, 'plain'), 'auroc_ref']:.3f} / {qd.loc[(setting, 'dec'), 'auroc_ref']:.3f}"
        kmax = K_MAX[setting]
        for rule, k in (("B6", kmax), ("C2", 50), ("C2", 100), ("C2", 200), ("C2", kmax)):
            if (setting, rule, k) not in C.index:
                continue
            L.append(f"| {SHORT[setting]} | {au} | {rule if rule == 'C2' else 'B6 (no recalibration)'} | "
                     f"{'–' if rule == 'B6' else k} | "
                     + " | ".join(f"{cell(setting, rule, k, m, rng=False)} / {cell(setting, rule + '-dec', k, m, rng=False)}"
                                  for m in ("ref_home", "urgent_urgent", "healthy_home")) + " |")
    # 7. raw against calibrated scores
    L += ["", "## 7. Why the conformal scores use raw probabilities", "",
          "E1's isotonic map is learnt on APTOS. It is a step function whose lowest step is exactly 0, so "
          "every photo below a certain raw score gets probability 0 for that class, and their order is lost. "
          "At APTOS few sick eyes sit there; at Messidor-2 many do.", "",
          "| Clinic | Routine eyes with calibrated P(routine) = 0 | Urgent eyes with calibrated P(urgent) = 0 |",
          "|---|---|---|"]
    Pc = models["plain"]["P_cal"]
    for ds in ("aptos", "idrid", "messidor2"):
        s = (man["dataset"] == ds).to_numpy()
        L.append(f"| {ds if ds == 'aptos' else SHORT[ds]} | {100 * np.mean(Pc[s & (y == 1), 1] == 0):.1f} % | "
                 f"{100 * np.mean(Pc[s & (y == 2), 2] == 0):.1f} % |")
    L += ["", "When more than a tenth of a class's calibration photos tie at 0, the conformal threshold of that "
          "class is 1: the class enters every set, and every eye is referred. Same draws, Messidor-2:", "",
          "| k | Healthy sent home: C2 raw / calibrated | Referable sent home: raw / calibrated | Urgent referred "
          "urgently: raw / calibrated |", "|---|---|---|---|"]
    for k in (100, 200, 300, 400):
        if ("messidor2", "C2-cal", k) in C.index:
            L.append(f"| {k} | " + " | ".join(
                f"{cell('messidor2', 'C2', k, m, rng=False)} / {cell('messidor2', 'C2-cal', k, m, rng=False)}"
                for m in ("healthy_home", "ref_home", "urgent_urgent")) + " |")
    L += ["", "Conformal coverage holds for any score, calibrated or not; the score only decides how useful the "
          "sets are. The calibrated probabilities stay useful where they are read as risks (C3).", ""]

    L += ["## Notes", "",
          "- Coverage check: with random splits (IDRiD, Messidor-2), C2's per-class coverage averaged over the "
          f"draws is at least {_min_cov(curves[curves['setting'] != 'idrid_split'], 'C2')} % at every k (target "
          f"{100 * (1 - a):.0f} %; {args.draws} draws leave about ±0.3 points of simulation noise), as conformal "
          "theory promises when the calibration and test photos are exchangeable. On IDRiD's official split it falls to "
          f"{_min_cov(curves[curves['setting'] == 'idrid_split'], 'C2')} % (section 2).",
          "- Group sampling: copies of one photo and the two eyes of a Messidor-2 exam are drawn together, and a "
          "test photo whose group was drawn is left out (this removes the IDRiD photo that sits in both official "
          "splits when its twin is drawn).",
          f"- Made {time.strftime('%Y-%m-%d %H:%M')} by e2_new_clinics.py ({elapsed:.0f} s).", ""]
    (out / "summary.md").write_text("\n".join(L))


def binom_sf(x, n, p):
    from scipy.stats import binom
    return binom.sf(x, n, p)


def split_note(man, P, y):
    """Do IDRiD's official training and test photos look alike to the model? (urgent eyes)"""
    from scipy.stats import mannwhitneyu
    idr = (man["dataset"] == "idrid").to_numpy()
    tr = P[idr & (man["split"] == "train").to_numpy() & (y == 2), 2]
    te = P[idr & (man["split"] == "test").to_numpy() & (y == 2), 2]
    below = np.mean(te < np.quantile(tr, 0.1))
    p = mannwhitneyu(tr, te, alternative="two-sided").pvalue
    return ("On the official split the urgent target is missed for k ≥ 40. Here the k photos come from IDRiD's "
            "training split and the test is always the same 103 photos, so nothing averages the test set out. Its "
            f"{len(te)} urgent eyes score lower than the {len(tr)} urgent eyes of the training split: median P(urgent) "
            f"{np.median(te):.2f} against {np.median(tr):.2f}; {100 * below:.0f} % of them sit below the training "
            f"split's 10th percentile, where 10 % would be expected (Mann–Whitney p = {p:.2f}). With so few eyes this "
            "may be chance, but it is the warning that matters in practice: the guarantee holds only if the "
            "calibration photos are like the photos the rule is then used on.")


def _min_n_pac(alpha, delta):
    n = 1
    while pac_index(n, alpha, delta) == 0:
        n += 1
    return n


def _min_cov(curves, rule):
    c = curves[curves["rule"] == rule]
    return f"{100 * np.nanmin(c[['coverage_0_mean', 'coverage_1_mean', 'coverage_2_mean']].to_numpy()):.1f}"


# --------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"C2": "#2a78d6", "B3-k": "#eb6834", "B4-k": "#1baf7a", "B6": "#52514e", "B7a": "#eda100", "B7b": "#e87ba4",
       "C2-PAC": "#6250d6", "C2-dec": "#1baf7a", "B6-dec": "#1baf7a"}


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=7.5, length=0)


def _grid_figure(curves, alpha, rules, path, title, band_rule="C2"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("ref_home", f"Referable eyes sent home (%)\ntarget ≤ {100 * alpha:.0f}", 100 * alpha),
            ("urgent_urgent", f"Urgent eyes referred urgently (%)\ntarget ≥ {100 * (1 - alpha):.0f}", 100 * (1 - alpha)),
            ("healthy_home", "Healthy eyes sent home (%)\nhigher = more useful", None)]
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 8.2), dpi=200, sharex="col")
    for j, setting in enumerate(SETTINGS):
        for i, (m, lab, target) in enumerate(rows):
            ax = axes[i, j]
            for rule in rules:
                c = curves[(curves["setting"] == setting) & (curves["rule"] == rule)].sort_values("k")
                if c.empty:
                    continue
                x = c["k"].to_numpy()
                if rule.startswith("B6"):
                    ax.axhline(100 * c[m + "_mean"].mean(), color=COL[rule], lw=1.4, ls=(0, (1.5, 1.5)),
                               label=RULE_NAMES[rule])
                    continue
                if rule == band_rule:
                    ax.fill_between(x, 100 * c[m + "_p05"], 100 * c[m + "_p95"], color=COL[rule], alpha=0.13, lw=0)
                ax.plot(x, 100 * c[m + "_mean"], "-o", color=COL[rule], lw=1.8, ms=3.5, label=RULE_NAMES[rule])
            if target is not None:
                ax.axhline(target, color=INK, lw=0.9, ls=(0, (4, 3)))
            ax.set_xscale("log")
            ax.set_xticks([10, 20, 50, 100, 200, 400])
            ax.set_xticklabels(["10", "20", "50", "100", "200", "400"])
            ax.minorticks_off()
            ax.set_ylim(-3, 103)
            _style(ax)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8, color=INK2)
            if i == 0:
                ax.set_title(SHORT[setting], fontsize=9.5, color=INK, loc="left")
            if i == 2:
                ax.set_xlabel("k labelled local photos", fontsize=8, color=INK2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    fig.legend(seen.values(), seen.keys(), frameon=False, fontsize=8, ncol=min(4, len(seen)), loc="lower left",
               bbox_to_anchor=(0.01, -0.005))
    fig.suptitle(title, fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_targets(curves, alpha, path):
    _grid_figure(curves, alpha, ["B6", "C2", "B3-k", "B4-k"], path,
                 "E2: refitting the thresholds at a new clinic from k labelled photos (mean over draws; band: "
                 "5–95 % of draws for C2)")


def plot_retrain(curves, alpha, path):
    _grid_figure(curves, alpha, ["B6", "C2", "B7a", "B7b"], path,
                 "E2: recalibrate (C2) or retrain the model (B7a: local photos only; B7b: APTOS + local photos)")


def plot_extensions(curves, alpha, path):
    _grid_figure(curves, alpha, ["B6", "B6-dec", "C2", "C2-dec", "C2-PAC"], path,
                 "Additional work: a promise for each clinic (C2-PAC) and the camera-decorrelated model")


if __name__ == "__main__":
    main()
