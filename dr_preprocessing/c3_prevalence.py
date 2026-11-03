#!/usr/bin/env python3
"""
c3_prevalence.py - C3 and E3: the referral rule at clinics where sick eyes are rarer (or commoner) than in APTOS.

Part A, APTOS resampled (label shift only, by construction). The out-of-fold decisions of C1 (results/c1) are
resampled to 5 %, 12.5 % and 25 % referable eyes, keeping APTOS's mix of routine and urgent eyes and of grades.
Questions:
  - does each class keep its guarantee when only prevalence changes [11], and what happens to the marginal
    guarantee of standard conformal prediction (B4)?
  - can the workload be predicted from the clinic's unlabelled photos? Workload: referrals, urgent referrals and
    uncertain eyes per 1,000, missed referable eyes per 1,000, and the PPV of a referral.
Each repeat splits APTOS by group into two halves. Half A gives each class's decision rates r_y; half B,
resampled, is the clinic.
Forecast: per 1,000 = 1000 * sum_y pi_y * r_y(decision); PPV = sum_{y >= 1} pi_y r_y(refer) / sum_y pi_y r_y(refer).

Class shares pi_y of the clinic:
  B8    APTOS's shares (no prevalence correction)
  EM    Saerens et al. [12]: the model's calibrated probabilities re-weighted until the shares they imply
        agree with the shares used to re-weight them (from unlabelled photos)
  BBSE  black-box shift estimation (additional; Lipton et al. 2018): the share of each predicted class,
        corrected with the model's confusion matrix on APTOS
  B9    the true shares (oracle)
Bayes-adjusted risks: p'(y|x) proportional to p(y|x) * pi_y / pi_APTOS,y, checked by their calibration (ECE).

Part B, the real clinics: the same estimates at IDRiD and Messidor-2 (unlabelled photos, the all-APTOS model of
results/e2), and the workload of C2 (thresholds refitted on k labelled local photos) forecast from those k
photos.

Outputs (in --out, default ../results/c3): summary.md, workload_aptos.csv, prevalence_clinics.csv,
workload_clinics.csv, c3_workload.png, c3_prevalence.png

Usage:
  python c3_prevalence.py --data ../data_512 --results ../results
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from c1_referral import HOME, ROUTINE, URGENT, UNCERTAIN, classwise_thresholds, decide_from_sets
from e2_new_clinics import Pool

DEC = {"no referral": HOME, "routine": ROUTINE, "urgent": URGENT, "uncertain": UNCERTAIN}
LEVELS = [0.05, 0.125, 0.25]
PI_METHODS = ["B8", "EM", "BBSE", "B9"]
PI_NAMES = {"B8": "B8 APTOS shares (no correction)", "EM": "EM from unlabelled photos [12]",
            "BBSE": "BBSE from unlabelled photos (additional)", "B9": "B9 true shares (oracle)",
            "CC": "share of most likely classes", "PCC": "mean probability", "local": "k labelled local photos"}
WORK = ["referred", "urgent_dec", "uncertain_dec", "missed", "ppv"]
WORK_NAMES = {"referred": "Referred per 1,000", "urgent_dec": "Urgent referrals per 1,000",
              "uncertain_dec": "Uncertain per 1,000", "missed": "Referable eyes sent home per 1,000",
              "ppv": "PPV of a referral (%)"}


# --------------------------------------------------------------------------------------------
# prevalence estimation
# --------------------------------------------------------------------------------------------

def em_shares(P, prior_src, tol=1e-10, max_iter=10000):
    """Saerens, Latinne & Decaestecker (2002). P: calibrated probabilities under the source shares."""
    pi = prior_src.copy()
    for _ in range(max_iter):
        W = P * (pi / prior_src)
        W /= W.sum(1, keepdims=True)
        new = W.mean(0)
        if np.abs(new - pi).max() < tol:
            pi = new
            break
        pi = new
    return pi


def bbse_shares(pred_src, y_src, pred_tgt, prior_src, k=3):
    """Lipton, Wang & Smola (2018): C[i, j] = P(pred = i, y = j) on the source; mu = P(pred = i) on the target;
    w = C^-1 mu are the ratios pi_tgt / pi_src."""
    C = np.zeros((k, k))
    np.add.at(C, (pred_src, y_src), 1)
    C /= len(y_src)
    mu = np.bincount(pred_tgt, minlength=k) / len(pred_tgt)
    w = np.linalg.lstsq(C, mu, rcond=None)[0]
    pi = np.clip(w * prior_src, 0, None)
    return pi / pi.sum()


def adjust(P, pi, prior_src):
    W = P * (pi / prior_src)
    return W / W.sum(1, keepdims=True)


def ece(pos, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    gap = np.abs(np.bincount(b, pos.astype(float), bins) - np.bincount(b, p, bins))
    return gap.sum() / len(p)


# --------------------------------------------------------------------------------------------
# workload
# --------------------------------------------------------------------------------------------

def rates(d, y):
    """r[y, decision]: share of class-y eyes given each decision."""
    r = np.zeros((3, 4))
    for c in range(3):
        s = y == c
        if s.any():
            r[c] = np.bincount(d[s], minlength=4) / s.sum()
    return r


def forecast(pi, r):
    refer = r[:, ROUTINE] + r[:, URGENT]
    tot = float(pi @ refer)
    return {"referred": 1000 * tot, "urgent_dec": 1000 * float(pi @ r[:, URGENT]),
            "uncertain_dec": 1000 * float(pi @ r[:, UNCERTAIN]), "missed": 1000 * float(pi[1:] @ r[1:, HOME]),
            "ppv": 100 * float(pi[1:] @ refer[1:]) / tot if tot > 0 else np.nan}


def observed(d, y):
    ref = (d == ROUTINE) | (d == URGENT)
    return {"referred": 1000 * ref.mean(), "urgent_dec": 1000 * np.mean(d == URGENT),
            "uncertain_dec": 1000 * np.mean(d == UNCERTAIN), "missed": 1000 * np.mean((y >= 1) & (d == HOME)),
            "ppv": 100 * np.mean(y[ref] >= 1) if ref.any() else np.nan,
            "sens_ref": 100 * np.mean(d[y >= 1] != HOME), "urgent_urgent": 100 * np.mean(d[y == 2] == URGENT),
            "healthy_home": 100 * np.mean(d[y == 0] == HOME)}


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="C3/E3: prevalence and workload.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", default="../data_512")
    ap.add_argument("--results", default="../results", help="folder holding e1, c1 and e2")
    ap.add_argument("--out", default="../results/c3")
    ap.add_argument("--model", default="lr")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R = Path(args.results)
    rng = np.random.default_rng(args.seed)

    man = pd.read_csv(Path(args.data) / "manifest.csv").set_index("uid")
    apt = man[man["dataset"] == "aptos"]
    prior_src = np.bincount(apt["ref3"], minlength=3) / len(apt)

    # ---------- Part A: APTOS out-of-fold, resampled ----------
    frames = [pd.read_csv(f) for f in sorted((R / "e1" / args.model).glob("fold[0-9]*.csv.gz"))]
    oof = pd.concat(frames).query("role == 'test'").set_index("uid").loc[apt.index]
    Pc = oof[["c0", "c1", "c2"]].to_numpy()
    Pc = Pc / Pc.sum(1, keepdims=True)
    dec = pd.read_csv(R / "c1" / f"decisions_{args.model}.csv.gz").query("role == 'test'").set_index("uid").loc[apt.index]
    a_tag = f"a{args.alpha:g}"
    rules = {"C1": dec[f"C1_{a_tag}"].map(DEC).to_numpy(), "B4": dec[f"B4_{a_tag}"].map(DEC).to_numpy(),
             "B3": dec[f"B3_{a_tag}"].map(DEC).to_numpy(), "argmax": dec[f"argmax_{a_tag}"].map(DEC).to_numpy()}
    y = apt["ref3"].to_numpy(int)
    grade = apt["grade"].to_numpy(int)
    groups = pd.factorize(apt["group_id"])[0]
    pred = Pc.argmax(1)

    rows, cov_rows = [], []
    for rep in range(args.repeats):
        half = rng.permutation(groups.max() + 1) % 2
        A, B = np.flatnonzero(half[groups] == 0), np.flatnonzero(half[groups] == 1)
        r_A = {k: rates(v[A], y[A]) for k, v in rules.items()}
        pi_A = np.bincount(y[A], minlength=3) / len(A)
        nonref = B[y[B] == 0]
        for level in LEVELS + ["APTOS"]:
            if level == "APTOS":
                S = B
            else:
                n_ref = int(round(len(nonref) * level / (1 - level)))
                ref_pool = B[y[B] >= 1]
                # keep APTOS's grade mix among referable eyes: draw each referable grade in proportion
                take = []
                for g in (2, 3, 4):
                    pool = ref_pool[grade[ref_pool] == g]
                    n_g = int(round(n_ref * np.mean(grade[ref_pool] == g)))
                    take.append(rng.choice(pool, min(n_g, len(pool)), replace=False))
                S = np.concatenate([nonref] + take)
            yS = y[S]
            pi_true = np.bincount(yS, minlength=3) / len(S)
            pis = {"B8": pi_A, "EM": em_shares(Pc[S], prior_src),
                   "BBSE": bbse_shares(pred[A], y[A], pred[S], pi_A), "B9": pi_true}
            obs = {k: observed(v[S], yS) for k, v in rules.items()}
            base = {"repeat": rep, "level": level, "n": len(S), **{f"true_pi_{c}": pi_true[c] for c in range(3)}}
            for k in ("C1", "B4", "B3", "argmax"):
                rows.append({**base, "rule": k, "method": "observed", **obs[k]})
                for m, pi in pis.items():
                    rows.append({**base, "rule": k, "method": m, **forecast(pi, r_A[k]),
                                 **{f"pi_{c}": pi[c] for c in range(3)}})
            # coverage under label shift: C1 per class, B4 marginal
            for k in ("C1", "B4"):
                d = rules[k][S]
                cov_rows.append({"repeat": rep, "level": level, "rule": k, **{f"pi_{c}": pi_true[c] for c in range(3)},
                                 "sent_home_ref": 100 * np.mean(d[yS >= 1] == HOME),
                                 "urgent_urgent": 100 * np.mean(d[yS == 2] == URGENT)})
            # risks: calibration of P(referable) before and after the Bayes adjustment
            pos = yS >= 1
            for m in ("none", "EM", "B9"):
                Q = Pc[S] if m == "none" else adjust(Pc[S], pis[m], prior_src)
                rows.append({**base, "rule": "risk", "method": m, "ece_ref": ece(pos, Q[:, 1:].sum(1)),
                             "mean_risk_ref": Q[:, 1:].sum(1).mean()})
        if rep % 50 == 49:
            print(f"part A: {rep + 1} repeats ({time.time() - t0:.0f} s)")
    work = pd.DataFrame(rows)
    work.to_csv(out / "workload_aptos.csv", index=False, float_format="%.5f")

    # ---------- Part B: the real clinics ----------
    e2 = pd.read_csv(R / "e2" / "aptos_model.csv.gz").set_index("uid")
    P_raw_all = e2[[f"plain_p{c}" for c in range(3)]]
    P_cal_all = e2[[f"plain_c{c}" for c in range(3)]]
    src = apt.index
    P_src_cal = P_cal_all.loc[src].to_numpy()             # APTOS: out-of-fold, cross-fitted calibration
    pred_src = P_src_cal.argmax(1)
    prev_rows, clinic_rows = [], []
    for ds, ks in (("idrid", (50, 100, 200)), ("messidor2", (100, 200, 400))):
        sub = man[man["dataset"] == ds]
        yy = sub["ref3"].to_numpy(int)
        Pc_t = P_cal_all.loc[sub.index].to_numpy()
        Pr_t = P_raw_all.loc[sub.index].to_numpy()
        truth = np.bincount(yy, minlength=3) / len(yy)
        est = {"B8": prior_src, "CC": np.bincount(Pc_t.argmax(1), minlength=3) / len(yy), "PCC": Pc_t.mean(0),
               "EM": em_shares(Pc_t, prior_src), "BBSE": bbse_shares(pred_src, y, Pc_t.argmax(1), prior_src),
               "B9": truth}
        for m, pi in est.items():
            prev_rows.append({"dataset": ds, "method": m, "k": 0, **{f"pi_{c}": pi[c] for c in range(3)}})
        # labelled local photos: shares from k photos, and the workload of C2 refitted on them
        grp = pd.factorize(sub["group_id"])[0]
        pool = Pool(np.arange(len(yy)), np.arange(len(yy)), grp)    # whole groups, as in E2
        for k in ks:
            for rep in range(args.repeats):
                cal, test = pool.draw(k, rng)
                pi_local = np.bincount(yy[cal], minlength=3) / len(cal)
                prev_rows.append({"dataset": ds, "method": "local", "k": k, "repeat": rep,
                                  **{f"pi_{c}": pi_local[c] for c in range(3)}})
                q = classwise_thresholds(Pr_t[cal], yy[cal], args.alpha)
                d_cal = decide_from_sets((1 - Pr_t[cal]) <= q[None, :])
                d_test = decide_from_sets((1 - Pr_t[test]) <= q[None, :])
                r_loc = rates(d_cal, yy[cal])
                pis = {"local": pi_local, "EM": em_shares(Pc_t[test], prior_src), "B8": prior_src,
                       "B9": np.bincount(yy[test], minlength=3) / len(test)}
                base = {"dataset": ds, "k": k, "repeat": rep}
                clinic_rows.append({**base, "method": "observed", **observed(d_test, yy[test])})
                for m, pi in pis.items():
                    clinic_rows.append({**base, "method": m, **forecast(pi, r_loc)})
        print(f"part B: {ds} done ({time.time() - t0:.0f} s)")
    prev = pd.DataFrame(prev_rows)
    clin = pd.DataFrame(clinic_rows)
    prev.to_csv(out / "prevalence_clinics.csv", index=False, float_format="%.5f")
    clin.to_csv(out / "workload_clinics.csv", index=False, float_format="%.5f")

    c1m = pd.read_csv(R / "c1" / "metrics.csv")
    b4_cov = c1m[(c1m["model"] == args.model) & (c1m["alpha"] == args.alpha) & (c1m["rule"] == "B4")].iloc[0]
    b4_coverage = np.array([b4_cov[f"coverage_{c}"] for c in range(3)])
    write_summary(out, args, work, pd.DataFrame(cov_rows), prev, clin, prior_src, time.time() - t0, b4_coverage)
    plot_workload(work, out / "c3_workload.png")
    plot_prevalence(prev, out / "c3_prevalence.png")
    print(f"wrote {out / 'summary.md'} ({time.time() - t0:.0f} s)")


# --------------------------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------------------------

def lvl_name(level):
    return "APTOS (40.6 %)" if level == "APTOS" else f"{100 * float(level):g} %"


def write_summary(out, args, work, cov, prev, clin, prior_src, elapsed, b4_coverage):
    L = ["# C3 and E3: prevalence and workload", "",
         f"Part A uses the out-of-fold C1 decisions on APTOS ({args.model}, α = {args.alpha:g}). In each of "
         f"{args.repeats} repeats APTOS is split by group into two halves. Half A gives every class's decision "
         "rates; half B becomes a clinic with 5 %, 12.5 % or 25 % referable eyes (all its non-referable eyes, and "
         "referable eyes drawn by grade so that APTOS's mix of grades 2, 3 and 4 is kept). Only the prevalence "
         "changes, so this is pure label shift. Means over the repeats.", ""]

    # A1: guarantees under label shift
    c = cov.groupby(["rule", "level"], sort=False)[["sent_home_ref", "urgent_urgent", "pi_0", "pi_1", "pi_2"]].mean()
    b4 = b4_coverage
    L += ["## 1. The guarantee when only the prevalence changes", "",
          "Referable eyes sent home and urgent eyes referred urgently, in %. Each class's rate does not depend on "
          "how common the class is, so C1's per-class guarantee holds at every prevalence [11]. Last column: "
          "the share of all eyes whose true class is in B4's set (its marginal coverage), from B4's per-class "
          f"coverage on APTOS ({100 * b4[0]:.1f} / {100 * b4[1]:.1f} / {100 * b4[2]:.1f} %) weighted by the clinic's "
          "class shares.", "",
          "| Referable eyes in the clinic | C1: referable sent home | C1: urgent referred urgently | B4: referable "
          "sent home | B4: urgent referred urgently | B4: all eyes covered |", "|---|---|---|---|---|---|"]
    for level in LEVELS + ["APTOS"]:
        pis = c.loc[("B4", level), ["pi_0", "pi_1", "pi_2"]].to_numpy(float)
        L.append(f"| {lvl_name(level)} | {c.loc[('C1', level), 'sent_home_ref']:.1f} | "
                 f"{c.loc[('C1', level), 'urgent_urgent']:.1f} | {c.loc[('B4', level), 'sent_home_ref']:.1f} | "
                 f"{c.loc[('B4', level), 'urgent_urgent']:.1f} | {100 * float(pis @ b4):.1f} |")
    obs = work[(work["method"] == "observed")]
    L += ["", "B4's marginal guarantee (90 % of all eyes covered) looks better as sick eyes get rarer, because "
          "healthy eyes dominate the average; its urgent eyes stay at about two in three throughout. A marginal "
          "guarantee cannot protect a rare class.", ""]

    # A2: workload
    L += ["## 2. The workload changes a lot", "",
          "Observed in half B (C1 rule), per 1,000 eyes screened.", "",
          "| Referable eyes in the clinic | Referred | Urgent referrals | Uncertain | Referable eyes sent home | "
          "PPV of a referral (%) |", "|---|---|---|---|---|---|"]
    o = obs[obs["rule"] == "C1"].groupby("level", sort=False)[WORK].mean()
    for level in LEVELS + ["APTOS"]:
        r = o.loc[level]
        L.append(f"| {lvl_name(level)} | {r['referred']:.0f} | {r['urgent_dec']:.0f} | {r['uncertain_dec']:.0f} | "
                 f"{r['missed']:.1f} | {r['ppv']:.1f} |")
    L.append("")

    # A3: forecasts
    L += ["## 3. Forecasting the workload from unlabelled photos", "",
          "Forecast minus observed, mean over repeats (mean absolute error in brackets), for the C1 rule. B8 assumes "
          "APTOS's shares; EM and BBSE estimate the shares from the clinic's unlabelled photos; B9 knows them.", ""]
    for m_key in WORK:
        L += [f"**{WORK_NAMES[m_key]}**", "", "| Referable eyes in the clinic | Observed | " +
              " | ".join(PI_NAMES[m] for m in PI_METHODS) + " |", "|---|---|" + "---|" * len(PI_METHODS)]
        sub = work[work["rule"] == "C1"]
        for level in LEVELS + ["APTOS"]:
            s = sub[sub["level"] == level]
            o_v = s[s["method"] == "observed"].set_index("repeat")[m_key]
            cells = []
            for m in PI_METHODS:
                f_v = s[s["method"] == m].set_index("repeat")[m_key]
                err = (f_v - o_v).dropna()
                cells.append(f"{err.mean():+.1f} ({err.abs().mean():.1f})")
            L.append(f"| {lvl_name(level)} | {o_v.mean():.1f} | " + " | ".join(cells) + " |")
        L.append("")

    # A4: prevalence estimates
    L += ["## 4. How well are the class shares estimated? (APTOS resampled)", "",
          "Estimated share of referable eyes (routine + urgent), mean (5th–95th percentile over repeats), in %.", "",
          "| Referable eyes in the clinic | EM | BBSE | B8 |", "|---|---|---|---|"]
    s = work[(work["rule"] == "C1") & work["method"].isin(["EM", "BBSE", "B8"])].copy()
    s["pi_ref"] = 100 * (s["pi_1"] + s["pi_2"])
    for level in LEVELS + ["APTOS"]:
        cells = []
        for m in ("EM", "BBSE", "B8"):
            v = s[(s["level"] == level) & (s["method"] == m)]["pi_ref"]
            cells.append(f"{v.mean():.1f} ({v.quantile(0.05):.1f}–{v.quantile(0.95):.1f})")
        L.append(f"| {lvl_name(level)} | " + " | ".join(cells) + " |")
    L.append("")

    # A5: risks
    rk = work[work["rule"] == "risk"].groupby(["level", "method"], sort=False)[["ece_ref", "mean_risk_ref"]].mean()
    L += ["## 5. Each eye's risk, adjusted for the clinic's prevalence", "",
          "Calibration error (ECE, 10 bins) of P(referable) and its mean, against the true share of referable "
          "eyes. The adjusted risk is p(y|x) × π_y / π_APTOS,y, renormalised.", "",
          "| Referable eyes in the clinic | ECE unadjusted | ECE adjusted with EM shares | ECE adjusted with true "
          "shares | Mean risk unadjusted / EM-adjusted (%) |", "|---|---|---|---|---|"]
    for level in LEVELS + ["APTOS"]:
        L.append(f"| {lvl_name(level)} | {rk.loc[(level, 'none'), 'ece_ref']:.3f} | {rk.loc[(level, 'EM'), 'ece_ref']:.3f} | "
                 f"{rk.loc[(level, 'B9'), 'ece_ref']:.3f} | {100 * rk.loc[(level, 'none'), 'mean_risk_ref']:.1f} / "
                 f"{100 * rk.loc[(level, 'EM'), 'mean_risk_ref']:.1f} |")
    L.append("")

    # B1: real clinics, shares
    L += ["## 6. The real clinics: estimating the class shares", "",
          "No referral / routine / urgent, in %. Unlabelled-photo estimates use the all-APTOS model of E2. "
          "\"k labelled local photos\": the shares among the k photos C2 is refitted on (mean, 5th–95th percentile "
          f"of the urgent share over {args.repeats} draws).", "",
          "| Clinic | Truth | B8 APTOS | Most likely class | Mean probability | EM | BBSE | k labelled local photos |",
          "|---|---|---|---|---|---|---|---|"]
    for ds, name in (("idrid", "IDRiD"), ("messidor2", "Messidor-2")):
        p = prev[(prev["dataset"] == ds) & (prev["k"] == 0)].set_index("method")
        fmt = lambda m: " / ".join(f"{100 * p.loc[m, f'pi_{c}']:.1f}" for c in range(3))
        loc = prev[(prev["dataset"] == ds) & (prev["method"] == "local")]
        loc_txt = "; ".join(f"k = {k}: urgent {100 * g['pi_2'].mean():.1f} ({100 * g['pi_2'].quantile(0.05):.1f}–"
                            f"{100 * g['pi_2'].quantile(0.95):.1f})" for k, g in loc.groupby("k"))
        L.append(f"| {name} | {fmt('B9')} | {fmt('B8')} | {fmt('CC')} | {fmt('PCC')} | {fmt('EM')} | {fmt('BBSE')} | "
                 f"{loc_txt} |")
    L += ["", "EM and BBSE assume that only the class shares differ from APTOS. At Messidor-2 the photos themselves "
          "differ (camera, population, grading), so the model's probabilities are off and so are these estimates. "
          "The labelled photos that C2 needs anyway give an unbiased estimate.", ""]

    # B2: real clinics, workload forecast for C2
    L += ["## 7. The real clinics: forecasting C2's workload", "",
          "C2 refitted on k labelled local photos; its decision rates per class are measured on those same k photos; "
          "the forecast for the clinic's other photos uses the class shares from the k photos (local), from EM on "
          "the unlabelled photos, from APTOS (B8) or the truth (B9). Forecast minus observed, mean over draws (mean "
          "absolute error).", "",
          "| Clinic | k | Measure | Observed | local | EM | B8 | B9 |", "|---|---|---|---|---|---|---|---|"]
    for (ds, k), g in clin.groupby(["dataset", "k"]):
        o_v = g[g["method"] == "observed"].set_index("repeat")
        for m_key in ("referred", "urgent_dec", "ppv"):
            cells = []
            for m in ("local", "EM", "B8", "B9"):
                f_v = g[g["method"] == m].set_index("repeat")[m_key]
                err = (f_v - o_v[m_key]).dropna()
                cells.append(f"{err.mean():+.1f} ({err.abs().mean():.1f})")
            L.append(f"| {'IDRiD' if ds == 'idrid' else 'Messidor-2'} | {k} | {WORK_NAMES[m_key]} | "
                     f"{o_v[m_key].mean():.1f} | " + " | ".join(cells) + " |")
    L += ["", "## Notes", "",
          "- Referrals per 1,000 can also simply be counted once a clinic has run the rule on its photos; the "
          "forecast matters before that (planning capacity) and for the quantities that need labels: the PPV and "
          "the referable eyes sent home.",
          f"- Made {time.strftime('%Y-%m-%d %H:%M')} by c3_prevalence.py ({elapsed:.0f} s).", ""]
    (out / "summary.md").write_text("\n".join(L))


# --------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"observed": "#0b0b0b", "B8": "#a8a7a1", "EM": "#2a78d6", "BBSE": "#1baf7a", "B9": "#eb6834"}


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8, length=0)


def plot_workload(work, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = work[work["rule"] == "C1"]
    levels = LEVELS + ["APTOS"]
    x = np.arange(len(levels))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), dpi=200)
    for ax, m_key in zip(axes, ("referred", "missed", "ppv")):
        width = 0.16
        methods = ["observed", "B8", "EM", "BBSE", "B9"]
        for j, m in enumerate(methods):
            v = [w[(w["level"] == lv) & (w["method"] == m)][m_key].mean() for lv in levels]
            ax.bar(x + (j - 2) * width, v, width=width * 0.9, color=COL[m],
                   label="observed (labels)" if m == "observed" else PI_NAMES[m])
        ax.set_xticks(x)
        ax.set_xticklabels([lvl_name(lv) for lv in levels], fontsize=7.5)
        ax.set_title(WORK_NAMES[m_key], fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("Referable eyes in the clinic", fontsize=8, color=INK2)
        _style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7.5, ncol=5, loc="lower left", bbox_to_anchor=(0.01, -0.02))
    fig.suptitle("C3: the C1 rule's workload at clinics with fewer sick eyes, forecast from unlabelled photos "
                 "(APTOS resampled)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_prevalence(prev, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4), dpi=200, sharey=True)
    methods = ["B9", "B8", "CC", "PCC", "EM", "BBSE"]
    labels = ["truth", "APTOS (B8)", "most likely class", "mean probability", "EM", "BBSE"]
    for ax, (ds, name) in zip(axes, (("idrid", "IDRiD"), ("messidor2", "Messidor-2"))):
        p = prev[(prev["dataset"] == ds) & (prev["k"] == 0)].set_index("method")
        y = np.arange(len(methods))[::-1]
        left = np.zeros(len(methods))
        for c, (cname, col) in enumerate((("no referral", "#d6d5d0"), ("routine", "#2a78d6"), ("urgent", "#104281"))):
            v = 100 * np.array([p.loc[m, f"pi_{c}"] for m in methods])
            ax.barh(y, v, left=left, color=col, height=0.62, edgecolor="white", linewidth=1, label=cname)
            for yi, l, vi in zip(y, left, v):
                if vi >= 7:
                    ax.text(l + vi / 2, yi, f"{vi:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if c else INK)
            left += v
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8, color=INK)
        ax.set_xlim(0, 100)
        ax.set_title(name, fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("Share of eyes (%)", fontsize=8, color=INK2)
        _style(ax)
        ax.grid(axis="x", color=GRID, lw=0.8)
    axes[1].legend(frameon=False, fontsize=7.5, ncol=3, loc="lower left", bbox_to_anchor=(0, -0.42))
    fig.suptitle("Class shares at the new clinics, estimated from unlabelled photos with the APTOS model",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
