#!/usr/bin/env python3
"""
label_efficient.py - additional work: label-efficient recalibration at a new clinic (C2-IW).

Problem. C2 refits C1's three thresholds on k labelled local photos. A class needs at least 9 photos before its
threshold is finite; until then it is in every set and every eye is referred. At Messidor-2 only 6.3 % of eyes are
urgent, so randomly chosen photos contain few urgent ones: C2 needs about 300 labels before it sends a quarter of
healthy eyes home.

Method (C2-IW). Choose which photos to label with known probabilities that favour likely-rare classes, then undo the
sampling bias with weights, class by class.
  1. Rarity score of every unlabelled clinic photo, from the APTOS model alone (no labels):
       r(x) = sum_c p(c | x) / m_c,  m_c = the mean of p(c | x) over the clinic's photos,
     which would give each class about the same expected number of labels if the model were calibrated and
     confident; its mean over the clinic is 3. Mixed with uniform sampling for safety (a defensive mixture):
     r'(x) = (1 - lam) + lam * r(x) / 3, so no photo's weight below exceeds 1 / (1 - lam) times its weight under
     uniform sampling.
  2. Poisson sampling by whole groups (the two eyes of an exam together): group g is labelled with probability
     pi_g = min(1, c * mean r'(x in g)), c set so that k photos are labelled on average.
  3. For each class c, the labelled class-c photos are weighted by w = 1 / pi, which restores the clinic's own
     class-c distribution of scores. The threshold is the weighted quantile at level (1 - alpha)(1 + 1 / n_eff),
     where n_eff = (sum w)^2 / sum w^2 is the effective number of labelled class-c photos (Kish); it is infinite
     while n_eff < 1/alpha - 1 (9). With equal weights, n_eff = n and this is exactly C2's threshold.
     Interpretation: it is the exact weighted conformal threshold (Tibshirani et al. 2019) with the test photo's own
     weight replaced by its expected value under the clinic's class-c distribution, sum w^2 / sum w. The exact rule
     protects test photos with a large weight (unlikely to be labelled) by giving them larger thresholds; the
     plug-in protects them only on average. Its coverage is therefore approximate, and the larger lambda is, the
     more uneven the weights and the larger the possible shortfall: the experiment measures it.
Rules compared, all on the same draws (common random numbers: in each draw every lambda sees the same split of the
clinic and the same lottery numbers for the exam groups):
  random       C2: k random photos by whole groups, as in E2
  iw@lam       C2-IW at lam = 0.25, 0.5, 0.75, 1
  naive@lam    the same rarity sampling, thresholds without the weights
  exact@lam    exact weighted conformal prediction: the test photo's weight 1 / pi(x) as a point mass at +infinity
               (exactly valid, but most healthy-looking eyes were unlikely to be labelled, so the rare classes get
               infinite thresholds for them and they are referred)
Ours is C2-IW at lam = 0.25: the largest lambda of the grid whose urgent coverage stays within Monte Carlo noise of
the target in every setting (summary, section on lambda).
The model: the all-APTOS logistic regression of E2 (results/e2/aptos_model.csv.gz), raw probabilities.

Evaluation. In every draw the clinic's exam groups are split at random into two halves: the labels are drawn from
half A (its unlabelled photos also give the rarity scores), and the rule is tested on half B, which plays the
clinic's future patients. (Testing on the photos left unlabelled would be biased against rarity sampling: they are
depleted of the photos it prefers.) Settings: Messidor-2, IDRiD, and APTOS resampled to 5 % referable eyes
(E2's out-of-fold probabilities).

Outputs (in --out, default ../results/label_efficient): summary.md, draws.csv.gz, curves.csv,
label_efficient.png

Usage:
  python label_efficient.py --data ../data_512 --results ../results      # about 5 minutes
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from c1_referral import HOME, ROUTINE, URGENT, UNCERTAIN, decide_from_sets
from e2_new_clinics import Pool

K_GRID = {"messidor2": [25, 50, 75, 100, 150, 200, 300], "idrid": [20, 30, 50, 75, 100],
          "aptos5": [50, 100, 150, 200, 300]}
CLINIC_NAMES = {"messidor2": "Messidor-2", "idrid": "IDRiD", "aptos5": "APTOS resampled to 5 % referable"}
LAMBDAS = (0.25, 0.5, 0.75, 1.0)
OURS = "iw@0.25"
MAIN = ("random", "iw@0.25", "iw@0.5", "naive@0.25", "exact@0.25")
NAMES = {"random": "C2: random photos", "iw@0.25": "C2-IW, λ = 0.25 (ours)", "iw@0.5": "C2-IW, λ = 0.5",
         "naive@0.25": "Rarity sampling without weights (λ = 0.25)",
         "exact@0.25": "Exact weighted conformal (λ = 0.25)"}
RULES = {"iw": "C2-IW (weighted)", "naive": "without weights", "exact": "exact weighted conformal"}
K_SHOW = {"messidor2": [75, 100, 150, 200, 300], "aptos5": [150, 200, 300], "idrid": [30, 50, 100]}


def rarity(P):
    m = P.mean(0)
    return (P / m[None, :]).sum(1)


def group_inclusion(r_photo, group, k):
    """Poisson inclusion probability per photo (by whole groups), proportional to the group's mean score, capped
    at 1, with an expected k labelled photos. Returns the per-photo probabilities and the scale c."""
    g, inv, size = np.unique(group, return_inverse=True, return_counts=True)
    rg = np.bincount(inv, weights=r_photo) / size
    lo, hi = 0.0, 1.0
    while np.sum(np.minimum(1.0, hi * rg) * size) < k:
        hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.sum(np.minimum(1.0, mid * rg) * size) < k:
            lo = mid
        else:
            hi = mid
    return np.minimum(1.0, hi * rg)[inv], hi


def group_mean(values, group):
    _, inv, size = np.unique(group, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=values) / size)[inv]


def weighted_thresholds(Pc, yc, wc, Pt, wt, alpha, mode="ess"):
    """Class thresholds from labelled photos with weights wc (per test photo: an n_test x 3 array).
    mode "ess":   the weighted quantile at level (1 - alpha)(1 + 1/n_eff), the same for every test photo
                  (equal weights: exactly the usual conformal threshold);
    mode "exact": weighted conformal prediction with the test photo's weight wt as a mass at +infinity."""
    q = np.full((len(Pt), 3), np.inf)
    for c in range(3):
        sel = yc == c
        if not sel.any():
            continue
        s, w = 1.0 - Pc[sel, c], wc[sel]
        o = np.argsort(s, kind="stable")
        s, cw = s[o], np.cumsum(w[o])
        W = cw[-1]
        if mode == "ess":
            n_eff = W * W / np.sum(w * w)
            level = (1 - alpha) * (1 + 1 / n_eff)
            if level <= 1 + 1e-12:
                j = int(np.searchsorted(cw / W, level - 1e-12, side="left"))
                q[:, c] = s[min(j, len(s) - 1)]
        else:
            need = (1 - alpha) * (W + wt)
            j = np.searchsorted(cw, need - 1e-12 * W, side="left")
            ok = j < len(s)
            q[ok, c] = s[j[ok]]
    return q


def metrics(d, y, in_set, q):
    ref, urg, non = y >= 1, y == 2, y == 0
    cov = in_set[np.arange(len(y)), y]
    return {"ref_home": np.mean(d[ref] == HOME), "urgent_urgent": np.mean(d[urg] == URGENT),
            "healthy_home": np.mean(d[non] == HOME), "uncertain": np.mean(d == UNCERTAIN),
            "referred": np.mean((d == ROUTINE) | (d == URGENT)),
            **{f"coverage_{c}": cov[y == c].mean() for c in range(3)},
            "urgent_finite": np.mean(np.isfinite(q[:, 2]))}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Additional work: label-efficient recalibration at a new clinic.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", default="../data_512")
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default="../results/label_efficient")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    a = args.alpha
    man = pd.read_csv(Path(args.data) / "manifest.csv")
    e2 = pd.read_csv(Path(args.results) / "e2" / "aptos_model.csv.gz")
    assert (e2["uid"].astype(str) == man["uid"].astype(str)).all()
    P_all = e2[[f"plain_p{c}" for c in range(3)]].to_numpy()
    y_all = man["ref3"].to_numpy(int)
    group_all = pd.factorize(man["dataset"] + "/" + man["group_id"].astype(str))[0]
    rng = np.random.default_rng(args.seed)
    # a third setting: APTOS resampled to 5 % referable eyes (grade mix kept); its probabilities are E2's
    # cross-fitted out-of-fold ones, so no photo is scored by a model that saw it
    rs = np.random.default_rng(args.seed + 7)
    apt = man["dataset"].to_numpy() == "aptos"
    grade = man["grade"].to_numpy()
    non = np.flatnonzero(apt & (y_all == 0))
    n_ref = int(round(len(non) * 0.05 / 0.95))
    ref_idx = np.flatnonzero(apt & (y_all >= 1))
    share = np.array([np.mean(grade[ref_idx] == gr) for gr in (2, 3, 4)])
    pick = [rs.choice(np.flatnonzero(apt & (grade == gr)), int(round(n_ref * sh)), replace=False)
            for gr, sh in zip((2, 3, 4), share)]
    clinic_idx = {"messidor2": np.flatnonzero((man["dataset"] == "messidor2").to_numpy()),
                  "idrid": np.flatnonzero((man["dataset"] == "idrid").to_numpy()),
                  "aptos5": np.sort(np.concatenate([non] + pick))}
    rows = []
    for ds, ks in K_GRID.items():
        idx = clinic_idx[ds]
        P, y, grp = P_all[idx], y_all[idx], group_all[idx]
        ugroups = np.unique(grp)
        for k in ks:
            for b in range(args.draws):
                half = rng.permutation(ugroups)[: len(ugroups) // 2]
                inA = np.isin(grp, half)
                A, B = np.flatnonzero(inA), np.flatnonzero(~inA)
                # C2: random photos from half A, by whole groups (as in E2)
                cal, _ = Pool(A, A, grp).draw(k, rng)
                q = weighted_thresholds(P[cal], y[cal], np.ones(len(cal)), P[B], np.ones(len(B)), a, "ess")
                in_set = (1 - P[B]) <= q
                rows.append({"dataset": ds, "k": k, "draw": b, "method": "random", "n_labelled": len(cal),
                             "n_urgent": int(np.sum(y[cal] == 2)), "ess_urgent": float(np.sum(y[cal] == 2)),
                             "max_weight_ratio": 1.0, **metrics(decide_from_sets(in_set), y[B], in_set, q)})
                # rarity sampling in half A; the scores come from A's unlabelled photos only; one lottery number
                # per exam group, shared by every lambda
                rA = rarity(P[A])
                mA = P[A].mean(0)
                rB = (P[B] / mA[None, :]).sum(1)
                gA = np.unique(grp[A])
                U = rng.random(len(gA))
                for lam in LAMBDAS:
                    piA, c = group_inclusion((1 - lam) + lam * rA / 3.0, grp[A], k)
                    piB = np.minimum(1.0, c * group_mean((1 - lam) + lam * rB / 3.0, grp[B]))
                    gpi = pd.Series(piA).groupby(grp[A]).first().reindex(gA).to_numpy()
                    lab = np.isin(grp[A], gA[U < gpi])
                    cal = A[lab]
                    if len(cal) == 0:
                        continue
                    w = 1.0 / piA[lab]
                    for rule, wc, wt, mode in (("iw", w, 1.0 / piB, "ess"),
                                               ("naive", np.ones(len(cal)), np.ones(len(B)), "ess"),
                                               ("exact", w, 1.0 / piB, "exact")):
                        q = weighted_thresholds(P[cal], y[cal], wc, P[B], wt, a, mode)
                        in_set = (1 - P[B]) <= q
                        u = y[cal] == 2
                        ess = (wc[u].sum() ** 2 / np.sum(wc[u] ** 2)) if u.any() else 0.0
                        rows.append({"dataset": ds, "k": k, "draw": b, "method": f"{rule}@{lam:g}",
                                     "n_labelled": len(cal), "n_urgent": int(u.sum()), "ess_urgent": ess,
                                     "max_weight_ratio": float(wc.max() / wc.mean()),
                                     **metrics(decide_from_sets(in_set), y[B], in_set, q)})
            print(f"{ds} k = {k} done ({time.time() - t0:.0f} s)", flush=True)
    draws = pd.DataFrame(rows)
    draws.to_csv(out / "draws.csv.gz", index=False, float_format="%.5f")
    met = ["n_labelled", "n_urgent", "ess_urgent", "max_weight_ratio", "ref_home", "urgent_urgent", "healthy_home",
           "uncertain", "referred", "coverage_0", "coverage_1", "coverage_2", "urgent_finite"]
    gb = draws.groupby(["dataset", "method", "k"])
    curves = gb[met].mean()
    curves["coverage_2_se"] = gb["coverage_2"].std() / np.sqrt(gb["coverage_2"].size())
    for c in range(3):
        curves[f"below_{c}"] = gb[f"coverage_{c}"].apply(lambda x: np.mean(x < 1 - a))
    curves["targets_met"] = gb.apply(lambda x: np.mean((x["ref_home"] <= a) & (x["urgent_urgent"] >= 1 - a)),
                                     include_groups=False)
    curves = curves.reset_index()
    curves.to_csv(out / "curves.csv", index=False, float_format="%.5f")
    write_summary(out, args, curves, time.time() - t0)
    plot(curves, out / "label_efficient.png", a)
    print(f"wrote {out / 'summary.md'} ({time.time() - t0:.0f} s)")


def write_summary(out, args, curves, elapsed):
    a = args.alpha
    L = ["# Additional work: label-efficient recalibration at a new clinic (C2-IW)", "",
         "All-APTOS logistic regression (E2), raw probabilities. For each clinic and label budget k, "
         f"{args.draws} draws. C2: k random photos by whole groups. Rarity sampling: Poisson sampling of whole groups "
         "with probability proportional to (1 − λ) + λ r(x)/3, r(x) = Σ_c p(c | x) / mean p(c), k photos labelled on "
         "average; thresholds weighted by 1/π (C2-IW), unweighted, or exact weighted conformal. Each draw splits the "
         "clinic's exam groups at random into two halves: labels come from half A, and the rule is tested on half B "
         f"(the clinic's future patients); every λ sees the same split and lottery numbers. In %, means over draws; "
         f"α = {a:g}; ± is the Monte Carlo standard error.", ""]
    for ds in ("messidor2", "aptos5", "idrid"):
        L += [f"## {CLINIC_NAMES[ds]}", "",
              "| k | Method | Labelled (urgent) | Coverage no referral / routine | Urgent coverage | Urgent coverage below "
              "90 % (share of draws) | Referable sent home | Healthy sent home | Uncertain | Urgent threshold finite |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        c = curves[curves["dataset"] == ds]
        for k in sorted(c["k"].unique()):
            for m in MAIN:
                r = c[(c["k"] == k) & (c["method"] == m)]
                if not len(r):
                    continue
                r = r.iloc[0]
                L.append(f"| {k} | {NAMES[m]} | {r['n_labelled']:.0f} ({r['n_urgent']:.1f}) | "
                         f"{100 * r['coverage_0']:.1f} / {100 * r['coverage_1']:.1f} | "
                         f"{100 * r['coverage_2']:.1f} ± {100 * r['coverage_2_se']:.1f} | {100 * r['below_2']:.0f} | "
                         f"{100 * r['ref_home']:.1f} | {100 * r['healthy_home']:.1f} | {100 * r['uncertain']:.1f} | "
                         f"{100 * r['urgent_finite']:.0f} |")
        L.append("")
    L += ["## λ: validity against efficiency", "",
          "Each cell: urgent coverage ± Monte Carlo standard error, and in brackets the healthy eyes sent home. The "
          "rows marked 'largest weight' give the largest weight relative to the mean weight of the labelled photos "
          "(average over draws, at the largest k shown).", "",
          "| Clinic | k | Rule | " + " | ".join(f"λ = {lam:g}" for lam in LAMBDAS) + " |",
          "|---|---|---|" + "---|" * len(LAMBDAS)]
    for ds, ks in K_SHOW.items():
        c = curves[curves["dataset"] == ds]
        for k in ks:
            rnd = c[(c["k"] == k) & (c["method"] == "random")].iloc[0]
            for rule, rname in RULES.items():
                cells_ = []
                for lam in LAMBDAS:
                    r = c[(c["k"] == k) & (c["method"] == f"{rule}@{lam:g}")].iloc[0]
                    cells_.append(f"{100 * r['coverage_2']:.1f} ± {100 * r['coverage_2_se']:.1f} "
                                  f"({100 * r['healthy_home']:.1f})")
                L.append(f"| {CLINIC_NAMES[ds]} | {k} | {rname} | " + " | ".join(cells_) + " |")
            L.append(f"| {CLINIC_NAMES[ds]} | {k} | C2, random photos (for reference) | "
                     f"{100 * rnd['coverage_2']:.1f} ± {100 * rnd['coverage_2_se']:.1f} ({100 * rnd['healthy_home']:.1f})"
                     + " |" * len(LAMBDAS))
        kmax = max(ks)
        L.append(f"| {CLINIC_NAMES[ds]} | {kmax} | largest weight / mean weight | " + " | ".join(
            f"{c[(c['k'] == kmax) & (c['method'] == f'iw@{lam:g}')]['max_weight_ratio'].iloc[0]:.1f}"
            for lam in LAMBDAS) + " |")
    L += ["", f"Made {time.strftime('%Y-%m-%d %H:%M')} by label_efficient.py ({elapsed:.0f} s).", ""]
    (out / "summary.md").write_text("\n".join(L))


def plot(curves, path, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid_c = "#0b0b0b", "#52514e", "#e4e3df"
    cols = {"random": "#52514e", "naive@0.25": "#eb6834", "iw@0.5": "#9cc3ef", "iw@0.25": "#2a78d6",
            "exact@0.25": "#a8a7a1"}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6), dpi=200)
    for row, ds in enumerate(("messidor2", "aptos5")):
        c = curves[curves["dataset"] == ds]
        for ax, (key, title, target) in zip(axes[row], (("coverage_2", "urgent eyes covered (%)", 100 * (1 - a)),
                                                        ("healthy_home", "healthy eyes sent home (%)", None))):
            for m in MAIN:
                s_ = c[c["method"] == m].sort_values("k")
                ax.plot(s_["k"], 100 * s_[key], "-o", ms=3, lw=2.2 if m == OURS else 1.3, color=cols[m],
                        label=NAMES[m], zorder=3 if m == OURS else 2)
            if target is not None:
                ax.axhline(target, color=ink, lw=0.9, ls=(0, (4, 3)))
                ax.set_ylim(75, 101)
            ax.set_title(f"{CLINIC_NAMES[ds]}: {title}", fontsize=9, loc="left", color=ink)
            ax.set_xscale("log")
            ks = sorted(c["k"].unique())
            ax.set_xticks(ks)
            ax.set_xticklabels([str(k) for k in ks], fontsize=7)
            ax.minorticks_off()
            ax.grid(color=grid_c, lw=0.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(colors=ink2, labelsize=7, length=0)
        axes[row, 0].set_xlabel("labelled local photos (k)", fontsize=8, color=ink2)
        axes[row, 1].set_xlabel("labelled local photos (k)", fontsize=8, color=ink2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Additional work: choosing which photos to label at a new clinic (dashed: the 90 % target; "
                 "tested on a held-out half of the clinic)", fontsize=9.5, color=ink, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
