#!/usr/bin/env python3
"""
e0_conformal.py - E0: reproduce the coverage drop reported in [8] (Hasan et al., Diagnostics 2026).

Standard (marginal) split conformal prediction, as in [8, 9]: a model trained on APTOS and one threshold
for all classes, set on APTOS calibration images and then used unchanged at the other clinics.

For each outer fold of E1, the predictions saved by run_models.py give:
  calibration   the fold's 15 % calibration images (APTOS, never used for training)
  test sets     APTOS: the held-out fold (same clinic as the calibration images)
                IDRiD: all 516 images, and the official 103-image test split
                Messidor-2: 1,744 images
Conformity scores, alpha = 0.1 (target coverage 90 %), raw model probabilities:
  LAC  s = 1 - p(y | x); the set holds every class with s <= q        (Sadinle et al. 2019)
  APS  s = total probability of the classes at least as likely as y   (Romano et al. 2020, without randomisation)
q is the ceil((n + 1)(1 - alpha))-th smallest calibration score. Coverage = share of images whose true
class is in their set. Reported overall and per DR grade, as the mean (and range) over the 10 folds.

A model trained with --target grade (5 grades, e.g. folder lr-grade) is evaluated on grades, as in [8];
a 3-class model on our classes (a grade-g image is covered when its class is in the set).

Outputs (in --out, default ../results/e0): summary.md, coverage.csv, coverage_by_grade.png (the first two models)

Usage:
  python e0_conformal.py --data ../data_512 --results ../results/e1 --models lr-grade,lr
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

SETS = [("aptos", "APTOS, held-out fold"), ("idrid", "IDRiD, all 516"), ("idrid_test", "IDRiD, official test (103)"),
        ("messidor2", "Messidor-2")]
GRADES = ["0 none", "1 mild", "2 moderate", "3 severe", "4 proliferative"]


def lac(P):
    return 1 - P


def aps(P):
    """S[i, c] = sum of P[i, j] over the classes j with P[i, j] >= P[i, c]."""
    return (P[:, None, :] * (P[:, None, :] >= P[:, :, None])).sum(2)


def threshold(s_cal, alpha):
    n = len(s_cal)
    k = math.ceil((n + 1) * (1 - alpha))
    return np.inf if k > n else np.sort(s_cal)[k - 1]


def main(argv=None):
    ap = argparse.ArgumentParser(description="E0: standard conformal prediction, APTOS -> other clinics.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", default="../data_512")
    ap.add_argument("--results", default="../results/e1")
    ap.add_argument("--out", default="../results/e0")
    ap.add_argument("--models", default="lr-grade,lr", help="folders in --results")
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(Path(args.data) / "manifest.csv").set_index("uid")
    rows = []
    for key in [k.strip() for k in args.models.split(",") if k.strip()]:
        files = sorted(Path(args.results, key).glob("fold*.csv.gz"))
        files = [f for f in files if f.name[4:].split(".")[0].isdigit()]
        if not files:
            print(f"no predictions for {key}; run run_models.py first")
            continue
        for f in files:
            fold = int(f.name[4:].split(".")[0])
            df = pd.read_csv(f).join(man[["dataset", "split", "grade", "ref3"]], on="uid")
            k = sum(c.startswith("p") and c[1:].isdigit() for c in df.columns)
            target = "grade" if k == 5 else "ref3"
            P = df[[f"p{c}" for c in range(k)]].to_numpy()
            y = df[target].to_numpy(int)
            masks = {"aptos": (df["role"] == "test").to_numpy(),
                     "idrid": ((df["role"] == "external") & (df["dataset"] == "idrid")).to_numpy(),
                     "idrid_test": ((df["role"] == "external") & (df["dataset"] == "idrid")
                                    & (df["split"] == "test")).to_numpy(),
                     "messidor2": ((df["role"] == "external") & (df["dataset"] == "messidor2")).to_numpy()}
            cal = (df["role"] == "cal").to_numpy()
            for score_name, score in (("LAC", lac), ("APS", aps)):
                S = score(P)
                ic = np.flatnonzero(cal)
                q = threshold(S[ic, y[ic]], args.alpha)
                in_set = S <= q
                covered = in_set[np.arange(len(df)), y]
                size = in_set.sum(1)
                for set_key, m in masks.items():
                    g = df["grade"].to_numpy()
                    groups = [("all", m)] + [(f"grade {gr}", m & (g == gr)) for gr in range(5)]
                    for group, s in groups:
                        if s.sum() == 0:
                            continue
                        rows.append({"model": key, "target": target, "score": score_name, "fold": fold,
                                     "set": set_key, "group": group, "n": int(s.sum()),
                                     "coverage": covered[s].mean(), "mean_set_size": size[s].mean(),
                                     "empty_sets": (size[s] == 0).mean(), "threshold": q})
    if not rows:
        raise SystemExit("nothing to summarise")
    cov = pd.DataFrame(rows)
    cov.to_csv(out / "coverage.csv", index=False, float_format="%.4f")

    agg = cov.groupby(["model", "target", "score", "set", "group"], sort=False).agg(
        coverage=("coverage", "mean"), lo=("coverage", "min"), hi=("coverage", "max"),
        size=("mean_set_size", "mean"), n=("n", "first")).reset_index()

    lines = ["# E0: standard conformal prediction trained on APTOS, used at other clinics", "",
             f"Target coverage {1 - args.alpha:.0%} (alpha = {args.alpha}). One threshold for all classes, set on the "
             "15 % APTOS calibration images of each E1 outer fold. Coverage: mean over the 10 folds "
             "(lowest–highest fold in brackets). Published reference [8]: 66–77 % coverage on IDRiD at a "
             "90 % target, about 30 % for severe DR.", ""]
    for (key, target, score), a in agg.groupby(["model", "target", "score"], sort=False):
        label = "5 grades" if target == "grade" else "3 classes (no referral / routine / urgent)"
        lines += [f"## {key}: {label}, {score} score", "",
                  "| Test set | All | " + " | ".join(f"Grade {g}" for g in GRADES) + " | Mean set size |",
                  "|---|---|" + "---|" * len(GRADES) + "---|"]
        for set_key, set_name in SETS:
            s = a[a["set"] == set_key].set_index("group")
            if "all" not in s.index:
                continue
            cells = []
            for group in ["all"] + [f"grade {gr}" for gr in range(5)]:
                if group in s.index:
                    r = s.loc[group]
                    cells.append(f"{100 * r['coverage']:.1f} ({100 * r['lo']:.0f}–{100 * r['hi']:.0f})")
                else:
                    cells.append("–")
            lines.append(f"| {set_name} (n = {int(s.loc['all', 'n'])}) | " + " | ".join(cells)
                         + f" | {s.loc['all', 'size']:.2f} |")
        lines.append("")
    lines += ["Coverage in %. Grade columns: images of that DR grade, so the urgent class is grades 3 and 4.",
              f"Made {time.strftime('%Y-%m-%d %H:%M')} by e0_conformal.py.", ""]
    (out / "summary.md").write_text("\n".join(lines))
    plot(agg, out / "coverage_by_grade.png", 1 - args.alpha)
    print(f"wrote {out / 'summary.md'}")


def plot(agg, path, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid_c = "#0b0b0b", "#52514e", "#e4e3df"
    colors = {"aptos": "#2a78d6", "idrid": "#eb6834", "messidor2": "#1baf7a"}
    names = {"aptos": "APTOS (same clinic)", "idrid": "IDRiD", "messidor2": "Messidor-2"}
    panels = [(m, t) for (m, t, s), _ in agg.groupby(["model", "target", "score"], sort=False) if s == "LAC"][:2]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.8), sharey=True, dpi=200, squeeze=False)
    for ax, (model, tgt) in zip(axes[0], panels):
        a = agg[(agg["model"] == model) & (agg["target"] == tgt) & (agg["score"] == "LAC")]
        ax.axhline(100 * target, color=ink2, lw=1, ls=(0, (4, 3)))
        ax.text(4.25, 100 * target + 1.5, f"target {target:.0%}", fontsize=7.5, color=ink2, ha="right")
        for j, (set_key, col) in enumerate(colors.items()):
            s = a[(a["set"] == set_key) & a["group"].str.startswith("grade")]
            x = s["group"].str[-1].astype(int).to_numpy() + (j - 1) * 0.12
            ax.vlines(x, 100 * s["lo"], 100 * s["hi"], color=col, lw=2)
            ax.plot(x, 100 * s["coverage"], "-o", color=col, lw=2, ms=6, mec="white", mew=1.2, label=names[set_key])
        ax.set_xticks(range(5))
        ax.set_xticklabels(["0", "1", "2", "3", "4"])
        ax.set_xlabel("DR grade", fontsize=8.5, color=ink2)
        what = "5-grade model" if tgt == "grade" else "3-class model"
        ax.set_title(f"{what} ({model}), LAC score", fontsize=10, color=ink, loc="left")
        ax.set_ylim(0, 103)
        ax.grid(axis="y", color=grid_c, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(ink2)
        ax.tick_params(colors=ink2, labelsize=8, length=0)
    axes[0][0].set_ylabel("Coverage (%), mean and range over 10 folds", fontsize=8.5, color=ink2)
    axes[0][0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("E0: standard conformal prediction calibrated on APTOS", fontsize=10.5, color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
