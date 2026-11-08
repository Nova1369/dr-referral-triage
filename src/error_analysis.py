#!/usr/bin/env python3
"""
error_analysis.py - which eyes does the C1 rule get wrong, and is it fair across subgroups?

Uses the out-of-fold C1 decisions on APTOS (results/c1/decisions_<model>.csv.gz) and the photo table.
Two kinds of error matter:
  missed urgent eyes      grade 3-4 eyes not referred urgently (sent home, routine or uncertain)
  referable eyes sent home  grade 2-4 eyes sent home
For each, the rate in every subgroup, with a 95 % Wilson interval:
  - DR grade
  - camera group (H / M / S, as in the camera study)
  - image sharpness (thirds of the variance of the Laplacian)
  - quality warnings (dark, overexposed, low sharpness, doubtful circle fit)
  - copies with conflicting grades
The model's own probabilities show whether the errors are confident or borderline.

Outputs (in --out, default ../results/errors): summary.md, subgroups.csv, missed_urgent.csv (one row per
missed urgent eye, with its probabilities and photo details, for looking at the photos), errors_by_subgroup.png

Usage:
  python error_analysis.py --data ../data_512 --results ../results
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from c1_referral import camera_groups


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def main(argv=None):
    ap = argparse.ArgumentParser(description="Error analysis and subgroup fairness of C1.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", default="../data_512")
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default="../results/errors")
    ap.add_argument("--model", default="lr")
    ap.add_argument("--rule", default="C1_a0.1")
    args = ap.parse_args(argv)
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R = Path(args.results)

    man = pd.read_csv(Path(args.data) / "manifest.csv")
    man["camera_group"] = camera_groups(man)
    man = man.set_index("uid")
    dec = pd.read_csv(R / "c1" / f"decisions_{args.model}.csv.gz").query("role == 'test'").set_index("uid")
    frames = [pd.read_csv(f) for f in sorted((R / "e1" / args.model).glob("fold[0-9]*.csv.gz"))]
    prob = pd.concat(frames).query("role == 'test'").set_index("uid")[["p0", "p1", "p2"]]
    d = man.loc[dec.index].join(prob).assign(decision=dec[args.rule])
    d["camera"] = d["orig_w"].astype(str) + "x" + d["orig_h"].astype(str)
    edges = np.quantile(d["q_lapvar"], [1 / 3, 2 / 3])
    d["sharpness"] = pd.Categorical(np.array(["least sharp third", "middle third", "sharpest third"])[
        np.digitize(d["q_lapvar"], edges)], ["least sharp third", "middle third", "sharpest third"])
    flags = d["qc_flags"].fillna("")
    d["warning"] = np.where(flags == "", "no warning", "any quality warning")
    d["conflict"] = np.where(d["dup_class_conflict"].fillna(False).astype(bool), "copy with another class",
                             "no conflicting copy")
    urg, ref = d["ref3"] == 2, d["ref3"] >= 1
    d["missed_urgent"] = urg & (d["decision"] != "urgent")
    d["ref_home"] = ref & (d["decision"] == "no referral")

    subgroups = [("DR grade", "grade"), ("camera group", "camera_group"), ("sharpness", "sharpness"),
                 ("quality warning", "warning"), ("conflicting copy", "conflict")]
    rows = []
    for title, col in subgroups:
        for val, g in d.groupby(col, observed=True):
            for err, base in (("missed_urgent", g["ref3"] == 2), ("ref_home", g["ref3"] >= 1)):
                n, k = int(base.sum()), int(g.loc[base, err].sum())
                lo, hi = wilson(k, n)
                rows.append({"subgroup": title, "value": str(val), "error": err, "eyes": n, "errors": k,
                             "rate": k / n if n else np.nan, "lo": lo, "hi": hi})
    sub = pd.DataFrame(rows)
    sub.to_csv(out / "subgroups.csv", index=False, float_format="%.4f")
    miss = d[d["missed_urgent"]].sort_values("p2")
    miss[["grade", "decision", "p0", "p1", "p2", "camera", "camera_group", "q_lapvar", "qc_flags",
          "dup_class_conflict", "fold", "out_path"]].to_csv(out / "missed_urgent.csv", float_format="%.4f")

    # ---- summary ----
    n_urg, n_ref = int(urg.sum()), int(ref.sum())
    m_urg, m_ref = int(d["missed_urgent"].sum()), int(d["ref_home"].sum())
    L = [f"# Error analysis of the C1 rule ({args.model}, {args.rule.replace('_a', ', α = ')})", "",
         f"APTOS out-of-fold decisions: every photo decided once by a rule fitted without it. "
         f"{m_urg} of {n_urg} urgent eyes ({100 * m_urg / n_urg:.1f} %) were not referred urgently, and "
         f"{m_ref} of {n_ref} referable eyes ({100 * m_ref / n_ref:.1f} %) were sent home. Rates by subgroup, "
         "with 95 % Wilson intervals; small groups give wide intervals.", ""]
    L += ["## 1. What the missed urgent eyes received", "",
          "| Decision | Missed urgent eyes | Grade 3 | Grade 4 |", "|---|---|---|---|"]
    for dname in ("routine", "no referral", "uncertain"):
        s = d["missed_urgent"] & (d["decision"] == dname)
        L.append(f"| {dname} | {int(s.sum())} | {int((s & (d['grade'] == 3)).sum())} | "
                 f"{int((s & (d['grade'] == 4)).sum())} |")
    q = miss["p2"].quantile([0.25, 0.5, 0.75]).to_numpy()
    q_all = d.loc[urg, "p2"].quantile([0.25, 0.5, 0.75]).to_numpy()
    L += ["", f"P(urgent) of the missed urgent eyes: median {q[1]:.3f} (quartiles {q[0]:.3f}–{q[2]:.3f}), against "
          f"{q_all[1]:.3f} ({q_all[0]:.3f}–{q_all[2]:.3f}) for all urgent eyes. The model is fairly sure these eyes are "
          "not urgent: they are not borderline cases a slightly lower threshold would catch.", ""]
    for err, title in (("missed_urgent", "Urgent eyes not referred urgently"), ("ref_home", "Referable eyes sent home")):
        L += [f"## {'2' if err == 'missed_urgent' else '3'}. {title}, by subgroup", "",
              "| Subgroup | Group | Eyes | Errors | Rate (95 % CI) |", "|---|---|---|---|---|"]
        for _, r in sub[sub["error"] == err].iterrows():
            if r["eyes"] == 0:
                continue
            L.append(f"| {r['subgroup']} | {r['value']} | {r['eyes']} | {r['errors']} | "
                     f"{100 * r['rate']:.1f} ({100 * r['lo']:.1f}–{100 * r['hi']:.1f}) |")
        L.append("")
    h = d[(d["camera_group"] == "H") & ref]
    S = sub.set_index(["error", "subgroup", "value"])

    def rate(err, sg, val):
        r = S.loc[(err, sg, val)]
        return f"{100 * r['rate']:.1f} % ({int(r['errors'])} of {int(r['eyes'])})"
    ref_by_third = d[ref].groupby("sharpness", observed=True)["camera_group"].apply(lambda x: np.mean(x == "H"))
    h_err_sharp = (d["ref_home"] & (d["sharpness"] == "sharpest third") & (d["camera_group"] == "H")).sum()
    L += ["## 4. Reading the tables", "",
          f"- **The camera is the largest effect.** On the mostly-healthy cameras (H), {int(h['ref_home'].sum())} of "
          f"{len(h)} referable eyes are sent home, {int((h['ref_home'] & (h['grade'] == 2)).sum())} of them grade 2. "
          "This is the camera shortcut studied in the additional work: the class-wise guarantee holds on average over cameras, "
          "not within each camera.",
          f"- **Grade.** Referable eyes sent home are mostly grade 2: {rate('ref_home', 'DR grade', '2')}, against "
          f"{rate('ref_home', 'DR grade', '3')} for grade 3 and {rate('ref_home', 'DR grade', '4')} for grade 4. Urgent "
          f"eyes are missed about equally often in grade 3 ({rate('missed_urgent', 'DR grade', '3')}) and grade 4 "
          f"({rate('missed_urgent', 'DR grade', '4')}), mostly by being called routine.",
          f"- **Sharpness is tangled with the camera.** Referable eyes are sent home most often in the sharpest third "
          f"({rate('ref_home', 'sharpness', 'sharpest third')}), but {int(h_err_sharp)} of those errors are H-camera "
          f"eyes ({100 * ref_by_third.iloc[2]:.0f} % of the referable eyes in that third come from H cameras, against "
          f"{100 * ref_by_third.iloc[0]:.0f} % in the least sharp third). Urgent "
          f"eyes are missed somewhat more often in blurrier photos ({rate('missed_urgent', 'sharpness', 'least sharp third')} "
          f"against {rate('missed_urgent', 'sharpness', 'sharpest third')}); the intervals overlap.",
          "- **Quality warnings and conflicting copies** show no clear difference; the groups are small and the "
          "intervals overlap.",
          f"- **The misses are confident.** Missed urgent eyes have a median P(urgent) of {q[1]:.3f}, so a slightly "
          "lower threshold would not catch them; they look non-urgent to the model.", "",
          "`missed_urgent.csv` lists every missed urgent eye with its photo path, for looking at the photos "
          "(keep them on your own computer: the licences do not allow sharing them).", "",
          f"Made {time.strftime('%Y-%m-%d %H:%M')} by error_analysis.py.", ""]
    (out / "summary.md").write_text("\n".join(L))
    plot(sub, out / "errors_by_subgroup.png")
    print(f"wrote {out / 'summary.md'} ({time.time() - t0:.0f} s)")


def plot(sub, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid_c = "#0b0b0b", "#52514e", "#e4e3df"
    shown = sub[sub["eyes"] > 0]
    keys = list(dict.fromkeys(zip(shown["subgroup"], shown["value"])))       # one row per group, both panels
    ypos = {k: len(keys) - 1 - i for i, k in enumerate(keys)}
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 0.36 * len(keys) + 1.3), dpi=200, sharey=True)
    for ax, (err, title) in zip(axes, (("missed_urgent", "Urgent eyes not referred urgently (%)"),
                                       ("ref_home", "Referable eyes sent home (%)"))):
        s = sub[(sub["error"] == err) & (sub["eyes"] > 0)]
        y = np.array([ypos[(a, b)] for a, b in zip(s["subgroup"], s["value"])])
        ax.errorbar(100 * s["rate"], y, xerr=[100 * (s["rate"] - s["lo"]), 100 * (s["hi"] - s["rate"])],
                    fmt="o", color="#2a78d6", ecolor="#86b6ef", elinewidth=2, ms=5, capsize=0)
        for yi, (_, r) in zip(y, s.iterrows()):
            ax.text(101, yi, f"n = {r['eyes']}", va="center", fontsize=6.8, color=ink2)
        first = s[s["subgroup"] == s["subgroup"].iloc[0]]
        overall = first["errors"].sum() / first["eyes"].sum()
        ax.axvline(100 * overall, color=ink2, lw=1, ls=(0, (4, 3)))
        ax.set_title(title, fontsize=9.5, color=ink, loc="left")
        ax.grid(axis="x", color=grid_c, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(ink2)
        ax.tick_params(colors=ink2, labelsize=7.5, length=0)
        ax.set_xlim(0, 100)
    axes[0].set_yticks([ypos[k] for k in keys])
    axes[0].set_yticklabels([f"{a}: {b}" for a, b in keys], fontsize=7.2, color=ink)
    fig.suptitle("C1 errors on APTOS by subgroup (95 % Wilson intervals; dashed: all eyes; n: eyes in the group)",
                 fontsize=10.5, color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
