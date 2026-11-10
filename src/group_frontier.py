#!/usr/bin/env python3
"""
group_frontier.py - additional work: do the baselines of group_robust.py lose only because of their
settings? Each method's own setting is swept, and the methods are compared at matched fairness.

group_robust.py (README section 23) runs every method at one standard setting. A fair comparison asks more: each
baseline has a knob that trades fairness for usefulness, so it might beat our weights at another setting. Here each
family is run at several settings under the same protocol as group_robust.py (E1's 10 outer folds, logistic
regression on standardised features, C1 on each fold's calibration photos, every APTOS photo decided out of fold,
training camera groups from each fold's training photos only). For each family we then take the setting that sends
the most healthy eyes home among those that meet both targets in every camera group (referable eyes sent home at
most 10 %, urgent eyes referred urgently at least 90 %). That setting is chosen with hindsight, on the very results
it is judged by, which favours the baselines. Our weights are judged at their principled setting (t = 1: the camera
then says nothing about the class); their own sweep is shown for reference.

Families and knobs (C = 0.003 unless the knob is C; the standard setting of group_robust.py is marked *):
  erm        plain training (once)
  decorr     w = (P(y) / P(y | g))^t,  t = 0.25, 0.5, 0.75, 1*, 1.5           (ours; t = 0 is plain training)
  balanced   w ~ (1 / n(y, g))^t,      t = 0.25, 0.5, 0.75, 1*                (t = 1: every cell weighs the same)
  groupdro   step eta = 0.05, 0.1, 0.5*, 1, 2                                 (20 rounds, last 10 averaged)
  jtt        error upweighting factor = 2, 5*, 10, 20*, 50                    (no camera labels)
  dfr        C = 0.003, 0.01, 0.03*, 0.1, 0.3                                 (the same 10 subsamples for every C)
The seeds are those of group_robust.py, so its standard settings are reproduced exactly (a built-in check).

Outputs (in --out, default ../results/group_frontier): summary.md, frontier.csv, frontier.png

Usage:
  python group_frontier.py --data ../data_512 --results ../results      # about 10 minutes on 2 cores
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import binomtest
from sklearn.preprocessing import StandardScaler

from conformal import HOME, URGENT, classwise_thresholds, decide_from_sets
from c1_referral import camera_groups
from dr_preprocess import assign_folds
from group_robust import C_FIX, GROUPS, auroc, averaged, camera_groups_from, cells, fit, proba, wilson

GRID = {"erm": [0.0], "decorr": [0.25, 0.5, 0.75, 1.0, 1.5], "balanced": [0.25, 0.5, 0.75, 1.0],
        "groupdro": [0.05, 0.1, 0.5, 1.0, 2.0], "jtt": [2.0, 5.0, 10.0, 20.0, 50.0],
        "dfr": [0.003, 0.01, 0.03, 0.1, 0.3]}
STANDARD = {"erm": [0.0], "decorr": [1.0], "balanced": [1.0], "groupdro": [0.5], "jtt": [5.0, 20.0], "dfr": [0.03]}
FAMILY = {"erm": "Plain training (ERM)", "decorr": "Camera-decorrelated weights (ours)",
          "balanced": "Group-balanced weights", "groupdro": "Group DRO", "jtt": "JTT", "dfr": "DFR"}
KNOB = {"erm": "–", "decorr": "strength t", "balanced": "strength t", "groupdro": "step", "jtt": "factor",
        "dfr": "C"}


def decorr_weights(y, g):
    p_y = np.bincount(y, minlength=3) / len(y)
    w = np.ones(len(y))
    for grp in np.unique(g):
        s = g == grp
        p_yg = np.bincount(y[s], minlength=3) / s.sum()
        for c in range(3):
            w[s & (y == c)] = p_y[c] / p_yg[c] if p_yg[c] > 0 else 0.0
    return w


def group_dro(Z, y, cell, eta):
    uniq, ci, cnt = np.unique(cell, return_inverse=True, return_counts=True)
    q = np.ones(len(uniq)) / len(uniq)
    model, params = None, []
    for t in range(20):
        w = q[ci] / cnt[ci]
        model = fit(Z, y, w / w.mean(), init=model)
        P = proba(model, Z)
        ll = -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None))
        L = np.bincount(ci, weights=ll) / cnt
        q = q * np.exp(eta * L)
        q /= q.sum()
        if t >= 10:
            params.append((model.coef_.copy(), model.intercept_.copy()))
    return averaged(params, model.classes_, Z.shape[1])


def fold_family(k, family, X, y, man, fit_idx, other_idx, seed):
    """Every setting of one family in one fold: {setting: probabilities of other_idx}."""
    g = camera_groups_from(man, fit_idx)[fit_idx]
    sc = StandardScaler().fit(X[fit_idx])
    Z, Zo, yy = sc.transform(X[fit_idx]), sc.transform(X[other_idx]), y[fit_idx]
    out = {}
    if family == "erm":
        out[0.0] = proba(fit(Z, yy), Zo)
    elif family == "decorr":
        base = decorr_weights(yy, g)
        for t in GRID[family]:
            w = base ** t
            out[t] = proba(fit(Z, yy, w / w.mean()), Zo)
    elif family == "balanced":
        _, inv, cnt = np.unique(cells(yy, g), return_inverse=True, return_counts=True)
        for t in GRID[family]:
            w = (1.0 / cnt[inv]) ** t
            out[t] = proba(fit(Z, yy, w / w.mean()), Zo)
    elif family == "groupdro":
        cell = cells(yy, g)
        for eta in GRID[family]:
            out[eta] = proba(group_dro(Z, yy, cell, eta), Zo)
    elif family == "jtt":
        inner = assign_folds(man.iloc[fit_idx].reset_index(drop=True), 5, seed)[0].to_numpy()
        oof = np.zeros((len(yy), 3))
        for j in range(5):
            tr, va = inner != j, inner == j
            oof[va] = proba(fit(Z[tr], yy[tr]), Z[va])
        err = oof.argmax(1) != yy
        for lam in GRID[family]:
            w = np.where(err, lam, 1.0)
            out[lam] = proba(fit(Z, yy, w / w.mean()), Zo)
    elif family == "dfr":
        cell = cells(yy, g)
        uniq = np.unique(cell)
        m = min(int(np.sum(cell == c)) for c in uniq)
        rng = np.random.default_rng(seed)
        subsamples = [np.concatenate([rng.choice(np.flatnonzero(cell == c), m, replace=False) for c in uniq])
                      for _ in range(10)]
        for C in GRID[family]:
            params = []
            for sel in subsamples:
                mm = fit(Z[sel], yy[sel], C=C)
                params.append((mm.coef_, mm.intercept_))
            out[C] = proba(averaged(params, np.arange(3), Z.shape[1]), Zo)
    else:
        raise ValueError(family)
    return k, family, out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Additional work: each baseline's setting swept; comparison at matched fairness.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", default="../data_512")
    ap.add_argument("--features", default="features/resnet50_224.npz")
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default="../results/group_frontier")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    t0 = time.time()
    out, R, a = Path(args.out), Path(args.results), args.alpha
    out.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(Path(args.data) / "manifest.csv")
    with np.load(Path(args.data) / args.features) as z:
        X, uid = z["X"], z["uid"]
    assert (uid == man["uid"].astype(str).to_numpy()).all()
    y = man["ref3"].to_numpy(int)
    g_all = camera_groups(man).fillna("").to_numpy()
    row = pd.Series(np.arange(len(man)), index=man["uid"])
    apt = np.flatnonzero((man["dataset"] == "aptos").to_numpy())
    folds = {}
    for k in range(10):
        f = pd.read_csv(R / "e1" / "lr" / f"fold{k}.csv.gz")
        folds[k] = {r: row[f.loc[f["role"] == r, "uid"]].to_numpy() for r in ("test", "cal")}
        folds[k]["order"] = row[f["uid"]].to_numpy()

    tasks = []
    for k, f in folds.items():
        fit_idx = np.setdiff1d(apt, np.concatenate([f["test"], f["cal"]]))
        for family in GRID:
            tasks.append(delayed(fold_family)(k, family, X, y, man, fit_idx, f["order"], args.seed + 1000 + k))
    res = Parallel(n_jobs=args.jobs, verbose=5)(tasks)
    P = {}
    for k, family, outk in res:
        for s, Pk in outk.items():
            P.setdefault((family, s), {})[k] = Pk
    print(f"models done ({time.time() - t0:.0f} s)", flush=True)

    # C1 per fold on the fold's calibration photos; metrics on all APTOS photos, out of fold
    ya, ga = y[apt], g_all[apt]
    ref, urg, non = ya >= 1, ya == 2, ya == 0
    rows, dec = [], {}
    for (family, s), Pf in P.items():
        Po, d = np.full((len(man), 3), np.nan), np.full(len(man), -1)
        for k, f in folds.items():
            pos = {u: i for i, u in enumerate(f["order"])}
            cal = np.array([pos[i] for i in f["cal"]])
            te = np.array([pos[i] for i in f["test"]])
            q = classwise_thresholds(Pf[k][cal], y[f["cal"]], a)
            Po[f["test"]] = Pf[k][te]
            d[f["test"]] = decide_from_sets((1 - Pf[k][te]) <= q[None, :])
        da, Pa = d[apt], Po[apt]
        dec[(family, s)] = da
        r = {"family": family, "setting": s, "standard": s in STANDARD[family],
             "auroc_ref": auroc(ref, Pa[:, 1:].sum(1)), "ref_home": np.mean(da[ref] == HOME),
             "urgent_urgent": np.mean(da[urg] == URGENT), "healthy_home": np.mean(da[non] == HOME)}
        worst, worst_u = (-1.0, None), 1.0
        for grp in GROUPS:
            sg = ga == grp
            n_r, k_r = int((sg & ref).sum()), int((sg & ref & (da == HOME)).sum())
            n_u, k_u = int((sg & urg).sum()), int((sg & urg & (da == URGENT)).sum())
            if k_r / n_r > worst[0]:
                worst = (k_r / n_r, grp, k_r, n_r)
            worst_u = min(worst_u, k_u / n_u)
        lo, hi = wilson(worst[2], worst[3])
        r.update({"worst_group": worst[1], "worst_ref_home": worst[0], "worst_lo": lo, "worst_hi": hi,
                  "worst_urgent_urgent": worst_u})
        r["meets_both"] = (r["worst_ref_home"] <= a + 1e-12) and (r["worst_urgent_urgent"] >= 1 - a - 1e-12)
        rows.append(r)
    fr = pd.DataFrame(rows).sort_values(["family", "setting"], key=lambda c: c.map(list(GRID).index)
                                        if c.name == "family" else c).reset_index(drop=True)

    # best setting per family, among those meeting both targets in every camera group (chosen with hindsight)
    ours = ("decorr", 1.0)
    best = []
    for family in GRID:
        sub = fr[(fr["family"] == family) & fr["meets_both"]]
        if not len(sub):
            best.append({"family": family, "setting": np.nan, "healthy_home": np.nan})
            continue
        b = sub.loc[sub["healthy_home"].idxmax()]
        m_home, o_home = dec[(family, b["setting"])][non] == HOME, dec[ours][non] == HOME
        only_m, only_o = int(np.sum(m_home & ~o_home)), int(np.sum(~m_home & o_home))
        p = binomtest(only_m, only_m + only_o, 0.5).pvalue if only_m + only_o else 1.0
        best.append({"family": family, "setting": b["setting"], "healthy_home": b["healthy_home"],
                     "worst_ref_home": b["worst_ref_home"], "worst_group": b["worst_group"],
                     "worst_urgent_urgent": b["worst_urgent_urgent"], "auroc_ref": b["auroc_ref"],
                     "home_only_family": only_m, "home_only_ours": only_o, "p": p})
    best = pd.DataFrame(best)
    fr.to_csv(out / "frontier.csv", index=False, float_format="%.5f")
    write_summary(out, fr, best, int(non.sum()), time.time() - t0)
    plot(fr, out / "frontier.png")
    print(f"wrote {out / 'summary.md'} ({time.time() - t0:.0f} s)")


def fmt_setting(family, s):
    if family == "erm":
        return "–"
    if family == "dfr":
        return f"C = {s:g}"
    if family == "jtt":
        return f"×{s:g}"
    if family == "groupdro":
        return f"step {s:g}"
    return f"t = {s:g}"


def write_summary(out, fr, best, n_healthy, elapsed):
    L = ["# Additional work: each baseline's setting swept, compared at matched fairness", "",
         "Same protocol as group_robust.py (E1's 10 outer folds; C1 on each fold's calibration photos; every APTOS "
         "photo decided out of fold). Both targets in every camera group: referable eyes sent home at most 10 %, "
         "urgent eyes referred urgently at least 90 %. * = the standard setting used in group_robust.py. In %.", "",
         "## 1. Best setting of each family that meets both targets in every camera group", "",
         "Chosen with hindsight on these results, which favours the baselines; ours is shown at its principled "
         f"setting (t = 1). Last column: exact McNemar on all {n_healthy} healthy eyes, sent home or not, against "
         "ours at t = 1 (eyes only that setting sends home / eyes only ours sends home).", "",
         "| Family | Best setting | Healthy sent home | Worst group: referable sent home | Worst group: urgent referred "
         "urgently | AUROC referable | Against ours (t = 1) |", "|---|---|---|---|---|---|---|"]
    for _, b in best.iterrows():
        if pd.isna(b["setting"]):
            L.append(f"| {FAMILY[b['family']]} | no setting meets both targets | | | | | |")
            continue
        vs = ("–" if (b["family"] == "decorr" and b["setting"] == 1.0)
              else f"{int(b['home_only_family'])} / {int(b['home_only_ours'])}, p = {b['p']:.2g}")
        L.append(f"| {FAMILY[b['family']]} | {fmt_setting(b['family'], b['setting'])} | {100 * b['healthy_home']:.1f} | "
                 f"{100 * b['worst_ref_home']:.1f} ({b['worst_group']}) | {100 * b['worst_urgent_urgent']:.1f} | "
                 f"{b['auroc_ref']:.3f} | {vs} |")
    L += ["", "## 2. Every setting", "",
          "| Family | Setting | Worst group: referable sent home (95 % CI) | Worst group: urgent referred urgently | "
          "Healthy sent home | AUROC referable | Both targets in every group |", "|---|---|---|---|---|---|---|"]
    for _, r in fr.iterrows():
        star = "*" if r["standard"] else ""
        L.append(f"| {FAMILY[r['family']]} | {fmt_setting(r['family'], r['setting'])}{star} | "
                 f"{100 * r['worst_ref_home']:.1f} ({100 * r['worst_lo']:.0f}–{100 * r['worst_hi']:.0f}), "
                 f"{r['worst_group']} | {100 * r['worst_urgent_urgent']:.1f} | {100 * r['healthy_home']:.1f} | "
                 f"{r['auroc_ref']:.3f} | {'yes' if r['meets_both'] else 'no'} |")
    L += ["", f"Made {time.strftime('%Y-%m-%d %H:%M')} by group_frontier.py ({elapsed:.0f} s).", ""]
    (out / "summary.md").write_text("\n".join(L))


def plot(fr, path):
    """Each family's settings as a line: worst camera group's referable eyes sent home against healthy eyes sent
    home. Filled markers meet both targets in every group; a ring marks the standard setting. Our settings are
    labelled, and for each baseline only its best setting inside the target; family names sit in empty space."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    halo = [patheffects.withStroke(linewidth=3, foreground="white")]
    ink, ink2, grid_c, accent = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    colours = {"erm": ink, "decorr": accent, "balanced": "#e07b39", "groupdro": "#3a9e6f", "jtt": "#8e5bb5",
               "dfr": "#9a8f3a"}
    # family label (x, y) in data units, and the setting its leader line points at
    names = {"erm": ((43.6, 91.5), 0.0), "decorr": ((10.8, 91.5), None), "balanced": ((40.0, 77.0), 0.25),
             "jtt": ((24.0, 69.0), 10.0), "groupdro": ((6.6, 69.3), None), "dfr": ((8.0, 24.0), None)}
    shown = {"erm": "Plain training (ERM)", "decorr": "Camera-decorrelated weights (ours)",
             "balanced": "Group-balanced weights", "jtt": "JTT (needs no camera labels)", "groupdro": "Group DRO",
             "dfr": "DFR"}
    fig, ax = plt.subplots(figsize=(8.6, 5.8), dpi=200)
    ax.axvspan(0, 10, color=accent, alpha=0.06, lw=0)
    ax.axvline(10, color=ink, lw=0.9, ls=(0, (4, 3)))
    ax.text(9.6, 2.5, "every camera group\nmeets the 10 % target", ha="right", va="bottom", fontsize=7.5,
            color=ink2)
    for family in GRID:
        sub = fr[fr["family"] == family].sort_values("setting")
        col = colours[family]
        xs, ys = 100 * sub["worst_ref_home"].to_numpy(), 100 * sub["healthy_home"].to_numpy()
        if len(sub) > 1:
            ax.plot(xs, ys, color=col, lw=1.6 if family == "decorr" else 1.0, alpha=0.9, zorder=2)
        feasible = sub[sub["meets_both"]]
        best_s = feasible.loc[feasible["healthy_home"].idxmax(), "setting"] if len(feasible) else None
        for (_, r), xv, yv in zip(sub.iterrows(), xs, ys):
            ax.plot([xv], [yv], "o", ms=6 if family == "decorr" else 4.8, mec=col, mew=1.2,
                    mfc=col if r["meets_both"] else "white", zorder=3)
            if r["standard"]:
                ax.plot([xv], [yv], "o", ms=11, mfc="none", mec=col, mew=0.8, zorder=3)
            if family == "decorr" or (family != "erm" and r["setting"] == best_s):
                txt = fmt_setting(family, r["setting"]) + ("" if family == "decorr" else " (best)")
                left = family == "decorr" and r["setting"] == 1.0
                ax.annotate(txt, (xv, yv), xytext=(-7, 3) if left else (5, 3), textcoords="offset points",
                            fontsize=6.6, color=col, ha="right" if left else "left", path_effects=halo, zorder=4)
        (lx, ly), target = names[family]
        ax.text(lx, ly, shown[family], ha="center" if family == "erm" else "left", va="center", fontsize=7.6,
                color=col, fontweight="bold" if family == "decorr" else "normal", path_effects=halo, zorder=4)
        if target is not None and family != "erm":
            r = sub[sub["setting"] == target].iloc[0]
            ax.plot([100 * r["worst_ref_home"], lx - 0.3], [100 * r["healthy_home"], ly], color=col, lw=0.5,
                    zorder=1)
    ax.set_xlim(0, max(50, 100 * fr["worst_ref_home"].max() + 4))
    ax.set_ylim(0, 100)
    ax.set_xlabel("worst camera group: referable eyes sent home (%)", fontsize=8.5, color=ink2)
    ax.set_ylabel("healthy eyes sent home (%), all cameras", fontsize=8.5, color=ink2)
    ax.grid(color=grid_c, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=ink2, labelsize=8, length=0)
    ax.set_title("Each method's own setting swept (filled: both targets met in every camera group; ring: standard "
                 "setting)", fontsize=9.5, color=ink, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
