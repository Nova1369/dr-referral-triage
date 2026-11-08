#!/usr/bin/env python3
"""
summarize_e1.py - tables and figures for E1 from the predictions saved by run_models.py.

Every APTOS image is predicted once, by the model of the outer fold that held it out; metrics are
computed on these pooled out-of-fold predictions. 95 % confidence intervals come from a bootstrap that
resamples whole groups (copies of a photo and same-eye photos together), with the same resamples for
every model, so differences between models are paired.

Metrics (our three classes: 0 no referral, 1 routine, 2 urgent):
  AUROC referable   P(routine) + P(urgent) against grade >= 2          (raw probabilities)
  AUROC urgent      P(urgent) against grade >= 3                        (raw probabilities)
  QWK, accuracy     the most likely class (argmax of the raw probabilities)
  Sens / Spec       referable sensitivity and specificity, urgent sensitivity, at the argmax
  Refer / 1,000     eyes referred (routine or urgent) per 1,000 screened, at the argmax
  Brier             multi-class Brier score, raw -> isotonic-calibrated probabilities
  ECE referable     expected calibration error of P(referable), 10 equal-width bins, raw -> calibrated
B0 (refer none / refer all urgently) is added here; it needs no training.

Outputs (in --out, default: the results folder):
  summary.md        the tables, with notes on the protocol
  metrics.csv       every metric with its 95 % CI, plus the difference to logistic regression
  per_fold.csv      AUROCs in each outer fold
  best_params.csv   the setting chosen in each outer fold, and whether it was at the edge of the grid
  auroc_models.png  AUROC with 95 % CIs for every trained model
  same_camera_auroc.png   referable AUROC over all pairs against pairs from the same camera

Usage:
  python summarize_e1.py --data ../data_512 --results ../results/e1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

NAMES = {"b1": "B1 Most likely class", "b2": "B2 Image quality only", "b2a": "B2a Image size only",
         "b2b": "B2b Quality, without size", "lr": "Logistic regression", "gnb": "Naive Bayes", "lda": "LDA",
         "qda": "QDA", "knn": "k-NN", "cart": "Decision tree", "rf": "Random forest", "ada": "AdaBoost",
         "gb": "Gradient boosting", "svm": "SVM (RBF)", "mlp": "MLP", "lr-grade": "Logistic regression, 5 grades"}
BASELINES = ["b0-none", "b0-all", "b1", "b2", "b2a", "b2b"]
B0_NAMES = {"b0-none": "B0 Refer none", "b0-all": "B0 Refer all (urgent)"}
REFERENCE = "lr"


# ------------------------------------------------------------------------------------------------
# metrics
# ------------------------------------------------------------------------------------------------

def auroc(pos, score):
    n1 = pos.sum()
    n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auroc_within(pos, score, strata):
    """AUROC over pairs taken with the same camera only: per-camera AUROCs weighted by their pair counts."""
    num = den = 0.0
    for s in strata:
        n1 = pos[s].sum()
        n0 = len(s) - n1
        if n1 and n0:
            num += auroc(pos[s], score[s]) * n1 * n0
            den += n1 * n0
    return num / den if den else np.nan


def qwk(y, pred, k=3):
    O = np.bincount(y * k + pred, minlength=k * k).reshape(k, k).astype(float)
    W = np.subtract.outer(np.arange(k), np.arange(k)) ** 2 / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (W * E).sum()
    return 1 - (W * O).sum() / den if den > 0 else np.nan


def ece(pos, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    gap = np.abs(np.bincount(b, pos.astype(float), bins) - np.bincount(b, p, bins))
    return gap.sum() / len(p)


def brier(y, P):
    onehot = np.eye(P.shape[1])[y]
    return ((P - onehot) ** 2).sum(1).mean()


def metrics(y, P, C, probabilistic=True, strata=None):
    pred = P.argmax(1)
    ref, urg = y >= 1, y == 2
    out = {
        "auroc_referable": auroc(ref, P[:, 1:].sum(1)) if probabilistic else np.nan,
        "auroc_urgent": auroc(urg, P[:, 2]) if probabilistic else np.nan,
        "qwk": qwk(y, pred), "accuracy": (pred == y).mean(),
        "sens_referable": (pred[ref] >= 1).mean(), "spec_referable": (pred[~ref] == 0).mean(),
        "sens_urgent": (pred[urg] == 2).mean(), "refer_per_1000": 1000 * (pred >= 1).mean(),
        "brier_raw": brier(y, P) if probabilistic else np.nan,
        "brier_cal": brier(y, C) if probabilistic else np.nan,
        "ece_referable_raw": ece(ref, P[:, 1:].sum(1)) if probabilistic else np.nan,
        "ece_referable_cal": ece(ref, C[:, 1:].sum(1)) if probabilistic else np.nan,
    }
    if strata is not None:
        out["auroc_referable_same_camera"] = auroc_within(ref, P[:, 1:].sum(1), strata) if probabilistic else np.nan
        out["auroc_urgent_same_camera"] = auroc_within(urg, P[:, 2], strata) if probabilistic else np.nan
    return out


# ------------------------------------------------------------------------------------------------
# loading
# ------------------------------------------------------------------------------------------------

def load_predictions(folder: Path, n_folds: int):
    """Pooled test predictions of one model, as (raw, calibrated) n x 3 arrays indexed by uid."""
    files = sorted(folder.glob("fold*.csv.gz"))
    files = [f for f in files if f.name[4:].split(".")[0].isdigit()]
    if len(files) < n_folds:
        return None
    df = pd.concat([pd.read_csv(f).assign(fold=int(f.name[4:].split(".")[0])) for f in files])
    test = df[df["role"] == "test"].set_index("uid")
    k = sum(c.startswith("p") and c[1:].isdigit() for c in test.columns)
    P, C = test[[f"p{c}" for c in range(k)]].to_numpy(), test[[f"c{c}" for c in range(k)]].to_numpy()
    if k == 5:                                       # 5-grade model: collapse to our three classes
        P = np.column_stack([P[:, :2].sum(1), P[:, 2], P[:, 3:].sum(1)])
        C = np.column_stack([C[:, :2].sum(1), C[:, 2], C[:, 3:].sum(1)])
    return test.index, P, C, test["fold"].to_numpy()


def grid_edges(folder: Path):
    """Chosen setting per fold, and which hyper-parameters sat at the edge of their grid."""
    rows = []
    for f in sorted(folder.glob("fold*_tuning.csv")):
        t = pd.read_csv(f)
        cfgs = [json.loads(c) for c in t["config"]]
        best = cfgs[int(np.flatnonzero(t["best"].to_numpy())[0])]
        edges = []
        for key, val in best.items():
            values = sorted({json.dumps(c[key]) for c in cfgs}, key=lambda s: (len(s), s))
            nums = [json.loads(v) for v in values]
            if len(nums) > 2 and all(isinstance(v, (int, float)) for v in nums):
                if key == "pca" and val >= 2048:          # all components: nothing left to widen
                    continue
                if val in (min(nums), max(nums)):
                    edges.append(f"{key}={val}")
        rows.append({"fold": int(f.name[4:].split("_")[0]), "best": json.dumps(best),
                     "inner_score": float(t.loc[t["best"], "score"].iloc[0]), "at_grid_edge": " ".join(edges),
                     "warnings": str(t.loc[t["best"], "warnings"].iloc[0]) if "warnings" in t else ""})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------

def fmt_ci(v, lo, hi, digits=3):
    if np.isnan(v):
        return "–"
    return f"{v:.{digits}f} ({lo:.{digits}f}–{hi:.{digits}f})"


def main(argv=None):
    ap = argparse.ArgumentParser(description="E1 tables and figures.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", default="../data_512")
    ap.add_argument("--results", default="../results/e1")
    ap.add_argument("--out", default=None, help="default: the results folder")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    t0 = time.time()
    res = Path(args.results)
    out = Path(args.out) if args.out else res
    out.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(Path(args.data) / "manifest.csv")
    apt = man[(man["dataset"] == "aptos") & (man["fold"] >= 0)].reset_index(drop=True)
    y = apt["ref3"].to_numpy(int)
    n_folds = apt["fold"].nunique()

    # ---- collect predictions ----
    models, edges = {}, {}
    for key in list(NAMES):
        d = res / key
        if not d.is_dir():
            continue
        got = load_predictions(d, n_folds)
        if got is None:
            print(f"skipping {key}: fewer than {n_folds} folds finished")
            continue
        uid, P, C, fold = got
        pos = pd.Index(uid).get_indexer(apt["uid"])
        if (pos < 0).any() or len(uid) != len(apt):
            raise SystemExit(f"{key}: test predictions do not cover every APTOS image exactly once")
        models[key] = (P[pos], C[pos], fold[pos])
        edges[key] = grid_edges(d)
    for key, cls in (("b0-none", 0), ("b0-all", 2)):
        onehot = np.eye(3)[np.full(len(apt), cls)]
        models[key] = (onehot, onehot, apt["fold"].to_numpy())
    print(f"{len(models)} models/baselines, {len(apt)} APTOS images")

    # ---- point estimates, per-fold AUROCs ----
    camera = pd.factorize(apt["orig_w"].astype(str) + "x" + apt["orig_h"].astype(str))[0]
    strata_of = lambda cam: [np.flatnonzero(cam == c) for c in np.unique(cam)]  # noqa: E731
    strata = strata_of(camera)
    point = {k: metrics(y, P, C, not k.startswith("b0"), strata) for k, (P, C, _) in models.items()}
    per_fold = []
    for k, (P, C, fold) in models.items():
        if k.startswith("b0"):
            continue
        for f in range(n_folds):
            s = fold == f
            per_fold.append({"model": k, "fold": f, "auroc_referable": auroc(y[s] >= 1, P[s, 1:].sum(1)),
                             "auroc_urgent": auroc(y[s] == 2, P[s, 2])})
    per_fold = pd.DataFrame(per_fold)

    # ---- group bootstrap (same resamples for every model) ----
    groups = pd.factorize(apt["group_id"])[0]
    G = groups.max() + 1
    rng = np.random.default_rng(args.seed)
    boot = {k: [] for k in models}
    for _ in range(args.boot):
        w = np.bincount(rng.integers(0, G, G), minlength=G)[groups]
        idx = np.repeat(np.arange(len(apt)), w)
        st = strata_of(camera[idx])
        for k, (P, C, _) in models.items():
            boot[k].append(metrics(y[idx], P[idx], C[idx], not k.startswith("b0"), st))
    boot = {k: pd.DataFrame(v) for k, v in boot.items()}

    rows = []
    for k in models:
        r = {"model": k, "name": B0_NAMES.get(k, NAMES.get(k, k))}
        for m, v in point[k].items():
            lo, hi = np.nanpercentile(boot[k][m], [2.5, 97.5]) if not np.isnan(v) else (np.nan, np.nan)
            r[m], r[m + "_lo"], r[m + "_hi"] = v, lo, hi
        if REFERENCE in models and not k.startswith("b0") and k != REFERENCE:
            for m in ("auroc_referable", "auroc_urgent"):
                d = boot[k][m] - boot[REFERENCE][m]
                r[f"{m}_minus_lr"] = point[k][m] - point[REFERENCE][m]
                r[f"{m}_minus_lr_lo"], r[f"{m}_minus_lr_hi"] = np.percentile(d, [2.5, 97.5])
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab["mean_auroc"] = tab[["auroc_referable", "auroc_urgent"]].mean(axis=1)
    base = tab[tab["model"].isin(BASELINES)].set_index("model").loc[[b for b in BASELINES if b in models]]
    course = tab[~tab["model"].isin(BASELINES)].sort_values("mean_auroc", ascending=False).set_index("model")
    tab = pd.concat([base, course]).reset_index()
    tab.to_csv(out / "metrics.csv", index=False, float_format="%.4f")
    per_fold.to_csv(out / "per_fold.csv", index=False, float_format="%.4f")
    pd.concat([e.assign(model=k) for k, e in edges.items()])[
        ["model", "fold", "best", "inner_score", "at_grid_edge", "warnings"]].to_csv(
        out / "best_params.csv", index=False, float_format="%.4f")

    # ---- figure ----
    is_b2 = tab["model"].str.startswith("b2")
    is_course = ~tab["model"].isin(BASELINES)
    fig_rows = pd.concat([tab[is_b2].iloc[::-1], tab[is_course].iloc[::-1]])
    plot_auroc(fig_rows, int(is_b2.sum()), out / "auroc_models.png")
    plot_shortcut(fig_rows, int(is_b2.sum()), out / "same_camera_auroc.png")

    # ---- summary.md ----
    def ci(r, m, d=3):
        return fmt_ci(r[m], r[m + "_lo"], r[m + "_hi"], d)

    def pct(r, m):
        return "–" if np.isnan(r[m]) else f"{100 * r[m]:.1f}"

    lines = [
        "# E1: baselines and course models (APTOS, 10-fold cross-validation)", "",
        f"{len(apt)} APTOS images, each predicted once by the model of the outer fold that held it out. "
        f"95 % CIs: {args.boot} group-bootstrap resamples. Classes: no referral (grade 0–1, {np.mean(y == 0):.1%}), "
        f"routine (grade 2, {np.mean(y == 1):.1%}), urgent (grade 3–4, {np.mean(y == 2):.1%}).", "",
        "## Ranking and agreement", "",
        "| Model | AUROC referable | AUROC urgent | QWK | Accuracy |", "|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        lines.append(f"| {r['name']} | {ci(r, 'auroc_referable')} | {ci(r, 'auroc_urgent')} | "
                     f"{ci(r, 'qwk')} | {ci(r, 'accuracy')} |")
    lines += ["", "## Decisions at the most likely class, and calibration", "",
              "Sensitivity and specificity in %. These use the plain argmax (no tuned threshold, no conformal step): "
              "they show what the models do before C1.", "",
              "| Model | Sens. referable | Spec. referable | Sens. urgent | Referred per 1,000 | Brier raw → calibrated | ECE referable raw → calibrated |",
              "|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        br = "–" if np.isnan(r["brier_raw"]) else f"{r['brier_raw']:.3f} → {r['brier_cal']:.3f}"
        ec = "–" if np.isnan(r["ece_referable_raw"]) else \
            f"{r['ece_referable_raw']:.3f} → {r['ece_referable_cal']:.3f}"
        lines.append(f"| {r['name']} | {pct(r, 'sens_referable')} | {pct(r, 'spec_referable')} | "
                     f"{pct(r, 'sens_urgent')} | {r['refer_per_1000']:.0f} | {br} | {ec} |")
    # confusion matrices of the reference model and the best trained model (most likely class)
    course_keys = [k for k in tab["model"] if k not in BASELINES]
    for k in dict.fromkeys([REFERENCE] + course_keys[:1]):
        if k not in models:
            continue
        pred = models[k][0].argmax(1)
        cm = np.bincount(y * 3 + pred, minlength=9).reshape(3, 3)
        lines += ["", f"## Confusion matrix: {NAMES.get(k, k)} (most likely class)", "",
                  "Rows: true class; columns: predicted class.", "",
                  "| True \\ predicted | No referral | Routine | Urgent | Total |", "|---|---|---|---|---|"]
        for i, nm in enumerate(["No referral", "Routine", "Urgent"]):
            lines.append(f"| {nm} | " + " | ".join(str(v) for v in cm[i]) + f" | {cm[i].sum()} |")
    lines += ["", "## Shortcut check: AUROC within the same camera", "",
              "APTOS photo size identifies the camera, and the cameras differ hugely in how sick their patients are "
              "(see results/kmeans). Here a referable eye is only compared with a non-referable eye taken with the "
              "same camera (per-camera AUROCs weighted by their number of pairs), so the camera alone cannot help. "
              "A model that only knows the camera scores 0.5.", "",
              "| Model | AUROC referable, all pairs | Same camera | AUROC urgent, all pairs | Same camera |",
              "|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        if np.isnan(r["auroc_referable"]):
            continue
        lines.append(f"| {r['name']} | {r['auroc_referable']:.3f} | {ci(r, 'auroc_referable_same_camera')} | "
                     f"{r['auroc_urgent']:.3f} | {ci(r, 'auroc_urgent_same_camera')} |")
    if REFERENCE in models:
        lines += ["", "## Difference to logistic regression (reference model)", "",
                  "AUROC difference with its paired 95 % CI; a CI that excludes 0 marks a real difference.", "",
                  "| Model | Δ AUROC referable | Δ AUROC urgent |", "|---|---|---|"]
        for _, r in tab.iterrows():
            if "auroc_referable_minus_lr" in r and not pd.isna(r.get("auroc_referable_minus_lr")):
                lines.append(f"| {r['name']} | {r['auroc_referable_minus_lr']:+.3f} "
                             f"({r['auroc_referable_minus_lr_lo']:+.3f} to {r['auroc_referable_minus_lr_hi']:+.3f}) | "
                             f"{r['auroc_urgent_minus_lr']:+.3f} "
                             f"({r['auroc_urgent_minus_lr_lo']:+.3f} to {r['auroc_urgent_minus_lr_hi']:+.3f}) |")
    lines += ["", "## Chosen settings", "",
              "Most frequent setting over the 10 outer folds, and how often a chosen value sat at the edge of "
              "its grid (a sign the grid could be widened).", "",
              "| Model | Most frequent setting | Folds | At grid edge |", "|---|---|---|---|"]
    for k in tab["model"]:
        if k not in edges:
            continue
        e = edges[k]
        top = e["best"].value_counts()
        n_edge = (e["at_grid_edge"] != "").sum()
        which = " ".join(sorted({w for s in e["at_grid_edge"] for w in s.split()}))
        lines.append(f"| {NAMES.get(k, k)} | `{top.index[0]}` | {top.iloc[0]}/{len(e)} | "
                     f"{n_edge}/{len(e)}{' (' + which + ')' if which else ''} |")
    lines += ["", "## Protocol", "",
              "- Outer folds: the `fold` column of `manifest.csv` (10 folds, stratified by grade, groups never split).",
              "- In each outer fold, 15 % of the training images (by group) are kept for the conformal step (C1); "
              "the models never see them. Hyper-parameters are tuned by 5-fold grouped cross-validation on the "
              "other 85 %, choosing the best mean of the two AUROCs; the best setting is refitted on all 85 %.",
              "- ResNet-50 features: standardised and reduced with PCA inside every training split.",
              "- B2 uses only the columns of `manifest.csv` that describe the photo, not the retina: "
              "B2a the photo and retina size, B2b brightness, contrast, colour and sharpness.",
              "- Calibration: one-vs-rest isotonic regression fitted on the inner out-of-fold predictions.",
              "- Copies of the same photo with different grades are all kept (each copy with its own grade); "
              "they always fall in the same fold.",
              f"- Made {time.strftime('%Y-%m-%d %H:%M')} by summarize_e1.py."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'summary.md'} ({time.time() - t0:.0f} s)")


def plot_auroc(rows, n_base, path):
    """Forest plot, best model at the top, the B2 baselines below a rule at the bottom."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid_c = "#0b0b0b", "#52514e", "#e4e3df"
    colors = {"auroc_referable": "#2a78d6", "auroc_urgent": "#eb6834"}
    titles = {"auroc_referable": "Referable DR (grade ≥ 2)", "auroc_urgent": "Urgent (grade 3–4)"}
    n = len(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.36 * n + 1.5), sharey=True, dpi=200)
    lo_all = min(rows["auroc_referable_lo"].min(), rows["auroc_urgent_lo"].min())
    for ax, m in zip(axes, colors):
        yy = np.arange(n)
        ax.hlines(yy, rows[m + "_lo"], rows[m + "_hi"], color=colors[m], lw=2)
        ax.plot(rows[m], yy, "o", color=colors[m], ms=6, mec="white", mew=1.2)
        for yi, v in zip(yy, rows[m]):
            ax.text(rows[m + "_hi"].iloc[yi] + 0.004, yi, f"{v:.3f}", va="center", fontsize=7.5, color=ink2)
        if n_base:
            ax.axhline(n_base - 0.5, color=ink2, lw=0.6)
        ax.set_title(titles[m], fontsize=10, color=ink, loc="left")
        ax.set_xlim(max(0.5, np.floor(lo_all * 20) / 20), 1.0)
        ax.grid(axis="x", color=grid_c, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(ink2)
        ax.tick_params(colors=ink2, labelsize=8, length=0)
        ax.set_xlabel("AUROC (95 % CI)", fontsize=8.5, color=ink2)
    axes[0].set_yticks(np.arange(n))
    axes[0].set_yticklabels(rows["name"], fontsize=8.5, color=ink)
    fig.suptitle("E1: APTOS 10-fold cross-validation, pooled out-of-fold predictions", fontsize=10.5,
                 color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_shortcut(rows, n_base, path):
    """Referable AUROC over all pairs (filled) and within the same camera (open), one row per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid_c = "#0b0b0b", "#52514e", "#e4e3df"
    c_all, c_cam = "#2a78d6", "#eb6834"
    n = len(rows)
    yy = np.arange(n)
    fig, ax = plt.subplots(figsize=(7.5, 0.36 * n + 1.6), dpi=200)
    a, c = rows["auroc_referable"].to_numpy(), rows["auroc_referable_same_camera"].to_numpy()
    ax.hlines(yy, c, a, color=grid_c, lw=3, zorder=1)
    ax.hlines(yy, rows["auroc_referable_same_camera_lo"], rows["auroc_referable_same_camera_hi"],
              color=c_cam, lw=1.2, zorder=2)
    ax.plot(c, yy, "o", ms=6.5, mfc="white", mec=c_cam, mew=1.8, zorder=3, label="Same camera only")
    ax.plot(a, yy, "o", ms=6.5, color=c_all, mec="white", mew=1.2, zorder=3, label="All pairs")
    c_lo = rows["auroc_referable_same_camera_lo"].to_numpy()
    for yi in yy:
        ax.text(a[yi] + 0.006, yi, f"{a[yi]:.3f}", va="center", fontsize=7.5, color=ink2)
        ax.text(min(c[yi], c_lo[yi]) - 0.008, yi, f"{c[yi]:.3f}", va="center", ha="right", fontsize=7.5,
                color=ink2, bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=4)
    if n_base:
        ax.axhline(n_base - 0.5, color=ink2, lw=0.6)
    ax.axvline(0.5, color=ink2, lw=0.8, ls=(0, (3, 3)))
    ax.set_yticks(yy)
    ax.set_yticklabels(rows["name"], fontsize=8.5, color=ink)
    ax.set_xlim(0.42, 1.02)
    ax.grid(axis="x", color=grid_c, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(ink2)
    ax.tick_params(colors=ink2, labelsize=8, length=0)
    ax.set_xlabel("AUROC, referable DR (open circles: 95 % CI shown)", fontsize=8.5, color=ink2)
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=2, bbox_to_anchor=(0, 1.0))
    ax.set_title("Camera shortcut on APTOS: comparing eyes from the same camera only", fontsize=10,
                 color=ink, loc="left", pad=22)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
