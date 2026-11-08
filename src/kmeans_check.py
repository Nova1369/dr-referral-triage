#!/usr/bin/env python3
"""
kmeans_check.py - the unsupervised course method (k-means): do the images group by DR grade, or by
clinic and camera?

All 5,922 images: ResNet-50 features standardised and reduced to --pca components (default 50), then
k-means for k = 2..6. Agreement between the clusters and each labelling is measured with the adjusted
Rand index (ARI: 0 = chance, 1 = identical partitions) and normalised mutual information (NMI: 0..1).
The same is repeated within APTOS alone, against our classes, the grades and the camera (photo size
is used as a proxy for the camera, since APTOS does not name them).

This is a descriptive check: nothing here is used for prediction.

Outputs (in --out, default ../results/kmeans): summary.md, kmeans.csv, aptos_by_camera.csv, pca_scatter.png,
aptos_cameras.png

Usage:
  python kmeans_check.py --data ../data_512
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

CLASS_NAMES = ["no referral (0–1)", "routine (2)", "urgent (3–4)"]
DATASET_NAMES = {"aptos": "APTOS", "idrid": "IDRiD", "messidor2": "Messidor-2"}


def camera_proxy(man, n_top=8):
    """Photo size as a camera proxy: the commonest sizes, everything else 'other'."""
    size = man["orig_w"].astype(str) + "x" + man["orig_h"].astype(str)
    top = size.value_counts().index[:n_top]
    return size.where(size.isin(top), "other")


def agreement(clusters, labels):
    return adjusted_rand_score(labels, clusters), normalized_mutual_info_score(labels, clusters)


def main(argv=None):
    ap = argparse.ArgumentParser(description="k-means check: grade or clinic?",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", default="../data_512")
    ap.add_argument("--features", default="features/resnet50_224.npz", help="relative to --data")
    ap.add_argument("--out", default="../results/kmeans")
    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(Path(args.data) / "manifest.csv")
    with np.load(Path(args.data) / args.features) as z:
        X, uid = z["X"], z["uid"]
    assert (uid == man["uid"].astype(str).to_numpy()).all(), "features do not match manifest.csv"

    def embed(rows):
        Z = StandardScaler().fit_transform(X[rows])
        pca = PCA(n_components=args.pca, svd_solver="full", random_state=args.seed).fit(Z)
        return pca.transform(Z), pca.explained_variance_ratio_

    rows = []
    every = np.ones(len(man), bool)
    apt = (man["dataset"] == "aptos").to_numpy()
    cam = camera_proxy(man[apt]).reindex(man.index)          # APTOS photo sizes (only used within APTOS)
    E_all, var_all = embed(every)
    E_apt, _ = embed(apt)
    labelings = {
        "all images": (E_all, every, {"dataset": man["dataset"], "class": man["ref3"], "grade": man["grade"]}),
        "APTOS only": (E_apt, apt, {"class": man["ref3"], "grade": man["grade"], "camera (photo size)": cam}),
    }
    clusters = {}
    for scope, (E, sel, labs) in labelings.items():
        for k in range(2, 7):
            c = KMeans(n_clusters=k, n_init=20, random_state=args.seed).fit_predict(E)
            clusters[scope, k] = c
            for name, lab in labs.items():
                ari, nmi = agreement(c, lab.to_numpy()[sel])
                rows.append({"scope": scope, "k": k, "labels": name, "ari": ari, "nmi": nmi})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "kmeans.csv", index=False, float_format="%.4f")

    lines = ["# k-means check: do the images group by grade or by clinic?", "",
             f"ResNet-50 features, standardised, PCA to {args.pca} components "
             f"({var_all.sum():.0%} of the variance for all images), k-means with 20 restarts. "
             "ARI: 0 = chance, 1 = the clusters are the labels. NMI: 0–1.", ""]
    for scope in labelings:
        t = tab[tab["scope"] == scope]
        names = list(labelings[scope][2])
        lines += [f"## {scope}", "", "| k | " + " | ".join(f"ARI {n}" for n in names) + " | "
                  + " | ".join(f"NMI {n}" for n in names) + " |", "|---|" + "---|" * (2 * len(names))]
        for k, g in t.groupby("k"):
            g = g.set_index("labels")
            lines.append(f"| {k} | " + " | ".join(f"{g.loc[n, 'ari']:.3f}" for n in names) + " | "
                         + " | ".join(f"{g.loc[n, 'nmi']:.3f}" for n in names) + " |")
        lines.append("")
    c = clusters["all images", 3]
    ct_d2 = pd.crosstab(pd.Series(clusters["all images", 2], name="cluster"),
                        man["dataset"].map(DATASET_NAMES).rename("dataset"))
    ct_d = pd.crosstab(pd.Series(c, name="cluster"), man["dataset"].map(DATASET_NAMES).rename("dataset"))
    ct_c = pd.crosstab(pd.Series(c, name="cluster"),
                       man["ref3"].map(dict(enumerate(CLASS_NAMES))).rename("class"))[CLASS_NAMES]
    for title, ct in (("Clusters (k = 2, all images) by dataset", ct_d2), ("Clusters (k = 3, all images) by dataset", ct_d),
                      ("Clusters (k = 3, all images) by class", ct_c)):
        lines += [f"## {title}", "", "| Cluster | " + " | ".join(ct.columns) + " |", "|---|" + "---|" * ct.shape[1]]
        for i, r in ct.iterrows():
            lines.append(f"| {i} | " + " | ".join(str(v) for v in r) + " |")
        lines.append("")
    # APTOS cameras: how sick each camera's patients are, and where its photos land (k = 2, all images)
    c2 = clusters["all images", 2]
    mess_cluster = pd.Series(c2[~apt & (man["dataset"] == "messidor2").to_numpy()]).mode()[0]
    a = man[apt].assign(camera=cam[apt], same=(c2[apt] == mess_cluster))
    by_cam = a.groupby("camera").agg(n=("uid", "size"), referable=("ref3", lambda v: (v >= 1).mean()),
                                     urgent=("ref3", lambda v: (v == 2).mean()), same=("same", "mean"))
    by_cam = by_cam.sort_values("referable")
    lines += ["## APTOS by camera (photo size)", "",
              "Share of referable and urgent eyes per photo size, and share of the photos that k-means (k = 2, "
              "all images) puts in the same cluster as almost all Messidor-2 photos.", "",
              "| Photo size | Images | Referable | Urgent | In the Messidor-2 cluster |", "|---|---|---|---|---|"]
    for size, r in by_cam.iterrows():
        lines.append(f"| {size} | {r['n']:.0f} | {r['referable']:.0%} | {r['urgent']:.0%} | {r['same']:.0%} |")
    lines.append("")
    by_cam.to_csv(out / "aptos_by_camera.csv", float_format="%.4f")
    plot_cameras(E_all, man, apt, cam, by_cam, out / "aptos_cameras.png", args.seed)
    lines.append(f"Made {time.strftime('%Y-%m-%d %H:%M')} by kmeans_check.py.")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    plot(E_all, man, out / "pca_scatter.png", args.seed)
    print(f"wrote {out / 'summary.md'}")


def plot(E, man, path, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid_c = "#0b0b0b", "#52514e", "#e4e3df"
    order = np.random.default_rng(seed).permutation(len(man))          # no group is always drawn on top
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), dpi=200, sharex=True, sharey=True)
    panels = [("By dataset (clinic)", man["dataset"].map(DATASET_NAMES),
               {"APTOS": "#2a78d6", "IDRiD": "#eb6834", "Messidor-2": "#1baf7a"}),
              ("By referral class", man["ref3"].map(dict(enumerate(CLASS_NAMES))),
               dict(zip(CLASS_NAMES, ["#86b6ef", "#2a78d6", "#104281"])))]
    for ax, (title, lab, cmap) in zip(axes, panels):
        lab = lab.to_numpy()[order]
        ax.scatter(E[order, 0], E[order, 1], s=3, c=[cmap[v] for v in lab], alpha=0.55, linewidths=0)
        for name, col in cmap.items():
            ax.scatter([], [], s=30, c=col, label=f"{name} (n = {(lab == name).sum():,})")
        ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.92, fontsize=8,
                  loc="upper right", markerscale=1)
        ax.set_title(title, fontsize=10, color=ink, loc="left")
        ax.set_xlabel("Principal component 1", fontsize=8.5, color=ink2)
        ax.grid(color=grid_c, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(ink2)
        ax.tick_params(colors=ink2, labelsize=8)
    axes[0].set_ylabel("Principal component 2", fontsize=8.5, color=ink2)
    fig.suptitle("ResNet-50 features of all 5,922 images, first two principal components", fontsize=10.5,
                 color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_cameras(E, man, apt, cam, by_cam, path, seed):
    """APTOS photos on the first two principal components, coloured by how sick their camera's patients are."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid_c = "#0b0b0b", "#52514e", "#e4e3df"
    share = by_cam["referable"]
    groups = [("cameras with ≤ 10 % referable eyes", share <= 0.10, "#86b6ef"),
              ("cameras with 10–80 % referable", (share > 0.10) & (share < 0.80), "#2a78d6"),
              ("cameras with ≥ 80 % referable", share >= 0.80, "#104281")]
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=200)
    other = ~apt
    ax.scatter(E[other, 0], E[other, 1], s=3, c="#d6d5d0", alpha=0.5, linewidths=0,
               label=f"IDRiD and Messidor-2 (n = {other.sum():,})")
    idx = np.flatnonzero(apt)
    idx = idx[np.random.default_rng(seed).permutation(len(idx))]
    cam_a = cam.to_numpy()[idx]
    for name, mask, col in groups:
        sizes = set(share.index[mask])
        sel = np.isin(cam_a, list(sizes))
        ax.scatter(E[idx[sel], 0], E[idx[sel], 1], s=4, c=col, alpha=0.7, linewidths=0,
                   label=f"APTOS, {name} (n = {sel.sum():,})")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.92, fontsize=7.5, loc="upper right",
              markerscale=3)
    ax.set_xlabel("Principal component 1", fontsize=8.5, color=ink2)
    ax.set_ylabel("Principal component 2", fontsize=8.5, color=ink2)
    ax.set_title("APTOS photos by camera (photo size)", fontsize=10, color=ink, loc="left")
    ax.grid(color=grid_c, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(ink2)
    ax.tick_params(colors=ink2, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
