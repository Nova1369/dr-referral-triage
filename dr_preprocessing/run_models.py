#!/usr/bin/env python3
"""
run_models.py - E1 of the mid report: the baselines and the course models, 10-fold cross-validation on APTOS.

For every outer fold k (column `fold` of manifest.csv, made by dr_preprocess.py):

  test   APTOS fold k (so every APTOS image is a test image exactly once)
  train  the other nine folds, split by group (copies and same-eye photos stay together) into
           cal  15 %  kept aside for the conformal step (C1): predicted and saved, not used here
           fit  85 %  hyper-parameters tuned by 5-fold grouped cross-validation inside `fit`,
                      then the best setting is refitted on all of `fit`

The ResNet-50 features are standardised and reduced with PCA inside every training split, so no test
image ever influences the scaling or the PCA. The inner folds pick the setting with the best mean of
AUROC(referable) and AUROC(urgent). Probabilities are calibrated with one-vs-rest isotonic regression,
fitted on the inner out-of-fold predictions of the chosen setting; raw and calibrated probabilities are
both saved. Every fitted model also predicts IDRiD and Messidor-2 (role "external"), for E0 and E2.

Models (--models, comma-separated; default: all):
  b1    B1 most likely class (the class shares of the training split, the same for every image)
  b2    B2 image quality only: gradient boosting on the quality and image-size columns of manifest.csv
  b2a   B2 with the image-size columns only (photo width and height, retina radius)
  b2b   B2 with the quality columns only (brightness, contrast, colour, sharpness, ...)
  lr gnb lda qda knn cart rf ada gb svm mlp    the course models, on ResNet-50 features + PCA
B0 (refer all / refer none) needs no training; summarize_e1.py adds it.

Outputs (in --out, default ../results/e1):
  <model>/fold<k>.csv.gz        uid, role (test / cal / external), p0.. raw and c0.. calibrated probabilities
  <model>/fold<k>_tuning.csv    every setting tried, with its inner-CV AUROCs, time and warnings
  run_config.json               settings, grids and library versions (run_config-grade.json for --target grade)
With --target grade the models predict the five grades instead, and the folders are named <model>-grade.
Finished (model, fold) pairs are skipped when the script runs again; --time-budget stops it cleanly
(exit code 3), and the same command continues.

Usage:
  python run_models.py --data ../data_512                          # everything
  python run_models.py --data ../data_512 --models b1,b2,lr        # a subset
  python run_models.py --data ../data_512 --models lr --target grade
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dr_preprocess import assign_folds  # noqa: E402  (same grouped, grade-stratified greedy folds)

SIZE_COLS = ["orig_w", "orig_h", "fov_r"]                     # photo and retina size: camera / clinic
QUAL_COLS = ["q_valid_frac", "q_mean", "q_std", "q_r", "q_g", "q_b", "q_dark", "q_sat",
             "q_lapvar", "q_entropy"]                           # brightness, contrast, colour, sharpness
TABLE_FEATURES = {"quality": QUAL_COLS + SIZE_COLS, "size": SIZE_COLS, "qual": QUAL_COLS}
INNER_FOLDS = 5
CAL_FOLDS = (3, 20)             # 3 of 20 grouped, stratified folds = 15 % of the training split


def grid(**axes):
    keys = list(axes)
    return [dict(zip(keys, values)) for values in itertools.product(*(axes[k] for k in keys))]


# key: (name in tables, feature set, hyper-parameter grid)
MODELS = {
    "b1": ("B1 Most likely class", "none", [{}]),
    "b2": ("B2 Image quality only", "quality", grid(lr=[0.05, 0.1], leaves=[7, 15, 31, 63])),
    "b2a": ("B2a Image size only", "size", grid(lr=[0.05, 0.1], leaves=[7, 15, 31, 63])),
    "b2b": ("B2b Quality, without size", "qual", grid(lr=[0.05, 0.1], leaves=[7, 15, 31, 63])),
    "lr": ("Logistic regression", "cnn", grid(pca=[256, 512, 1024, 2048], C=[3e-4, 1e-3, 3e-3, 1e-2])),
    "gnb": ("Naive Bayes", "cnn", grid(pca=[8, 16, 32, 64, 128])),
    "lda": ("LDA", "cnn", grid(pca=[16, 32, 64, 128, 256, 512])),
    "qda": ("QDA", "cnn", grid(pca=[32, 64, 128], reg=[0.1, 0.3, 0.5, 0.7, 0.9])),
    "knn": ("k-NN", "cnn", grid(pca=[64, 128, 256], k=[25, 50, 100, 200])),
    "cart": ("Decision tree", "cnn", grid(pca=[16, 32, 64], depth=[5, 8, 12], leaf=[10, 30, 60, 100])),
    "rf": ("Random forest", "cnn", grid(pca=[32, 64, 128], max_features=[0.2, 0.4])),
    "ada": ("AdaBoost", "cnn", grid(pca=[64, 128], depth=[2, 3], lr=[0.5, 1.0])),
    "gb": ("Gradient boosting", "cnn", grid(pca=[128, 256], lr=[0.05, 0.1], leaves=[7, 15])),
    "svm": ("SVM (RBF)", "cnn", grid(pca=[256, 512], C=[2.0, 8.0, 32.0], gamma=[0.125, 0.25, 0.5])),
    "mlp": ("MLP", "cnn", grid(pca=[64, 128, 256], hidden=[(256,), (512, 128)], alpha=[1e-4, 1e-3, 1e-2])),
}


def make_model(key, p, seed, Xtr):
    if key == "b1":
        return DummyClassifier(strategy="prior")
    if key in ("b2", "b2a", "b2b", "gb"):
        return HistGradientBoostingClassifier(learning_rate=p["lr"], max_leaf_nodes=p["leaves"], max_iter=500,
                                              early_stopping=True, validation_fraction=0.1,
                                              n_iter_no_change=20, random_state=seed)
    if key == "lr":
        return LogisticRegression(C=p["C"], max_iter=5000)
    if key == "gnb":
        return GaussianNB()
    if key == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if key == "qda":
        return QuadraticDiscriminantAnalysis(reg_param=p["reg"])
    if key == "knn":
        return KNeighborsClassifier(n_neighbors=p["k"], weights="distance")
    if key == "cart":
        return DecisionTreeClassifier(max_depth=p["depth"], min_samples_leaf=p["leaf"], random_state=seed)
    if key == "rf":
        return RandomForestClassifier(n_estimators=200, max_features=p["max_features"], n_jobs=1,
                                      random_state=seed)
    if key == "ada":
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=p["depth"]), n_estimators=100,
                                  learning_rate=p["lr"], random_state=seed)
    if key == "svm":
        gamma = p["gamma"] / (Xtr.shape[1] * Xtr.var())     # multiples of scikit-learn's gamma="scale"
        try:      # libsvm's own probabilities: Platt scaling + pairwise coupling, 5-fold internal CV
            return SVC(C=p["C"], gamma=gamma, probability=True, random_state=seed)
        except TypeError:     # scikit-learn >= 1.11 removes probability=True; one-vs-rest Platt scaling instead
            return CalibratedClassifierCV(SVC(C=p["C"], gamma=gamma), method="sigmoid", cv=5, ensemble=False)
    if key == "mlp":
        return make_pipeline(StandardScaler(),               # inputs to the network on a common scale
                             MLPClassifier(hidden_layer_sizes=p["hidden"], alpha=p["alpha"],
                                           early_stopping=True, max_iter=500, random_state=seed))
    raise ValueError(key)


def fit_predict(key, params, seed, Xtr, ytr, X_eval, n_class):
    """Fit one model and return its class probabilities on each array in X_eval (one thread per job)."""
    t0 = time.time()
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message=".*probability.*deprecated", category=FutureWarning)
        n = params.get("pca")
        if n:
            Xtr, X_eval = Xtr[:, :n], [X[:, :n] for X in X_eval]
        model = make_model(key, params, seed, Xtr)
        model.fit(Xtr, ytr)
        out = []
        for X in X_eval:
            P = np.zeros((len(X), n_class))
            P[:, np.asarray(model.classes_, int)] = model.predict_proba(X)
            out.append(P)
    kinds = sorted({w.category.__name__ for w in caught})
    return out, time.time() - t0, kinds


def reduce(Xtr, X_eval, n_max):
    """Standardise and PCA, both fitted on the training rows only."""
    scaler = StandardScaler().fit(Xtr)
    pca = PCA(n_components=min(n_max, *Xtr.shape), svd_solver="full").fit(scaler.transform(Xtr))
    return pca.transform(scaler.transform(Xtr)), [pca.transform(scaler.transform(X)) for X in X_eval]


def isotonic_fit(P, y):
    return [IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(P[:, c], (y == c).astype(float))
            for c in range(P.shape[1])]


def isotonic_apply(cal, P):
    Q = np.column_stack([iso.predict(P[:, c]) for c, iso in enumerate(cal)])
    s = Q.sum(1, keepdims=True)
    return np.where(s > 0, Q / np.where(s > 0, s, 1), 1.0 / Q.shape[1])


def auroc_pair(y, P, ref_from, urg_from):
    return (roc_auc_score(y >= ref_from, P[:, ref_from:].sum(1)),
            roc_auc_score(y >= urg_from, P[:, urg_from:].sum(1)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="E1: baselines and course models, 10-fold CV on APTOS.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", default="../data_512", help="output folder of dr_preprocess.py")
    ap.add_argument("--features", default="features/resnet50_224.npz", help="relative to --data")
    ap.add_argument("--out", default="../results/e1")
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated keys, see the top of the file")
    ap.add_argument("--target", choices=["ref3", "grade"], default="ref3",
                    help="ref3 = our 3 classes; grade = the 5 DR grades (for E0)")
    ap.add_argument("--folds", default="all", help="outer folds to run, e.g. 0,1 (default: all)")
    ap.add_argument("--jobs", type=int, default=0, help="parallel jobs (0 = all CPU cores)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time-budget", type=float, default=0, help="stop after this many seconds (0 = no limit)")
    args = ap.parse_args(argv)

    t_start = time.time()
    data, out = Path(args.data), Path(args.out)
    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    bad = [k for k in keys if k not in MODELS]
    if bad:
        sys.exit(f"unknown model(s) {bad}; choose from {list(MODELS)}")
    jobs = args.jobs or os.cpu_count() or 1
    # PCA is computed once per split with the most components any chosen model needs; smaller settings
    # are the leading columns (PCA components are nested). 2048 = all of them, i.e. no reduction.
    pca_max = max([c["pca"] for k in keys for c in MODELS[k][2] if "pca" in c] or [1])

    man = pd.read_csv(data / "manifest.csv")
    with np.load(data / args.features) as z:
        X, uid = z["X"], z["uid"]
    if len(uid) != len(man) or not (uid == man["uid"].astype(str).to_numpy()).all():
        sys.exit("feature rows do not match manifest.csv; run extract_features.py again")
    tables = {name: man[cols].to_numpy(float) for name, cols in TABLE_FEATURES.items()}
    tables["none"] = np.zeros((len(man), 1))
    uids = man["uid"].astype(str).to_numpy()
    if args.target == "ref3":
        y, n_class, ref_from, urg_from, suffix = man["ref3"].to_numpy(int), 3, 1, 2, ""
    else:
        y, n_class, ref_from, urg_from, suffix = man["grade"].to_numpy(int), 5, 2, 3, "-grade"

    aptos = (man["dataset"] == "aptos").to_numpy() & (man["fold"] >= 0).to_numpy()
    ext_idx = np.flatnonzero(man["dataset"].isin(["idrid", "messidor2"]).to_numpy())
    folds = sorted(man.loc[aptos, "fold"].unique()) if args.folds == "all" else \
        [int(f) for f in args.folds.split(",")]

    out.mkdir(parents=True, exist_ok=True)
    cfg_file = out / f"run_config{suffix}.json"
    models_cfg = json.loads(cfg_file.read_text()).get("models", {}) if cfg_file.exists() else {}
    models_cfg.update({k: {"name": MODELS[k][0], "features": MODELS[k][1], "grid": MODELS[k][2]} for k in keys})
    cfg_file.write_text(json.dumps({
        "models": models_cfg,
        "target": args.target, "features_file": args.features, "seed": args.seed, "outer_folds": folds,
        "inner_folds": INNER_FOLDS, "calibration_share": CAL_FOLDS[0] / CAL_FOLDS[1],
        "selection": "mean of AUROC(referable) and AUROC(urgent) on the pooled inner out-of-fold predictions",
        "calibration": "one-vs-rest isotonic regression on the inner out-of-fold predictions",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__},
        "started": time.strftime("%Y-%m-%d %H:%M")}, indent=2, default=str))

    print(f"{len(keys)} model(s) x {len(folds)} fold(s), target {args.target}, {jobs} parallel jobs")
    for k in folds:
        todo = [m for m in keys if not (out / f"{m}{suffix}" / f"fold{k}.csv.gz").exists()]
        if not todo:
            continue
        if args.time_budget and time.time() - t_start > args.time_budget:
            print("\nTime budget reached. Run the same command again to continue.")
            sys.exit(3)

        # ---- splits for this outer fold ----
        test_idx = np.flatnonzero(aptos & (man["fold"].to_numpy() == k))
        train = man[aptos & (man["fold"].to_numpy() != k)]
        f20, _ = assign_folds(train, CAL_FOLDS[1], args.seed + 2000 + k)
        is_cal = f20.isin(range(CAL_FOLDS[0])).to_numpy()
        cal_idx = man.index.get_indexer(train.index[is_cal])
        fit_df = train[~is_cal]
        fit_idx = man.index.get_indexer(fit_df.index)
        inner = assign_folds(fit_df, INNER_FOLDS, args.seed + 1000 + k)[0].to_numpy()
        y_fit = y[fit_idx]

        # ---- standardise + PCA once per split; every CNN model reuses them ----
        t0 = time.time()
        feats = {}
        if any(MODELS[m][1] == "cnn" for m in todo):
            inner_sets = []
            for j in range(INNER_FOLDS):
                tr, va = fit_idx[inner != j], fit_idx[inner == j]
                Ztr, (Zva,) = reduce(X[tr], [X[va]], pca_max)
                inner_sets.append((Ztr, Zva))
            Zfit, Zeval = reduce(X[fit_idx], [X[cal_idx], X[test_idx], X[ext_idx]], pca_max)
            feats["cnn"] = (inner_sets, Zfit, Zeval)
        for name, A in tables.items():
            inner_sets = [(A[fit_idx[inner != j]], A[fit_idx[inner == j]]) for j in range(INNER_FOLDS)]
            feats[name] = (inner_sets, A[fit_idx], [A[cal_idx], A[test_idx], A[ext_idx]])
        print(f"\nfold {k}: fit {len(fit_idx)}, cal {len(cal_idx)}, test {len(test_idx)} "
              f"(PCA {time.time() - t0:.0f} s)", flush=True)

        for m in todo:
            t1 = time.time()
            name, fset, configs = MODELS[m]
            inner_sets, Ffit, Feval = feats[fset]
            seed = args.seed + k
            tasks = [(c, j) for c in range(len(configs)) for j in range(INNER_FOLDS)]
            res = Parallel(n_jobs=min(jobs, len(tasks)))(
                delayed(fit_predict)(m, configs[c], seed, inner_sets[j][0], y_fit[inner != j],
                                     [inner_sets[j][1]], n_class) for c, j in tasks)
            oof = np.zeros((len(configs), len(fit_idx), n_class))
            secs = np.zeros(len(configs))
            warn = [set() for _ in configs]
            for (c, j), (P, s, w) in zip(tasks, res):
                oof[c, inner == j] = P[0]
                secs[c] += s
                warn[c].update(w)
            scores = np.array([auroc_pair(y_fit, oof[c], ref_from, urg_from) for c in range(len(configs))])
            best = int(np.argmax(scores.mean(1)))            # first of equals, in grid order
            cal = isotonic_fit(oof[best], y_fit)
            (P_cal, P_test, P_ext), s_fit, w_fit = fit_predict(m, configs[best], seed, Ffit, y_fit, Feval, n_class)

            # ---- save ----
            d = out / f"{m}{suffix}"
            d.mkdir(parents=True, exist_ok=True)
            parts = []
            for role, idx, P in (("test", test_idx, P_test), ("cal", cal_idx, P_cal), ("external", ext_idx, P_ext)):
                df = pd.DataFrame({"uid": uids[idx], "role": role})
                C = isotonic_apply(cal, P)
                for c in range(n_class):
                    df[f"p{c}"] = P[:, c]
                for c in range(n_class):
                    df[f"c{c}"] = C[:, c]
                parts.append(df)
            pd.DataFrame({
                "config": [json.dumps(c) for c in configs], "auroc_referable": scores[:, 0],
                "auroc_urgent": scores[:, 1], "score": scores.mean(1), "best": np.arange(len(configs)) == best,
                "seconds": secs.round(1), "warnings": [" ".join(sorted(w)) for w in warn]}).to_csv(
                d / f"fold{k}_tuning.csv", index=False)
            tmp = d / f"fold{k}.tmp.csv.gz"
            pd.concat(parts).to_csv(tmp, index=False, float_format="%.6f", compression="gzip")
            os.replace(tmp, d / f"fold{k}.csv.gz")           # written last: marks the pair as done
            a_ref, a_urg = auroc_pair(y[test_idx], P_test, ref_from, urg_from)
            print(f"  {m:5s} best {json.dumps(configs[best]):48s} inner {scores[best].mean():.3f} | "
                  f"test AUROC ref {a_ref:.3f} urg {a_urg:.3f} | {time.time() - t1:5.0f} s"
                  + (f" | warnings: {' '.join(w_fit)}" if w_fit else ""), flush=True)

    missing = [(m, k) for m in keys for k in folds if not (out / f"{m}{suffix}" / f"fold{k}.csv.gz").exists()]
    print("\nAll done." if not missing else f"\n{len(missing)} (model, fold) pairs still missing.",
          f"({(time.time() - t_start) / 60:.1f} min this run)")


if __name__ == "__main__":
    main()
