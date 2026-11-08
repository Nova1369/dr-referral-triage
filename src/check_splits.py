"""Leakage check: rebuild every E1 split from the saved fold files and confirm that no duplicate group
crosses the training, calibration and test sets, that each test set is exactly its fold, that every APTOS
photo is tested once, and that no group spans two datasets.

    python src/check_splits.py --data data_512 --results results_rerun/e1 [--model lr]

Needs the per-image fold files (<results>/<model>/fold<k>.csv.gz), which are not committed (see results/README.md).
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_512")
    ap.add_argument("--results", default="results_rerun/e1")
    ap.add_argument("--model", default="lr")
    a = ap.parse_args()
    man = pd.read_csv(Path(a.data) / "manifest.csv")
    grp = dict(zip(man["uid"], man["dataset"] + "/" + man["group_id"].astype(str)))
    apt = man[man["dataset"] == "aptos"]
    problems, tested = [], []
    if man.groupby("group_id")["dataset"].nunique().max() > 1:
        problems.append("a group spans two datasets")
    if apt.groupby("group_id")["fold"].nunique().max() > 1:
        problems.append("an APTOS group spans two folds")
    for k in range(10):
        f = pd.read_csv(Path(a.results) / a.model / f"fold{k}.csv.gz")
        test, cal = set(f.loc[f.role == "test", "uid"]), set(f.loc[f.role == "cal", "uid"])
        train = set(apt["uid"]) - test - cal
        if test != set(apt.loc[apt["fold"] == k, "uid"]):
            problems.append(f"fold {k}: test set is not fold {k}")
        G = {n: {grp[u] for u in s} for n, s in (("train", train), ("cal", cal), ("test", test))}
        for x, y in (("train", "cal"), ("train", "test"), ("cal", "test")):
            if G[x] & G[y]:
                problems.append(f"fold {k}: {len(G[x] & G[y])} groups shared by {x} and {y}")
        tested += list(test)
        print(f"fold {k}: train {len(train)}, cal {len(cal)}, test {len(test)} photos; no shared groups"
              if not any(p.startswith(f"fold {k}:") for p in problems) else f"fold {k}: PROBLEM")
    if sorted(tested) != sorted(apt["uid"]):
        problems.append("not every APTOS photo is tested exactly once")
    print("OK: no leakage found" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))


if __name__ == "__main__":
    main()
