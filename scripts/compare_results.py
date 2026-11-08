"""Compare a re-run (default results_rerun/) with the committed reference outputs in results/.

    python scripts/compare_results.py                 # results/ vs results_rerun/
    python scripts/compare_results.py results other/

For every CSV present in both folders it prints the largest absolute difference over the numeric columns
(and flags differing non-numeric cells). Exit code 0 always; read the table.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ref = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
new = Path(sys.argv[2] if len(sys.argv) > 2 else "results_rerun")
rows = []
for f in sorted(ref.rglob("*.csv*")):
    g = new / f.relative_to(ref)
    if not g.exists():
        rows.append((str(f.relative_to(ref)), "missing in re-run", ""))
        continue
    a, b = pd.read_csv(f), pd.read_csv(g)
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        rows.append((str(f.relative_to(ref)), f"shape {a.shape} vs {b.shape}", ""))
        continue
    num = a.select_dtypes("number").columns
    d = float(np.nanmax(np.abs(a[num].to_numpy(float) - b[num].to_numpy(float)))) if len(num) else 0.0
    other = [c for c in a.columns if c not in num]
    txt = int((a[other].astype(str) != b[other].astype(str)).to_numpy().sum()) if other else 0
    rows.append((str(f.relative_to(ref)), f"max |diff| = {d:.3g}", f"{txt} text cells differ" if txt else ""))
w = max(len(r[0]) for r in rows) if rows else 10
for r in rows:
    print(f"{r[0]:<{w}}  {r[1]}  {r[2]}")
