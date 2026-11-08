# Reference results

The outputs of `run_all.sh` as used in the report: summary tables (`*/summary.md`), aggregate CSVs and
figures. A re-run writes to `results_rerun/` by default; compare with `python scripts/compare_results.py`.

Left out on purpose:

* **Per-image outputs** (out-of-fold predictions `e1/*/fold*.csv.gz`, `c1/decisions_lr.csv.gz`,
  `group_robust/oof_*.csv.gz`, `e2/aptos_model.csv.gz`, the list of missed urgent eyes, and the per-photo
  files and example photos of `evidence_maps/`). They identify individual licensed photos; the scripts
  regenerate them locally.
* **Large draw-level files** over 1.5 MB (`e2/draws.csv`, `label_efficient/draws.csv.gz`,
  `c1/per_fold.csv`, `c3/workload_aptos.csv`, `evidence_maps/heads.npz`). Their summaries are here.

| Folder | Experiment (report section) |
|---|---|
| `e1/` | E1, the 11 course models and baselines B0-B2 under nested 10-fold CV (4.1) |
| `e0/` | E0, standard conformal prediction at new clinics (4.1) |
| `kmeans/` | k-means check: do photos group by grade, or by clinic and camera? |
| `c1/` | C1 and baselines B3-B5, C1-q, C1-g (4.2) |
| `e2/` | C2/E2 at IDRiD and Messidor-2, B6, B7, C2-PAC (4.3) |
| `c3/` | C3/E3, prevalence and workload, B8, B9, BBSE (4.4) |
| `errors/` | error analysis and subgroups (4.5) |
| `ablations/` | 512 px, Graham filter, PCA size, blur/dark/JPEG (4.5) |
| `camera_shortcut/`, `camera_fixes/` | additional work: the APTOS camera shortcut and fixes tuned under the E1 protocol (5) |
| `group_robust/`, `group_frontier/` | additional work: the 11-method comparison and the baseline sweeps (5) |
| `evidence_maps/` | additional work: exact evidence maps, deletion and mirror tests (5) |
| `mirror/` | additional work: mirror-averaged features (5) |
| `label_efficient/` | additional work: C2-IW (5) |
