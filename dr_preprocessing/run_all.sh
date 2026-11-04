#!/usr/bin/env bash
# run_all.sh - every step of the project, in order (macOS / Linux; on Windows run the same lines one by one).
#
# Run it from the folder that holds the scripts (dr_preprocessing/). Edit the paths below to where your
# downloads are. Every step can be stopped and run again: finished work is cached and skipped.
#
#   bash run_all.sh            # everything
#   bash run_all.sh 3          # start at step 3 (steps 1-2 already done)
set -euo pipefail

PY="${PY:-python3}"
APTOS="../aptos2019-blindness-detection"           # Kaggle download: train.csv + train_images/
IDRID="../IRIDIS"                                   # IDRiD "B. Disease Grading" (any nesting)
MESSIDOR="../Messidor - 2"                          # ADCIS download: IMAGES/ (+ messidor-2.csv exam list)
MESSIDOR_LABELS="../Messidor - 2/messidor_data.csv" # Krause et al. grades (Kaggle)
WEIGHTS="../models/resnet50_ram-a26f946b.pth"       # timm ImageNet ResNet-50 weights (README section 8)
DATA="../data_512"
RESULTS="../results"
START="${1:-1}"

step() { echo; echo "=== Step $1: $2 ==="; }

if [ "$START" -le 1 ]; then
  step 1 "preprocessing: crop, quality features, duplicates, folds -> $DATA"
  "$PY" dr_preprocess.py --aptos "$APTOS" --idrid "$IDRID" --messidor "$MESSIDOR" \
        --messidor-labels "$MESSIDOR_LABELS" --out "$DATA"
fi
if [ "$START" -le 2 ]; then
  step 2 "ResNet-50 features -> $DATA/features/resnet50_224.npz"
  "$PY" extract_features.py --data "$DATA" --weights "$WEIGHTS"
fi
if [ "$START" -le 3 ]; then
  step 3 "E1: baselines and course models (10-fold nested CV) -> $RESULTS/e1"
  "$PY" run_models.py --data "$DATA" --out "$RESULTS/e1"
  "$PY" run_models.py --data "$DATA" --out "$RESULTS/e1" --models lr --target grade
  "$PY" summarize_e1.py --data "$DATA" --results "$RESULTS/e1"
fi
if [ "$START" -le 4 ]; then
  step 4 "E0: standard conformal prediction, APTOS -> IDRiD, Messidor-2 -> $RESULTS/e0"
  "$PY" e0_conformal.py --data "$DATA" --results "$RESULTS/e1" --out "$RESULTS/e0" --models lr-grade,lr,svm
fi
if [ "$START" -le 5 ]; then
  step 5 "k-means check -> $RESULTS/kmeans"
  "$PY" kmeans_check.py --data "$DATA" --out "$RESULTS/kmeans"
fi
if [ "$START" -le 6 ]; then
  step 6 "C1: referral rule with one conformal threshold per class, against B3, B4, B5 -> $RESULTS/c1"
  "$PY" c1_referral.py --data "$DATA" --results "$RESULTS/e1" --out "$RESULTS/c1"
fi
if [ "$START" -le 7 ]; then
  step 7 "C2 and E2: thresholds refitted at IDRiD and Messidor-2, against B6 and B7 -> $RESULTS/e2"
  "$PY" e2_new_clinics.py --data "$DATA" --out "$RESULTS/e2"
fi
if [ "$START" -le 8 ]; then
  step 8 "C3 and E3: prevalence and workload, against B8 and B9 -> $RESULTS/c3"
  "$PY" c3_prevalence.py --data "$DATA" --results "$RESULTS" --out "$RESULTS/c3"
fi
if [ "$START" -le 9 ]; then
  step 9 "error analysis and subgroups -> $RESULTS/errors"
  "$PY" error_analysis.py --data "$DATA" --results "$RESULTS" --out "$RESULTS/errors"
fi
if [ "$START" -le 10 ]; then
  step 10 "ablations and robustness -> $RESULTS/ablations"
  # extra features (need the images and PyTorch); skipped when the file is already there
  [ -f "$DATA/features/resnet50_512.npz" ] || "$PY" extract_features.py --data "$DATA" --weights "$WEIGHTS" --size 512
  for v in graham blur dark jpeg; do
    [ -f "$DATA/features/resnet50_224_$v.npz" ] || "$PY" extract_variants.py --data "$DATA" --weights "$WEIGHTS" --variant "$v"
  done
  "$PY" ablations.py --data "$DATA" --results "$RESULTS" --out "$RESULTS/ablations"
fi
if [ "$START" -le 11 ]; then
  step extra "additional work: camera shortcut and fixes, evidence maps, mirror, baselines, C2-IW -> $RESULTS"
  "$PY" camera_shortcut.py --data "$DATA" --results "$RESULTS/e1" --out "$RESULTS/camera_shortcut"
  "$PY" camera_fixes.py --data "$DATA" --results "$RESULTS" --out "$RESULTS/camera_fixes"
  # evidence maps (needs the images and PyTorch)
  for st in heads maps deletion flip summary examples; do
    "$PY" evidence_maps.py "$st" --data "$DATA" --results "$RESULTS" --weights "$WEIGHTS" --out "$RESULTS/evidence_maps"
  done
  # mirror-averaged features (the mirrored photos' features need the images and PyTorch)
  [ -f "$DATA/features/resnet50_224_hflip.npz" ] || "$PY" extract_variants.py --data "$DATA" --weights "$WEIGHTS" --variant hflip
  "$PY" mirror_fix.py --data "$DATA" --results "$RESULTS" --out "$RESULTS/mirror"
  # the camera-group guarantee against strong baselines (needs camera_fixes above)
  "$PY" group_robust.py --data "$DATA" --results "$RESULTS" --out "$RESULTS/group_robust"
  # each baseline's own setting swept, compared at matched fairness
  "$PY" group_frontier.py --data "$DATA" --results "$RESULTS" --out "$RESULTS/group_frontier"
  # C2-IW, label-efficient recalibration at a new clinic (needs e2 above)
  "$PY" label_efficient.py --data "$DATA" --results "$RESULTS" --out "$RESULTS/label_efficient"
fi
echo; echo "All steps finished. Tables: $RESULTS/*/summary.md"
