#!/usr/bin/env bash
# run_all.sh - every step of the mid-term work, in order (macOS / Linux; on Windows run the same lines one by one).
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
  step extra "camera shortcut study (additional work) -> $RESULTS/camera_shortcut"
  "$PY" camera_shortcut.py --data "$DATA" --results "$RESULTS/e1" --out "$RESULTS/camera_shortcut"
fi
echo; echo "All steps finished. Tables: $RESULTS/*/summary.md"
