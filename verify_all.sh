#!/usr/bin/env bash
set -euo pipefail

# Per-fold, paper-faithful verification pipeline.
# The OSATS regressor is trained with leave-one-super-trial-out (LOSO) CV
# (python3 train.py --regression), producing 5 fold checkpoints. We export and
# formally verify each fold, anchoring every property on a trial that fold held
# out, then report certified epsilon per fold.

FOLDS="1 2 3 4 5"

echo "=== Step 0: Check LOSO fold checkpoints ==="
missing=0
for i in $FOLDS; do
    if [ ! -f "best_model_regression_fold${i}.pth" ]; then
        echo "  MISSING: best_model_regression_fold${i}.pth"
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    echo "ERROR: train the LOSO fold models first:  python3 train.py --regression"
    exit 1
fi
echo "  all 5 fold checkpoints present"

echo ""
echo "=== Step 1: Export each fold to ONNX (with equivalence + op-support check) ==="
for i in $FOLDS; do
    python3 export_onnx_regression.py "$i"
done

echo ""
echo "=== Steps 2-4: Per-fold thresholds, fixed-epsilon verdicts, certified-epsilon search ==="
echo "(generates property_*_fold{i}.vnnlib, verifies all four, binary-searches noise + range)"
echo ""
python3 generate_property_regression.py allfolds

echo ""
echo "Results written to regression_results.txt"
echo "Done."
