#!/usr/bin/env bash
set -euo pipefail

# Approx-only verification pipeline at T=100.
# Identical to verify_all.sh except:
#   - ONNX models are exported with T=100 (suffix _T20), stored alongside T=10 files
#   - The verifier runs approx-star only (--approx --category collins_rul_cnn_2022),
#     eliminating exact-star escalation and the associated memory blowup
#   - Properties and results land in new files (*_T20_approx.*) — nothing is overwritten
#
# Run verify_all.sh first to produce the baseline T=10 results before running this.

T=100
FOLDS="1 2 3 4 5"

echo "=== Step 0: Check LOSO fold checkpoints ==="
missing=0
for i in $FOLDS; do
    if [ ! -f "models/best_model_regression_fold${i}.pth" ]; then
        echo "  MISSING: models/best_model_regression_fold${i}.pth"
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    echo "ERROR: train the LOSO fold models first:  python3 train.py --regression"
    exit 1
fi
echo "  all 5 fold checkpoints present"

echo ""
echo "=== Step 1: Export each fold to ONNX at T=${T} ==="
for i in $FOLDS; do
    python3 export_onnx_regression.py "$i" "$T"
done

echo ""
echo "=== Steps 2-4: Per-fold thresholds, verdicts, certified-epsilon search (approx-only, T=${T}) ==="
echo "(properties -> properties/property_*_fold{i}_T${T}_approx.vnnlib)"
echo "(results    -> results/regression_results_T${T}_approx.txt)"
echo ""
python3 generate_property_regression.py allfolds --approx --T "$T"

echo ""
echo "Results written to results/regression_results_T${T}_approx.txt"
echo "Done."
