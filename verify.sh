#!/usr/bin/env bash
set -euo pipefail

# Unified verification entry point for the OSATS regression model.
#
# Usage:
#   bash verify.sh [--task TASK] [--T N] [--method exact|approx] [--verifier n2v|abcrown] [--no-search]
#
# Options:
#   --task TASK     Suturing, Knot_Tying, or Needle_Passing (default: Suturing)
#   --T N           window length in timesteps (default: 10)
#   --method M      exact or approx — controls n2v star-set mode (default: exact)
#                   ignored when --verifier abcrown (abcrown manages its own strategy)
#   --verifier V    n2v or abcrown (default: n2v)
#   --no-search     only emit the four property verdicts at the physical epsilon;
#                   skip the (expensive) certified-epsilon binary search.
#                   ~4 verifier calls/fold instead of ~28.
#
# Prerequisites:
#   - LOSO fold checkpoints in models/ (run: python3 train.py --regression [--task TASK])
#   - For --verifier abcrown: run setup_abcrown.sh first
#
# Results land in results/regression_results[_{task}][_T{N}][_approx][_abcrown][_verdicts].txt
# Suturing uses no task suffix (backward compatible). Other tasks use a compact slug.
# No combination of settings ever overwrites another run's output.

TASK=Suturing
T=10
METHOD=exact
VERIFIER=n2v
NO_SEARCH=0
FOLDS="1 2 3 4 5"

usage() {
    grep '^#' "$0" | grep -v '#!/' | sed 's/^# \?//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --task)      TASK="$2";     shift 2 ;;
        --T)         T="$2";        shift 2 ;;
        --method)    METHOD="$2";   shift 2 ;;
        --verifier)  VERIFIER="$2"; shift 2 ;;
        --no-search) NO_SEARCH=1;   shift ;;
        -h|--help)   usage ;;
        *) echo "ERROR: unknown option '$1'"; usage ;;
    esac
done

# --- Validate ---
if [[ "$TASK" != "Suturing" && "$TASK" != "Knot_Tying" && "$TASK" != "Needle_Passing" ]]; then
    echo "ERROR: --task must be Suturing, Knot_Tying, or Needle_Passing"; exit 1
fi
if [[ "$METHOD" != "exact" && "$METHOD" != "approx" ]]; then
    echo "ERROR: --method must be 'exact' or 'approx'"; exit 1
fi
if [[ "$VERIFIER" != "n2v" && "$VERIFIER" != "abcrown" ]]; then
    echo "ERROR: --verifier must be 'n2v' or 'abcrown'"; exit 1
fi
if [[ "$VERIFIER" == "abcrown" && "$METHOD" != "exact" ]]; then
    echo "Note: --method ignored for alpha-beta-CROWN (it manages its own strategy)"
fi
if [[ "$VERIFIER" == "abcrown" && ! -f "envs/abcrown/bin/python" ]]; then
    echo "ERROR: alpha-beta-CROWN venv not found. Run:  bash setup_abcrown.sh"
    exit 1
fi

# Derive task slug (empty for Suturing for backward compat)
TASK_SLUG=$(python3 -c "import config; print(config.task_slug('${TASK}'))")
CKPT_INFIX=$([ -n "$TASK_SLUG" ] && echo "_${TASK_SLUG}" || echo "")

echo "=== Settings: task=${TASK}  T=${T}  method=${METHOD}  verifier=${VERIFIER} ==="
echo ""

# --- Step 0: check fold checkpoints ---
echo "=== Step 0: Check LOSO fold checkpoints ==="
missing=0
for i in $FOLDS; do
    ckpt="models/best_model_regression${CKPT_INFIX}_fold${i}.pth"
    if [ ! -f "$ckpt" ]; then
        echo "  MISSING: $ckpt"
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    echo "ERROR: train first:  python3 train.py --regression --task ${TASK}"
    exit 1
fi
echo "  all 5 fold checkpoints present"
echo ""

# --- Step 1: export each fold to ONNX at T ---
echo "=== Step 1: Export each fold to ONNX (task=${TASK}  T=${T}) ==="
for i in $FOLDS; do
    python3 export_onnx_regression.py "$i" "$T" "$TASK"
done
echo ""

# --- Steps 2-4: generate properties, verify, binary-search epsilon ---
PROP_ARGS="allfolds --task $TASK --T $T"
if [[ "$VERIFIER" == "n2v" && "$METHOD" == "approx" ]]; then
    PROP_ARGS="$PROP_ARGS --approx"
fi
if [[ "$VERIFIER" == "abcrown" ]]; then
    PROP_ARGS="$PROP_ARGS --abcrown"
fi
if [[ "$NO_SEARCH" == "1" ]]; then
    PROP_ARGS="$PROP_ARGS --no-search"
fi

echo "=== Steps 2-4: Per-fold thresholds, verdicts, epsilon search ==="
echo ""
# shellcheck disable=SC2086
python3 generate_property_regression.py $PROP_ARGS

echo ""
echo "Done."
