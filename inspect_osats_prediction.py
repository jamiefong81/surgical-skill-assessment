"""Diagnostic: print predicted vs ground-truth OSATS for every trial.

Reports the empirical thresholds (v_expert, v_novice, C, L, delta) that
compute_thresholds in generate_property_regression uses, so they can be
verified before running formal certification.

Usage:
    python3 inspect_osats_prediction.py
"""

import torch

from dataset import load_osats_data
from model_flat_regression import build_trainable_regression, build_flat_regression
from generate_property_regression import (
    predict_overall, MARGIN_CEIL, MARGIN_FLOOR, DELTA, EXPERT_ANCHOR, NOVICE_ANCHOR
)

T             = 10
PTH_PATH      = "best_model_regression.pth"
TARGET_OUTPUT = 4


def main():
    """Load the trained regression model and print per-trial predictions and thresholds.

    Loads best_model_regression.pth, builds the flat model at T=10, then for
    every trial prints the ground-truth and predicted Overall Performance.
    Follows with a summary of expert prediction statistics and the derived
    thresholds C, L, and delta that match generate_property_regression exactly.
    """
    trained = build_trainable_regression()
    trained.load_state_dict(torch.load(PTH_PATH, weights_only=True))
    trained.eval()
    flat = build_flat_regression(trained, T)

    trials = load_osats_data('data')

    print(f"{'Trial':<22} {'Subj':<6} {'GT_Overall':>12} {'Pred_Overall':>14}")
    print('-' * 58)

    expert_preds = []
    v_expert = None
    v_novice = None

    for data, osats, subject, trial_num in trials:
        name = f"Suturing_{subject}{trial_num:03d}"
        x_star = data[:T].T.flatten()
        pred = predict_overall(flat, x_star, T)

        gt_overall   = osats[TARGET_OUTPUT]
        pred_overall = pred[TARGET_OUTPUT]
        print(f"{name:<22} {subject:<6} {gt_overall:>12.4f} {pred_overall:>14.4f}")

        if subject in ('D', 'E'):
            expert_preds.append(pred_overall)
        if name == EXPERT_ANCHOR:
            v_expert = pred_overall
        if name == NOVICE_ANCHOR:
            v_novice = pred_overall

    min_expert  = min(expert_preds)
    mean_expert = sum(expert_preds) / len(expert_preds)
    max_expert  = max(expert_preds)
    C = v_novice + MARGIN_CEIL
    L = min_expert - MARGIN_FLOOR

    print()
    print("Expert Overall Performance (predicted):")
    print(f"  min  = {min_expert:.6f}")
    print(f"  mean = {mean_expert:.6f}")
    print(f"  max  = {max_expert:.6f}")
    print()
    print("Derived thresholds:")
    print(f"  v_expert  = {v_expert:.6f}  (D001 predicted Overall)")
    print(f"  v_novice  = {v_novice:.6f}  (B001 predicted Overall)")
    print(f"  delta     = {DELTA:.6f}")
    print(f"  C         = {C:.6f}  (v_novice + {MARGIN_CEIL})")
    print(f"  L         = {L:.6f}  (min_expert - {MARGIN_FLOOR})")
    print(f"  C < L     = {C < L}  (monotonicity holds: {C < L})")


if __name__ == '__main__':
    main()
