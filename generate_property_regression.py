"""Generate VNN-LIB property files for the regression OSATS model and run epsilon searches.

Usage:
    python3 generate_property_regression.py generate   # write the four .vnnlib files
    python3 generate_property_regression.py search     # binary-search certified eps for noise + range
"""

import os
import signal
import subprocess
import sys

import numpy as np
import torch

from dataset import load_osats_data

T                = 10       # window length; must match export_onnx_regression.py
N_INPUTS         = 760      # 76 channels * T
N_OUTPUTS        = 6
TARGET_OUTPUT    = 4        # index of Overall Performance in the OSATS vector
EPSILON_PHYSICAL = 0.001    # physically motivated da Vinci encoder noise bound
EPSILON_BOUNDARY = 0.01     # larger perturbation for segment-boundary timesteps
DELTA            = 0.5      # allowed Y_4 deviation (half an OSATS rater point)
MARGIN_FLOOR     = 0.25     # subtracted from min expert prediction to set floor L
MARGIN_CEIL      = 0.5      # added to novice prediction to set ceiling C
BOUNDARY_TIMESTEPS = (0, 1, 8, 9)
EXPERT_SUBJECTS  = ('D', 'E')   # JIGSAWS self-proclaimed experts (>100h)

# --- Fold configuration (mutated by configure_fold) ---
# Defaults target the legacy single all-data model so `generate`/`search` and
# inspect_osats_prediction.py keep working unchanged. The per-fold pipeline
# (`allfolds`) overwrites these via configure_fold so every anchor, threshold
# and certificate refers to a trial the fold model was NOT trained on.
EXPERT_ANCHOR    = "Suturing_D001"
NOVICE_ANCHOR    = "Suturing_B001"
ONNX_PATH        = "surgical_fcn_regression.onnx"
CHECKPOINT       = "best_model_regression.pth"
HELDOUT_TRIAL    = None     # if set, the expert-floor set is restricted to held-out experts
PROP_SUFFIX      = ""       # filename suffix for the generated .vnnlib files
SEARCH_LO        = 0.0
SEARCH_HI        = 0.1      # 100x physical bound. Wide enough to resolve the range
                           # radius (which pinned at the old 0.02 ceiling) without
                           # reaching the eps where exact-star reliably explodes;
                           # the cgroup memory guard in run_verifier makes the larger
                           # probes safe (a blowup -> clean 'unknown' -> search narrows).
SEARCH_ITERS     = 12       # resolves eps to ~1/4096 of the bracket
RUN_INSTANCE     = "n2v/examples/VNN-COMP/run_instance.py"


def configure_fold(fold):
    """Point the module at one LOSO fold: its checkpoint, ONNX, and held-out anchors.

    Fold i holds out super-trial i (all trials with trial_num == i), so the
    held-out expert/novice anchors are Suturing_D{i:03d} and Suturing_B{i:03d} —
    trials the fold model never saw during training. Setting HELDOUT_TRIAL also
    restricts the expert-floor set to that fold's held-out experts. As a result
    v_expert, v_novice, C and L are all computed on unseen inputs, so each
    certificate is a statement about the fold model's generalization, not its fit.

    Args:
        fold: super-trial index 1..5.
    """
    global EXPERT_ANCHOR, NOVICE_ANCHOR, ONNX_PATH, CHECKPOINT, HELDOUT_TRIAL, PROP_SUFFIX
    EXPERT_ANCHOR = f"Suturing_D{fold:03d}"
    NOVICE_ANCHOR = f"Suturing_B{fold:03d}"
    ONNX_PATH     = f"surgical_fcn_regression_fold{fold}.onnx"
    CHECKPOINT    = f"best_model_regression_fold{fold}.pth"
    HELDOUT_TRIAL = fold
    PROP_SUFFIX   = f"_fold{fold}"


def load_anchor_window(anchor_name, T):
    """Load the first T timesteps of a named trial as a flat channel-first vector.

    Args:
        anchor_name: trial name string, e.g. 'Suturing_D001'.
        T:           number of timesteps to slice from the start of the trial.

    Returns:
        Flat float32 numpy array of length 76*T, ordered channel-first
        (X_{c*T + t} = channel c at timestep t), matching the VNN-LIB variable order.

    Raises:
        ValueError: if anchor_name is not found in the dataset.
    """
    trials = load_osats_data('data')
    for data, osats, subject, trial_num in trials:
        name = f"Suturing_{subject}{trial_num:03d}"
        if name == anchor_name:
            return data[:T].T.flatten()
    raise ValueError(f"Anchor trial {anchor_name!r} not found in dataset")


def predict_overall(flat_model, x_star, T):
    """Run the flat regression model on a pre-flattened input window.

    Uses the flat model (not SurgicalFCN) so predictions match exactly what
    n2v evaluates on the exported ONNX.

    Args:
        flat_model: trained SurgicalFCNFlat with 6 outputs.
        x_star:     flat float array of length 76*T (channel-first).
        T:          window length used to reshape x_star back to (1, 76, T).

    Returns:
        Numpy array of shape (6,) — the raw predicted OSATS scores.
    """
    x = torch.from_numpy(x_star.reshape(1, 76, T).astype(np.float32))
    with torch.no_grad():
        return flat_model(x).squeeze(0).numpy()


def write_bounds(lines, x_star, eps_per_index):
    """Append per-input box constraints to a VNN-LIB lines list.

    For each index i, appends:
        (assert (>= X_i  x_star[i] - eps_per_index[i]))
        (assert (<= X_i  x_star[i] + eps_per_index[i]))

    Args:
        lines:         list of VNN-LIB strings to append to (mutated in place).
        x_star:        flat anchor vector of length N_INPUTS.
        eps_per_index: per-index epsilon values, same length as x_star.
    """
    for i in range(len(x_star)):
        lines.append(f"(assert (>= X_{i} {x_star[i] - eps_per_index[i]:.8f}))")
        lines.append(f"(assert (<= X_{i} {x_star[i] + eps_per_index[i]:.8f}))")


def build_property(
    prop_name, anchor_name, flat_model, T, eps, boundary_eps,
    output_assertion_kind, threshold
):
    """Build the full VNN-LIB text for one property.

    Declares X_0..X_{N_INPUTS-1} and Y_0..Y_{N_OUTPUTS-1}, emits input box
    constraints, then appends the output violation assertion.  UNSAT means the
    violation is impossible in the region, i.e. the desired property is certified.

    Args:
        prop_name:             one of 'noise', 'monotonicity', 'segmentation', 'range'.
        anchor_name:           trial name whose first-T window centers the input box.
        flat_model:            trained SurgicalFCNFlat with 6 outputs.
        T:                     window length.
        eps:                   uniform input perturbation for non-boundary inputs.
        boundary_eps:          perturbation for boundary timesteps (segmentation only;
                               ignored for other properties).
        output_assertion_kind: 'two_sided', 'ceiling', or 'floor'.
        threshold:             precomputed literal(s); (v, delta) tuple for two_sided,
                               float for ceiling/floor.

    Returns:
        Full VNN-LIB property as a single newline-joined string.
    """
    x_star = load_anchor_window(anchor_name, T)

    lines = []
    for i in range(N_INPUTS):
        lines.append(f"(declare-const X_{i} Real)")
    for i in range(N_OUTPUTS):
        lines.append(f"(declare-const Y_{i} Real)")

    if prop_name == 'segmentation':
        eps_per_index = [boundary_eps if (i % T) in BOUNDARY_TIMESTEPS else eps for i in range(N_INPUTS)]
    else:
        eps_per_index = [eps] * N_INPUTS

    write_bounds(lines, x_star, eps_per_index)

    if output_assertion_kind == 'two_sided':
        v, delta = threshold
        lines.append(f"(assert (or (<= Y_{TARGET_OUTPUT} {v - delta:.8f}) (>= Y_{TARGET_OUTPUT} {v + delta:.8f})))")
    elif output_assertion_kind == 'ceiling':
        lines.append(f"(assert (>= Y_{TARGET_OUTPUT} {threshold:.8f}))")
    elif output_assertion_kind == 'floor':
        lines.append(f"(assert (<= Y_{TARGET_OUTPUT} {threshold:.8f}))")

    return "\n".join(lines)


def compute_thresholds(flat_model, T):
    """Compute the numeric literals used across all four property files.

    Thresholds are derived from the flat model's own predictions so they are
    deterministic and match what the verifier evaluates on the ONNX.

    Args:
        flat_model: trained SurgicalFCNFlat with 6 outputs.
        T:          window length.

    Returns:
        Dict with keys:
            v_expert      - Y_4 prediction on the EXPERT_ANCHOR window.
            delta         - fixed allowed deviation (DELTA = 0.5).
            C             - novice ceiling (v_novice + MARGIN_CEIL).
            L             - expert floor (min expert Y_4 prediction - MARGIN_FLOOR).
            expert_overall - list of Y_4 predictions for all D/E-series trials.
    """
    v_expert = predict_overall(flat_model, load_anchor_window(EXPERT_ANCHOR, T), T)[TARGET_OUTPUT]
    v_novice = predict_overall(flat_model, load_anchor_window(NOVICE_ANCHOR, T), T)[TARGET_OUTPUT]
    C = v_novice + MARGIN_CEIL

    all_trials = load_osats_data('data')
    expert_overall = [
        predict_overall(flat_model, data[:T].T.flatten(), T)[TARGET_OUTPUT]
        for data, osats, subject, trial_num in all_trials
        if subject in EXPERT_SUBJECTS and (HELDOUT_TRIAL is None or trial_num == HELDOUT_TRIAL)
    ]
    min_expert = min(expert_overall)
    L = min_expert - MARGIN_FLOOR

    return {
        'v_expert': v_expert,
        'v_novice': v_novice,
        'delta': DELTA,
        'C': C,
        'L': L,
        'expert_overall': expert_overall,
    }


def generate_all(flat_model, T):
    """Write all four VNN-LIB property files using precomputed thresholds.

    Args:
        flat_model: trained SurgicalFCNFlat with 6 outputs.
        T:          window length.

    Returns:
        Threshold dict from compute_thresholds (for logging/reporting).
    """
    thresholds = compute_thresholds(flat_model, T)
    v_expert = thresholds['v_expert']
    delta    = thresholds['delta']
    C        = thresholds['C']
    L        = thresholds['L']

    properties = [
        (f'property_noise_robustness{PROP_SUFFIX}.vnnlib', 'noise',        EXPERT_ANCHOR, EPSILON_PHYSICAL, None,             'two_sided', (v_expert, delta)),
        (f'property_monotonicity{PROP_SUFFIX}.vnnlib',     'monotonicity', NOVICE_ANCHOR, EPSILON_PHYSICAL, None,             'ceiling',   C),
        (f'property_segmentation{PROP_SUFFIX}.vnnlib',     'segmentation', EXPERT_ANCHOR, EPSILON_PHYSICAL, EPSILON_BOUNDARY, 'two_sided', (v_expert, delta)),
        (f'property_range_floor{PROP_SUFFIX}.vnnlib',      'range',        EXPERT_ANCHOR, EPSILON_PHYSICAL, None,             'floor',     L),
    ]

    for filename, prop_name, anchor, eps, boundary_eps, assertion_kind, threshold in properties:
        text = build_property(prop_name, anchor, flat_model, T, eps, boundary_eps, assertion_kind, threshold)
        with open(filename, 'w') as f:
            f.write(text)
        print(f"Wrote {filename}")

    return thresholds


def run_verifier(onnx_path, vnnlib_path):
    """Invoke the n2v verifier and return its verdict.

    Args:
        onnx_path:   path to the exported flat regression ONNX.
        vnnlib_path: path to the VNN-LIB property file.

    Returns:
        'unsat', 'sat', or 'unknown'.  Returns 'unknown' on crash or no match,
        which is the conservative/sound choice for certification purposes.
    """
    try:
        # Cap the verifier's *resident* memory (5 GB) via a transient systemd
        # cgroup scope so an exact-star blowup at large epsilon OOM-kills only
        # this scope, not the whole WSL VM. We use cgroups rather than
        # `ulimit -v` because the latter limits virtual address space, which
        # torch/numpy/BLAS reserve in the tens of GB regardless of real usage —
        # a -v cap kills the verifier at `import torch`. MemorySwapMax=0 stops
        # the blowup from thrashing into swap before the kill. start_new_session
        # keeps systemd-run + python a single process group so the SIGKILL-on-
        # timeout path below tears down the whole group.
        proc = subprocess.Popen(
            ['systemd-run', '--user', '--scope', '--quiet',
             '-p', 'MemoryMax=5G', '-p', 'MemorySwapMax=0',
             'python3', RUN_INSTANCE, onnx_path, vnnlib_path, '--workers', '1'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            start_new_session=True,
        )
        try:
            stdout, _ = proc.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.communicate()
            return 'unknown'
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            if 'unsat' in line:
                return 'unsat'
            if 'sat' in line:
                return 'sat'
            if 'unknown' in line:
                return 'unknown'
    except Exception:
        pass
    return 'unknown'


def binary_search_epsilon(flat_model, T, prop_name):
    """Binary-search the largest input epsilon for which the verifier returns unsat.

    The output threshold (delta for noise, L for range) is held fixed; only the
    input perturbation epsilon is varied.  'unknown' is treated as 'not certified'
    so the returned value is a sound lower bound on the true robustness radius.

    Args:
        flat_model: trained SurgicalFCNFlat with 6 outputs.
        T:          window length.
        prop_name:  'noise' or 'range'.

    Returns:
        Largest certified epsilon as a float (resolved to ~1/4096 of [0, 1]).
    """
    thresholds = compute_thresholds(flat_model, T)
    if prop_name == 'noise':
        filename       = f'property_noise_robustness{PROP_SUFFIX}.vnnlib'
        assertion_kind = 'two_sided'
        threshold      = (thresholds['v_expert'], thresholds['delta'])
        anchor         = EXPERT_ANCHOR
    else:  # range
        filename       = f'property_range_floor{PROP_SUFFIX}.vnnlib'
        assertion_kind = 'floor'
        threshold      = thresholds['L']
        anchor         = EXPERT_ANCHOR

    onnx_path = ONNX_PATH
    best = EPSILON_PHYSICAL if run_verifier(onnx_path, filename) == 'unsat' else 0.0
    lo, hi = SEARCH_LO, SEARCH_HI

    for _ in range(SEARCH_ITERS):
        m = (lo + hi) / 2
        text = build_property(prop_name, anchor, flat_model, T, m, None, assertion_kind, threshold)
        with open(filename, 'w') as f:
            f.write(text)
        if run_verifier(onnx_path, filename) == 'unsat':
            best = m
            lo = m
        else:
            hi = m

    return best


def load_flat_model(T):
    """Build the flat regression model from the currently configured CHECKPOINT."""
    from model_flat_regression import build_trainable_regression, build_flat_regression
    trained = build_trainable_regression()
    trained.load_state_dict(torch.load(CHECKPOINT, weights_only=True))
    trained.eval()
    return build_flat_regression(trained, T)


def verdict_all(flat_model, T):
    """Generate the four property files at the physical epsilon and verify each.

    Args:
        flat_model: the fold's flat regression model.
        T:          window length.

    Returns:
        (verdicts, thresholds): verdicts is a dict
        {'noise','monotonicity','segmentation','range'} -> 'unsat'/'sat'/'unknown';
        thresholds is the dict returned by generate_all.
    """
    thresholds = generate_all(flat_model, T)
    files = [
        ('noise',        f'property_noise_robustness{PROP_SUFFIX}.vnnlib'),
        ('monotonicity', f'property_monotonicity{PROP_SUFFIX}.vnnlib'),
        ('segmentation', f'property_segmentation{PROP_SUFFIX}.vnnlib'),
        ('range',        f'property_range_floor{PROP_SUFFIX}.vnnlib'),
    ]
    verdicts = {name: run_verifier(ONNX_PATH, path) for name, path in files}
    return verdicts, thresholds


def run_fold(fold, T):
    """Configure, generate, verify, and binary-search certified epsilon for one fold.

    Args:
        fold: super-trial index 1..5.
        T:    window length.

    Returns:
        Result dict with the fold's anchors, thresholds, four verdicts, and the
        certified noise/range epsilons.
    """
    configure_fold(fold)
    print(f"--- Fold {fold}: model={CHECKPOINT}  expert={EXPERT_ANCHOR}  novice={NOVICE_ANCHOR} ---")
    flat = load_flat_model(T)

    verdicts, th = verdict_all(flat, T)
    print(f"    thresholds: v_expert={th['v_expert']:.4f}  v_novice={th['v_novice']:.4f}  "
          f"C={th['C']:.4f}  L={th['L']:.4f}")
    print(f"    verdicts:   noise={verdicts['noise']}  monotonicity={verdicts['monotonicity']}  "
          f"segmentation={verdicts['segmentation']}  range={verdicts['range']}")

    eps_noise = binary_search_epsilon(flat, T, 'noise')
    eps_range = binary_search_epsilon(flat, T, 'range')
    print(f"    certified:  eps[noise]={eps_noise:.6f}  eps[range]={eps_range:.6f}")

    return {
        'fold': fold,
        'expert_anchor': EXPERT_ANCHOR.replace('Suturing_', ''),
        'novice_anchor': NOVICE_ANCHOR.replace('Suturing_', ''),
        'v_expert': th['v_expert'], 'v_novice': th['v_novice'],
        'C': th['C'], 'L': th['L'],
        'verdicts': verdicts,
        'eps_noise': eps_noise, 'eps_range': eps_range,
    }


def write_results_table(results):
    """Render and persist the per-fold verification summary to regression_results.txt."""
    import statistics

    lines = []
    lines.append("Per-fold formal robustness verification — LOSO, held-out anchors")
    lines.append(f"(physical noise bound epsilon = {EPSILON_PHYSICAL})")
    lines.append("")

    th_hdr = f"{'Fold':<4} | {'Expert':<6} | {'Novice':<6} | {'v_expert':>9} | {'v_novice':>9} | {'C':>8} | {'L':>8}"
    lines.append(th_hdr)
    lines.append("-" * len(th_hdr))
    for r in results:
        lines.append(f"{r['fold']:<4} | {r['expert_anchor']:<6} | {r['novice_anchor']:<6} | "
                     f"{r['v_expert']:>9.4f} | {r['v_novice']:>9.4f} | {r['C']:>8.4f} | {r['L']:>8.4f}")
    lines.append("")

    v_hdr = (f"{'Fold':<4} | {'noise':<7} | {'mono':<7} | {'seg':<7} | {'range':<7} | "
             f"{'eps[noise]':>11} | {'eps[range]':>11}")
    lines.append(v_hdr)
    lines.append("-" * len(v_hdr))
    for r in results:
        v = r['verdicts']
        lines.append(f"{r['fold']:<4} | {v['noise']:<7} | {v['monotonicity']:<7} | {v['segmentation']:<7} | "
                     f"{v['range']:<7} | {r['eps_noise']:>11.6f} | {r['eps_range']:>11.6f}")
    lines.append("")

    en = [r['eps_noise'] for r in results]
    er = [r['eps_range'] for r in results]
    lines.append(f"certified eps[noise]: mean={statistics.mean(en):.6f}  min={min(en):.6f}  "
                 f"max={max(en):.6f}  (mean = {statistics.mean(en) / EPSILON_PHYSICAL:.1f}x physical)")
    lines.append(f"certified eps[range]: mean={statistics.mean(er):.6f}  min={min(er):.6f}  "
                 f"max={max(er):.6f}  (mean = {statistics.mean(er) / EPSILON_PHYSICAL:.1f}x physical)")

    text = "\n".join(lines)
    print("\n" + text)
    with open('regression_results.txt', 'w') as f:
        f.write(text + "\n")


def main():
    """Entry point.

    Modes (first positional arg):
        allfolds          run the full per-fold pipeline (default) and write the table
        fold N            run the full pipeline for a single fold N (debugging)
        generate          legacy: write the 4 files for the single all-data model
        search            legacy: binary-search the single all-data model
    """
    mode = sys.argv[1] if len(sys.argv) > 1 else 'allfolds'

    if mode == 'allfolds':
        results = [run_fold(fold, T) for fold in range(1, 6)]
        write_results_table(results)
        return
    if mode == 'fold':
        run_fold(int(sys.argv[2]), T)
        return

    # Legacy single-model modes (operate on the configured defaults).
    flat = load_flat_model(T)
    if mode == 'generate':
        thresholds = generate_all(flat, T)
        print(f"v_expert        = {thresholds['v_expert']:.6f}")
        print(f"delta           = {thresholds['delta']:.6f}")
        print(f"C (novice ceil) = {thresholds['C']:.6f}")
        print(f"L (expert floor)= {thresholds['L']:.6f}")
    elif mode == 'search':
        for prop_name in ('noise', 'range'):
            eps = binary_search_epsilon(flat, T, prop_name)
            print(f"certified_eps[{prop_name}] = {eps:.6f}  (ratio to physical: {eps / EPSILON_PHYSICAL:.2f}x)")
    else:
        print(f"Unknown mode {mode!r}. Use 'allfolds', 'fold N', 'generate', or 'search'.")
        sys.exit(1)


if __name__ == '__main__':
    main()
