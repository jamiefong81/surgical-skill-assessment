"""Generate VNN-LIB property files for the regression OSATS model and run epsilon searches.

Usage:
    python3 generate_property_regression.py generate   # write the four .vnnlib files
    python3 generate_property_regression.py search     # binary-search certified eps for noise + range
"""

import os
import re
import signal
import subprocess
import sys

import numpy as np
import torch

import config
from dataset import load_osats_data, reorder_columns

T                = config.T_REGRESSION   # window length; overridden by --T flag in main()
N_OUTPUTS        = 6
TARGET_OUTPUT    = 4        # index of Overall Performance in the OSATS vector
EPSILON_PHYSICAL = 0.001    # physically motivated da Vinci encoder noise bound
EPSILON_BOUNDARY = 0.01     # larger perturbation for segment-boundary timesteps
DELTA            = 0.5      # allowed Y_4 deviation (half an OSATS rater point)
MARGIN_FLOOR     = 0.25     # subtracted from min expert prediction to set floor L
MARGIN_CEIL      = 0.5      # added to novice prediction to set ceiling C
# BOUNDARY_TIMESTEPS and N_INPUTS are computed from T at runtime (first 2 / last 2 timesteps)
EXPERT_SUBJECTS  = ('D', 'E')   # JIGSAWS self-proclaimed experts (>100h)
APPROX_ONLY      = False    # set True via --approx to skip exact-star escalation
ABCROWN          = False    # set True via --abcrown to use alpha-beta-CROWN instead of n2v
RUN_SEARCH       = True     # set False via --no-search to skip the certified-eps binary search
TASK             = 'Suturing'  # overridden by --task flag in main()

# Artifact directories (root-anchored; see config.py).
MODEL_DIR        = config.MODEL_DIR
ONNX_DIR         = config.ONNX_DIR
PROP_DIR         = config.PROP_DIR
RESULTS_DIR      = config.RESULTS_DIR

# --- Fold configuration (mutated by configure_fold) ---
# Defaults target the legacy single all-data model so `generate`/`search` and
# inspect_osats_prediction.py keep working unchanged. The per-fold pipeline
# (`allfolds`) overwrites these via configure_fold so every anchor, threshold
# and certificate refers to a trial the fold model was NOT trained on.
EXPERT_ANCHOR    = "Suturing_D001"
NOVICE_ANCHOR    = "Suturing_B001"
ONNX_PATH        = f"{ONNX_DIR}/surgical_fcn_regression.onnx"
CHECKPOINT       = f"{MODEL_DIR}/best_model_regression.pth"
HELDOUT_TRIAL    = None     # if set, the expert-floor set is restricted to held-out experts
PROP_SUFFIX      = ""       # filename suffix for the generated .vnnlib files
SEARCH_LO        = 0.0
SEARCH_HI        = 0.1      # 100x physical bound. Wide enough to resolve the range
                           # radius (which pinned at the old 0.02 ceiling) without
                           # reaching the eps where exact-star reliably explodes;
                           # the cgroup memory guard in run_verifier makes the larger
                           # probes safe (a blowup -> clean 'unknown' -> search narrows).
SEARCH_ITERS     = 12       # resolves eps to ~1/4096 of the bracket
RUN_INSTANCE     = config.RUN_INSTANCE


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
    slug            = config.task_slug(TASK)
    task_infix      = f"_{slug}" if slug else ""
    t_suffix        = f"_T{T}" if T != config.T_REGRESSION else ""
    method_suffix   = "_approx" if APPROX_ONLY else ""
    verifier_suffix = "_abcrown" if ABCROWN else ""
    EXPERT_ANCHOR = f"{TASK}_D{fold:03d}"
    NOVICE_ANCHOR = f"{TASK}_B{fold:03d}"
    ONNX_PATH     = f"{ONNX_DIR}/surgical_fcn_regression{task_infix}_fold{fold}{t_suffix}.onnx"
    CHECKPOINT    = f"{MODEL_DIR}/best_model_regression{task_infix}_fold{fold}.pth"
    HELDOUT_TRIAL = fold
    PROP_SUFFIX   = f"_fold{fold}{task_infix}{t_suffix}{method_suffix}{verifier_suffix}"


def prop_path(stem):
    """Path to a generated property file for the current fold, e.g.
    properties/property_noise_robustness_fold3.vnnlib."""
    return f"{PROP_DIR}/property_{stem}{PROP_SUFFIX}.vnnlib"


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
    # Derive task from anchor name so this works for all three JIGSAWS tasks.
    # 'Knot_Tying_D001' -> task='Knot_Tying', 'Suturing_D001' -> task='Suturing'
    # Loading kinematics directly avoids a dependency on the OSATS meta file,
    # which means anchors without OSATS scores (e.g. Needle_Passing_B005) work fine.
    task     = '_'.join(anchor_name.split('_')[:-1])
    kin_path = os.path.join(config.DATA_DIR, task, 'kinematics', 'AllGestures',
                            anchor_name + '.txt')
    if not os.path.exists(kin_path):
        raise ValueError(f"Anchor kinematics not found: {kin_path}")
    data = np.loadtxt(kin_path, dtype=np.float32)
    data = reorder_columns(data)
    return data[:T].T.flatten()


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
    n_inputs = 76 * T
    boundary_timesteps = (0, 1, T - 2, T - 1)
    x_star = load_anchor_window(anchor_name, T)

    lines = []
    for i in range(n_inputs):
        lines.append(f"(declare-const X_{i} Real)")
    for i in range(N_OUTPUTS):
        lines.append(f"(declare-const Y_{i} Real)")

    if prop_name == 'segmentation':
        eps_per_index = [boundary_eps if (i % T) in boundary_timesteps else eps for i in range(n_inputs)]
    else:
        eps_per_index = [eps] * n_inputs

    write_bounds(lines, x_star, eps_per_index)

    if output_assertion_kind == 'two_sided':
        v, delta = threshold
        lo = f"(<= Y_{TARGET_OUTPUT} {v - delta:.8f})"
        hi = f"(>= Y_{TARGET_OUTPUT} {v + delta:.8f})"
        if ABCROWN:
            # alpha-beta-CROWN's read_vnnlib.py requires the strict VNN-COMP DNF
            # form: each disjunct wrapped in (and ...). n2v accepts bare comparisons
            # so its baseline files keep the simpler unwrapped form (left branch).
            lines.append(f"(assert (or (and {lo}) (and {hi})))")
        else:
            lines.append(f"(assert (or {lo} {hi}))")
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

    all_trials = load_osats_data(task=TASK)
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
    L        = thresholds['L']

    # Monotonicity uses the expert floor L as the novice ceiling: proving the
    # novice's Y_4 stays <= L while property #4 proves the expert's Y_4 stays
    # >= L composes into a genuine skill-ordering certificate (novice <= L <=
    # expert) over both perturbation balls. (The 'ceiling' assertion's violation
    # is Y_4 >= L, so unsat == novice provably below the expert floor.)
    properties = [
        (prop_path('noise_robustness'), 'noise',        EXPERT_ANCHOR, EPSILON_PHYSICAL, None,             'two_sided', (v_expert, delta)),
        (prop_path('monotonicity'),     'monotonicity', NOVICE_ANCHOR, EPSILON_PHYSICAL, None,             'ceiling',   L),
        (prop_path('segmentation'),     'segmentation', EXPERT_ANCHOR, EPSILON_PHYSICAL, EPSILON_BOUNDARY, 'two_sided', (v_expert, delta)),
        (prop_path('range_floor'),      'range',        EXPERT_ANCHOR, EPSILON_PHYSICAL, None,             'floor',     L),
    ]

    os.makedirs(PROP_DIR, exist_ok=True)
    for filename, prop_name, anchor, eps, boundary_eps, assertion_kind, threshold in properties:
        text = build_property(prop_name, anchor, flat_model, T, eps, boundary_eps, assertion_kind, threshold)
        with open(filename, 'w') as f:
            f.write(text)
        print(f"Wrote {filename}")

    return thresholds


def output_violation_predicate(vnnlib_path):
    """Parse the Y_TARGET violation assertion from a VNN-LIB file.

    Returns a callable ``y4 -> bool`` that is True when y4 satisfies the asserted
    output violation (the point is a genuine counterexample on the output side),
    or None if no Y_TARGET assertion is found. Handles the three forms we emit:
        (assert (>= Y_4 t))                       -> y4 >= t
        (assert (<= Y_4 t))                       -> y4 <= t
        (assert (or (<= Y_4 a) (>= Y_4 b)))       -> y4 <= a or y4 >= b
    Checks are strict (no slack), so a borderline witness is conservatively
    rejected — the safe direction for trusting a 'sat'.
    """
    yvar = f"Y_{TARGET_OUTPUT}"
    num = r'-?\d+\.?\d*(?:[eE][-+]?\d+)?'
    for raw in open(vnnlib_path):
        line = raw.strip()
        if not line.startswith("(assert") or yvar not in line:
            continue
        nums = [float(n) for n in re.findall(num, line.replace(yvar, ''))]
        if "(or" in line:
            a, b = nums[0], nums[1]
            return lambda y, a=a, b=b: (y <= a) or (y >= b)
        if ">=" in line:
            t = nums[0]
            return lambda y, t=t: y >= t
        if "<=" in line:
            t = nums[0]
            return lambda y, t=t: y <= t
    return None


def counterexample_is_real(stdout, vnnlib_path, flat_model, T):
    """Re-evaluate a verifier-reported counterexample through the model.

    Parses the ``(X_i value)`` block from the verifier's stdout, runs the flat
    model on it, and checks the parsed output violation. Returns True only if the
    witness genuinely violates the property — making a 'sat' sound regardless of
    any over-approximation in the verifier (which is unsound for witness-free
    'sat'). Returns False if the witness is missing, malformed, or non-violating.
    """
    pred = output_violation_predicate(vnnlib_path)
    if pred is None:
        return False
    n_inputs = 76 * T
    xs = {int(i): float(v)
          for i, v in re.findall(r'\(X_(\d+)\s+(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\)', stdout)}
    if len(xs) < n_inputs:
        return False
    x = np.array([xs[i] for i in range(n_inputs)], dtype=np.float32)
    y4 = predict_overall(flat_model, x, T)[TARGET_OUTPUT]
    return bool(pred(y4))


def _run_n2v(onnx_path, vnnlib_path, flat_model, T):
    """Invoke n2v and return 'unsat', 'sat', or 'unknown'."""
    cmd = ['systemd-run', '--user', '--scope', '--quiet',
           '-p', 'MemoryMax=5G', '-p', 'MemorySwapMax=0',
           'python3', RUN_INSTANCE, onnx_path, vnnlib_path, '--workers', '1']
    if APPROX_ONLY:
        cmd.extend(['--category', 'collins_rul_cnn_2022'])
    proc = subprocess.Popen(
        cmd,
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
            if flat_model is not None and not counterexample_is_real(stdout, vnnlib_path, flat_model, T):
                return 'unknown'
            return 'sat'
        if 'unknown' in line:
            return 'unknown'
    return 'unknown'


def _run_abcrown(onnx_path, vnnlib_path, flat_model, T):
    """Invoke alpha-beta-CROWN and return 'unsat', 'sat', or 'unknown'.

    alpha-beta-CROWN writes its result to a file rather than stdout. The first
    line of that file is the VNN-COMP verdict; if 'sat' the remainder contains
    the counterexample in the same ((X_i v)...) format, which we validate with
    counterexample_is_real for consistency.
    """
    import tempfile
    if not os.path.isfile(config.ABCROWN_PYTHON):
        print(f"  [abcrown] venv not found at {config.ABCROWN_PYTHON} — returning unknown")
        return 'unknown'
    results_file = None
    try:
        fd, results_file = tempfile.mkstemp(suffix='.txt')
        os.close(fd)
        cmd = [config.ABCROWN_PYTHON, config.ABCROWN_SCRIPT,
               '--config', config.ABCROWN_CONFIG,
               '--onnx_path', onnx_path,
               '--vnnlib_path', vnnlib_path,
               '--results_file', results_file,
               '--timeout', '120',
               '--save_adv_example']
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=130)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            return 'unknown'
        with open(results_file) as f:
            content = f.read()
        first = content.strip().splitlines()[0].strip().lower() if content.strip() else ''
        if first == 'unsat':
            return 'unsat'
        if first == 'sat':
            if flat_model is not None and not counterexample_is_real(content, vnnlib_path, flat_model, T):
                return 'unknown'
            return 'sat'
        return 'unknown'
    except Exception:
        return 'unknown'
    finally:
        if results_file and os.path.exists(results_file):
            os.unlink(results_file)


def run_verifier(onnx_path, vnnlib_path, flat_model=None, T=None):
    """Invoke the configured verifier and return its verdict.

    Dispatches to alpha-beta-CROWN when ABCROWN is True, otherwise n2v.
    n2v uses a cgroup RSS cap to prevent exact-star memory blowup from
    killing the WSL VM; alpha-beta-CROWN manages its own GPU memory and
    does not need the cap.

    Args:
        onnx_path:   path to the exported flat regression ONNX.
        vnnlib_path: path to the VNN-LIB property file.
        flat_model:  if provided, a reported 'sat' is validated by re-evaluating
                     the counterexample through this model. Guards against
                     unsound witness-free sat from over-approximating reachability.
        T:           window length (needed to reshape the counterexample).

    Returns:
        'unsat', 'sat', or 'unknown'.
    """
    try:
        if ABCROWN:
            return _run_abcrown(onnx_path, vnnlib_path, flat_model, T)
        # n2v path (default)
        # Cap the verifier's *resident* memory (5 GB) via a transient systemd
        # cgroup scope so an exact-star blowup at large epsilon OOM-kills only
        # this scope, not the whole WSL VM. We use cgroups rather than
        # `ulimit -v` because the latter limits virtual address space, which
        # torch/numpy/BLAS reserve in the tens of GB regardless of real usage —
        # a -v cap kills the verifier at `import torch`. MemorySwapMax=0 stops
        # the blowup from thrashing into swap before the kill. start_new_session
        # keeps systemd-run + python a single process group so the SIGKILL-on-
        # timeout path below tears down the whole group.
        return _run_n2v(onnx_path, vnnlib_path, flat_model, T)
    except Exception:
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
        filename       = prop_path('noise_robustness')
        assertion_kind = 'two_sided'
        threshold      = (thresholds['v_expert'], thresholds['delta'])
        anchor         = EXPERT_ANCHOR
    else:  # range
        filename       = prop_path('range_floor')
        assertion_kind = 'floor'
        threshold      = thresholds['L']
        anchor         = EXPERT_ANCHOR

    os.makedirs(PROP_DIR, exist_ok=True)
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
        ('noise',        prop_path('noise_robustness')),
        ('monotonicity', prop_path('monotonicity')),
        ('segmentation', prop_path('segmentation')),
        ('range',        prop_path('range_floor')),
    ]
    verdicts = {name: run_verifier(ONNX_PATH, path, flat_model, T) for name, path in files}
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
    sep = "yes" if th['v_novice'] < th['L'] else "NO (novice >= floor)"
    print(f"    thresholds: v_expert={th['v_expert']:.4f}  v_novice={th['v_novice']:.4f}  "
          f"L={th['L']:.4f}  (novice<L for monotonicity: {sep})")
    print(f"    verdicts:   noise={verdicts['noise']}  monotonicity={verdicts['monotonicity']}  "
          f"segmentation={verdicts['segmentation']}  range={verdicts['range']}")

    if RUN_SEARCH:
        eps_noise = binary_search_epsilon(flat, T, 'noise')
        eps_range = binary_search_epsilon(flat, T, 'range')
        print(f"    certified:  eps[noise]={eps_noise:.6f}  eps[range]={eps_range:.6f}")
    else:
        eps_noise = eps_range = None
        print(f"    certified:  (binary search skipped via --no-search)")

    return {
        'fold': fold,
        'expert_anchor': EXPERT_ANCHOR.split('_')[-1],
        'novice_anchor': NOVICE_ANCHOR.split('_')[-1],
        'v_expert': th['v_expert'], 'v_novice': th['v_novice'], 'L': th['L'],
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

    lines.append("L = expert floor = monotonicity ceiling. Monotonicity (novice Y_4 <= L)")
    lines.append("can only hold where v_novice < L; otherwise the novice already outscores")
    lines.append("the expert floor and the property is correctly violated (sat).")
    lines.append("")
    th_hdr = (f"{'Fold':<4} | {'Expert':<6} | {'Novice':<6} | {'v_expert':>9} | {'v_novice':>9} | "
              f"{'L':>8} | {'novice<L':>8}")
    lines.append(th_hdr)
    lines.append("-" * len(th_hdr))
    for r in results:
        sep = 'yes' if r['v_novice'] < r['L'] else 'NO'
        lines.append(f"{r['fold']:<4} | {r['expert_anchor']:<6} | {r['novice_anchor']:<6} | "
                     f"{r['v_expert']:>9.4f} | {r['v_novice']:>9.4f} | {r['L']:>8.4f} | {sep:>8}")
    lines.append("")

    searched = any(r['eps_noise'] is not None for r in results)

    v_hdr = (f"{'Fold':<4} | {'noise':<7} | {'mono':<7} | {'seg':<7} | {'range':<7} | "
             f"{'eps[noise]':>11} | {'eps[range]':>11}")
    lines.append(v_hdr)
    lines.append("-" * len(v_hdr))
    for r in results:
        v = r['verdicts']
        en = f"{r['eps_noise']:>11.6f}" if r['eps_noise'] is not None else f"{'skipped':>11}"
        er = f"{r['eps_range']:>11.6f}" if r['eps_range'] is not None else f"{'skipped':>11}"
        lines.append(f"{r['fold']:<4} | {v['noise']:<7} | {v['monotonicity']:<7} | {v['segmentation']:<7} | "
                     f"{v['range']:<7} | {en} | {er}")
    lines.append("")

    if searched:
        en = [r['eps_noise'] for r in results]
        er = [r['eps_range'] for r in results]
        lines.append(f"certified eps[noise]: mean={statistics.mean(en):.6f}  min={min(en):.6f}  "
                     f"max={max(en):.6f}  (mean = {statistics.mean(en) / EPSILON_PHYSICAL:.1f}x physical)")
        lines.append(f"certified eps[range]: mean={statistics.mean(er):.6f}  min={min(er):.6f}  "
                     f"max={max(er):.6f}  (mean = {statistics.mean(er) / EPSILON_PHYSICAL:.1f}x physical)")
    else:
        lines.append("certified eps: binary search skipped (--no-search) — verdicts at physical epsilon only")

    slug            = config.task_slug(TASK)
    task_infix      = f"_{slug}" if slug else ""
    t_suffix        = f"_T{T}" if T != config.T_REGRESSION else ""
    method_suffix   = "_approx" if APPROX_ONLY else ""
    verifier_suffix = "_abcrown" if ABCROWN else ""
    search_suffix   = "" if searched else "_verdicts"
    out_path = f'{RESULTS_DIR}/regression_results{task_infix}{t_suffix}{method_suffix}{verifier_suffix}{search_suffix}.txt'
    text = "\n".join(lines)
    print("\n" + text)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(text + "\n")


def main():
    """Entry point.

    Modes (first positional arg):
        allfolds          run the full per-fold pipeline (default) and write the table
        fold N            run the full pipeline for a single fold N (debugging)
        generate          legacy: write the 4 files for the single all-data model
        search            legacy: binary-search the single all-data model
    """
    global T, APPROX_ONLY, ABCROWN, RUN_SEARCH, TASK
    args = sys.argv[1:]

    # Extract flag arguments before positional mode parsing.
    APPROX_ONLY = '--approx' in args
    if APPROX_ONLY:
        args.remove('--approx')
    ABCROWN = '--abcrown' in args
    if ABCROWN:
        args.remove('--abcrown')
    if '--no-search' in args:
        RUN_SEARCH = False
        args.remove('--no-search')
    if '--task' in args:
        idx = args.index('--task')
        TASK = args[idx + 1]
        args = args[:idx] + args[idx + 2:]
    if '--T' in args:
        idx = args.index('--T')
        T = int(args[idx + 1])
        args = args[:idx] + args[idx + 2:]

    mode = args[0] if args else 'allfolds'

    if mode == 'allfolds':
        results = [run_fold(fold, T) for fold in range(1, 6)]
        write_results_table(results)
        return
    if mode == 'fold':
        run_fold(int(args[1]), T)
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
