"""Central configuration: filesystem layout and shared constants.

All paths are anchored to this file's directory (the repo root), so scripts work
regardless of the current working directory — no more CWD-relative 'data' / 'n2v'
/ 'models' literals scattered across modules. Import from here instead of
hardcoding paths.
"""

import os

# Repo root = directory containing this file.
ROOT = os.path.dirname(os.path.abspath(__file__))

# Inputs / vendored deps.
DATA_DIR     = os.path.join(ROOT, "data")
N2V_DIR      = os.path.join(ROOT, "n2v")
RUN_INSTANCE = os.path.join(N2V_DIR, "examples", "VNN-COMP", "run_instance.py")

# Generated-artifact directories.
MODEL_DIR    = os.path.join(ROOT, "models")
ONNX_DIR     = os.path.join(ROOT, "onnx")
PROP_DIR     = os.path.join(ROOT, "properties")
RESULTS_DIR  = os.path.join(ROOT, "results")

# Window length (timesteps) per task — must match between training, export and
# property generation within each task.
T_REGRESSION     = 10
T_CLASSIFICATION = 20
