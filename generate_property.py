import os
import numpy as np

import config

# --- config ---
DATA_PATH = os.path.join(config.DATA_DIR, 'Suturing', 'kinematics', 'AllGestures', 'Suturing_D001.txt')
VNNLIB_PATH = os.path.join(config.PROP_DIR, 'property_D001.vnnlib')
EPSILON = 0.001
T = config.T_CLASSIFICATION   # must match DUMMY_TIMESTEPS in export_onnx.py
TRUE_CLASS = 2

# --- load data ---
data = np.loadtxt(DATA_PATH)
data = data[:T, :]
x_star = data.T.flatten()

# --- input variable declaration ---
lines = []
for i in range(len(x_star)):
    lines.append(f"(declare-const X_{i} Real)")

# --- output variable declaration ---
for i in range(3):
    lines.append(f"(declare-const Y_{i} Real)")

# --- input bound assertions ---
for i in range (len(x_star)):
    lines.append(f"(assert (>= X_{i} {x_star[i] - EPSILON:.8f}))")
    lines.append(f"(assert (<= X_{i} {x_star[i] + EPSILON:.8f}))")

# --- output property assertion ---
lines.append("(assert (or (<= Y_2 Y_0) (<= Y_2 Y_1)))")

# --- write to file ---
os.makedirs(os.path.dirname(VNNLIB_PATH), exist_ok=True)
with open(VNNLIB_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
