import os
import sys
import torch

import config

sys.path.insert(0, config.N2V_DIR)

# Positional args:
#   argv[1]: LOSO fold index 1..5 (optional; omit for the legacy single model)
#   argv[2]: T (timesteps window, optional; defaults to config.T_REGRESSION=10)
# When T differs from the default a _T{N} suffix is appended to the ONNX filename
# so the new export never overwrites the existing T=10 file.
FOLD = sys.argv[1] if len(sys.argv) > 1 else None
T_OVERRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else None
DUMMY_TIMESTEPS = T_OVERRIDE if T_OVERRIDE is not None else config.T_REGRESSION
T_SUFFIX = f"_T{DUMMY_TIMESTEPS}" if T_OVERRIDE is not None else ""
if FOLD is not None:
    ONNX_PATH = os.path.join(config.ONNX_DIR, f"surgical_fcn_regression_fold{FOLD}{T_SUFFIX}.onnx")
    PTH_PATH = os.path.join(config.MODEL_DIR, f"best_model_regression_fold{FOLD}.pth")
else:
    ONNX_PATH = os.path.join(config.ONNX_DIR, f"surgical_fcn_regression{T_SUFFIX}.onnx")
    PTH_PATH = os.path.join(config.MODEL_DIR, "best_model_regression.pth")
os.makedirs(config.ONNX_DIR, exist_ok=True)
NUM_OUTPUTS = 6
OPSET_VERSION = 13

from model import SurgicalFCN
from model_flat import SurgicalFCNFlat, verify_equivalence
from model_flat_regression import build_flat_regression
from n2v.utils.model_loader import load_onnx, get_model_summary

# --- Section A: build flat (verifier-friendly) model from trained weights ---

trained = SurgicalFCN(num_classes=NUM_OUTPUTS)
trained.load_state_dict(torch.load(PTH_PATH, map_location='cpu'))
trained.eval()

flat = SurgicalFCNFlat.from_trained(trained, T=DUMMY_TIMESTEPS)

max_diff = verify_equivalence(trained, flat, T=DUMMY_TIMESTEPS)
print(f'Equivalence verified: max |trained(x) - flat(x)| = {max_diff:.2e}')

# --- Section B: export the flat model to ONNX ---

dummy = torch.randn(1, 76, DUMMY_TIMESTEPS)
torch.onnx.export(
    flat,
    dummy,
    ONNX_PATH,
    dynamo=False,
    opset_version=13,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch'},
        'output': {0: 'batch'}
    }
)
print(f"Exported to: {ONNX_PATH}")

# --- Section C: load via model_loader (onnx2torch.convert internally) ---

onnx_model = load_onnx(ONNX_PATH)
print('Loaded ONNX model via model_loader.load_onnx — all ops recognized by onnx2torch')

# --- Section D: confirm all layers recognized via forward-pass hooks ---

summary = get_model_summary(onnx_model, input_shape=(76, DUMMY_TIMESTEPS))
total_params = sum(v['nb_params'] for v in summary.values())

if FOLD is None:
    # Full per-layer table only for the single-model export; folds print one line.
    print(f'\nLayers recognized: {len(summary)}')
    print(f'{"Layer":<35} {"Input shape":<25} {"Output shape":<25} {"Params":>8}')
    print('-' * 97)
    for name, info in summary.items():
        print(f'{name:<35} {str(info["input_shape"]):<25} {str(info["output_shape"]):<25} {info["nb_params"]:>8}')
    print(f'\nTotal parameters across recognized layers: {total_params:,}')
else:
    print(f'  fold {FOLD}: {len(summary)} layers recognized, {total_params:,} params')
