"""
Export the trained SurgicalFCN to ONNX, then load it via n2v's model_loader
(which uses onnx2torch internally) and confirm all layers are recognized.

Usage:
    1. python train.py          # produces best_model.pth
    2. python export_onnx.py    # export, load, verify
"""

import sys
import torch

sys.path.insert(0, 'n2v')

from model import SurgicalFCN
from model_flat import SurgicalFCNFlat, verify_equivalence
from n2v.utils.model_loader import load_onnx, get_model_summary

ONNX_PATH = 'surgical_fcn.onnx'
PTH_PATH = 'best_model.pth'
NUM_CLASSES = 3
DUMMY_TIMESTEPS = 10   # must match T in generate_property.py

# --- Section A: build flat (verifier-friendly) model from trained weights ---

trained = SurgicalFCN(num_classes=NUM_CLASSES)
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
    dynamo=False,          # force legacy TorchScript exporter so opset_version is honoured
    opset_version=13,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input':  {0: 'batch'},
        'output': {0: 'batch'},
    },
)
print(f'Exported to {ONNX_PATH}')

# --- Section C: load via model_loader (onnx2torch.convert internally) ---

onnx_model = load_onnx(ONNX_PATH)
print('Loaded ONNX model via model_loader.load_onnx — all ops recognized by onnx2torch')

# --- Section D: confirm all layers recognized via forward-pass hooks ---

summary = get_model_summary(onnx_model, input_shape=(76, DUMMY_TIMESTEPS))

print(f'\nLayers recognized: {len(summary)}')
print(f'{"Layer":<35} {"Input shape":<25} {"Output shape":<25} {"Params":>8}')
print('-' * 97)
for name, info in summary.items():
    print(f'{name:<35} {str(info["input_shape"]):<25} {str(info["output_shape"]):<25} {info["nb_params"]:>8}')

total_params = sum(v['nb_params'] for v in summary.values())
print(f'\nTotal parameters across recognized layers: {total_params:,}')
