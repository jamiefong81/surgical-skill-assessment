from model import SurgicalFCN
from model_flat import SurgicalFCNFlat, verify_equivalence


N_OUTPUTS = 6
DUMMY_TIMESTEPS = 10  # must be same in export_onnx_regression.py


def build_trainable_regression(num_outputs=N_OUTPUTS):
    return SurgicalFCN(num_outputs)


def build_flat_regression(trained, T):
    return SurgicalFCNFlat.from_trained(trained, T=T)


def check_regression_equivalence(trained, flat, T):
    return verify_equivalence(trained, flat, T)