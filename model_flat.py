"""
Verifier-friendly reformulation of SurgicalFCN.

The trained SurgicalFCN implements grouped convolutions in Python (slicing the
input by channel, applying per-group Conv1ds, concatenating), which traces to
ONNX Slice/Concat ops that n2v's flat-Star reachability cannot interpret on the
channel axis. SurgicalFCNFlat is mathematically identical but expresses every
operation as a single dense Conv1d / Linear layer, so its ONNX graph contains
only Conv, Relu, Flatten, and Gemm — all supported by n2v Star reachability.

The grouped behavior is preserved by zeroing out the off-block entries of the
unified weight tensors (block-structured weights). The temporal mean-pool is
expressed as Flatten + Linear with block-averaging weights, eliminating the
ReduceMean op. The model is parameterized by a fixed window length T because
the pool Linear's input size depends on T.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import SurgicalFCN, SUBCLUSTERS, N_MANIPULATORS, N_SUBCLUSTERS, BLOCK_SIZE


class SurgicalFCNFlat(nn.Module):
    def __init__(self, T, num_classes=3):
        super().__init__()
        self.T = T
        self.conv1 = nn.Conv1d(76, 160, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv1d(160, 64, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1, bias=True)
        self.flatten = nn.Flatten()
        self.pool_linear = nn.Linear(32 * T, 32, bias=False)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.pool_linear(x)
        return self.classifier(x)

    @classmethod
    def from_trained(cls, original: SurgicalFCN, T: int) -> 'SurgicalFCNFlat':
        flat = cls(T=T, num_classes=original.fc.out_features)

        with torch.no_grad():
            # Stage 1: 20 sub-cluster Conv1ds -> single Conv1d(76, 160)
            flat.conv1.weight.zero_()
            flat.conv1.bias.zero_()
            for m in range(N_MANIPULATORS):
                ch_offset = 0
                for k, sc_size in enumerate(SUBCLUSTERS):
                    conv_idx = m * N_SUBCLUSTERS + k
                    in_start = m * BLOCK_SIZE + ch_offset
                    in_end = in_start + sc_size
                    out_start = conv_idx * 8
                    out_end = out_start + 8

                    src = original.layer1_convs[conv_idx]
                    flat.conv1.weight[out_start:out_end, in_start:in_end, :] = src.weight
                    flat.conv1.bias[out_start:out_end] = src.bias

                    ch_offset += sc_size

            # Stage 2: 4 manipulator Conv1ds -> single Conv1d(160, 64)
            flat.conv2.weight.zero_()
            flat.conv2.bias.zero_()
            manip_in = N_SUBCLUSTERS * 8   # 40
            manip_out = 16
            for m in range(N_MANIPULATORS):
                in_start = m * manip_in
                in_end = in_start + manip_in
                out_start = m * manip_out
                out_end = out_start + manip_out

                src = original.layer2_convs[m]
                flat.conv2.weight[out_start:out_end, in_start:in_end, :] = src.weight
                flat.conv2.bias[out_start:out_end] = src.bias

            # Stage 3: identical
            flat.conv3.weight.copy_(original.layer3_conv.weight)
            flat.conv3.bias.copy_(original.layer3_conv.bias)

            # Temporal mean-pool: Flatten + block-averaging Linear
            # flatten of (1, 32, T) gives flat[i*T + t] = x[i, t],
            # so W[i, i*T:(i+1)*T] = 1/T makes out[i] = mean_t x[i, t].
            flat.pool_linear.weight.zero_()
            for i in range(32):
                flat.pool_linear.weight[i, i * T:(i + 1) * T] = 1.0 / T

            # Classifier: identical
            flat.classifier.weight.copy_(original.fc.weight)
            flat.classifier.bias.copy_(original.fc.bias)

        flat.eval()
        return flat


def verify_equivalence(original: SurgicalFCN, flat: SurgicalFCNFlat, T: int,
                       atol: float = 1e-5, seed: int = 0) -> float:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 76, T, generator=g)
    original.eval()
    flat.eval()
    with torch.no_grad():
        out_orig = original(x)
        out_flat = flat(x)
    max_diff = (out_orig - out_flat).abs().max().item()
    if not torch.allclose(out_orig, out_flat, atol=atol):
        raise AssertionError(
            f"Equivalence check FAILED: max abs diff = {max_diff:.2e} > {atol}"
        )
    return max_diff
