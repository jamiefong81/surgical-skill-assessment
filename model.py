import torch
import torch.nn as nn
import torch.nn.functional as F

# Sub-cluster channel sizes within each 19-channel manipulator block,
# in the order produced by dataset.py: xyz(3), linvel(3), rotvel(3), R(9), gripper(1).
SUBCLUSTERS = [3, 3, 3, 9, 1]  # paper architecture
N_MANIPULATORS = 4
N_SUBCLUSTERS = len(SUBCLUSTERS)   # 5
BLOCK_SIZE = 19                    # channels per manipulator block


class SurgicalFCN(nn.Module):
    """Grouped FCN for surgical skill assessment.

    Implements the architecture from Ismail Fawaz et al. (arXiv:1908.07319).
    Expects input shape (batch, 76, timesteps).
    """

    def __init__(self, num_classes=3):
        super().__init__()

        # --- Layer 1: one Conv1d per sub-cluster (20 total) ---
        # Each sub-cluster gets its own set of 8 filters.  kernel_size=3,
        # padding=1 keeps temporal length unchanged.  Both from paper.
        self.layer1_convs = nn.ModuleList()
        for _ in range(N_MANIPULATORS):
            for sc_size in SUBCLUSTERS:
                self.layer1_convs.append(
                    nn.Conv1d(sc_size, 8, kernel_size=3, padding=1, bias=True)  # 8 filters: paper
                )
        # After concat: (batch, 20*8=160, timesteps)

        # --- Layer 2: one Conv1d per manipulator (4 total) ---
        # Each manipulator contributes 8 filters × 5 sub-clusters = 40 channels.
        self.layer2_convs = nn.ModuleList([
            nn.Conv1d(40, 16, kernel_size=3, padding=1, bias=True)  # 16 filters: paper
            for _ in range(N_MANIPULATORS)
        ])
        # After concat: (batch, 4*16=64, timesteps)

        # --- Layer 3: shared convolution across all manipulators ---
        self.layer3_conv = nn.Conv1d(64, 32, kernel_size=3, padding=1, bias=True)  # 32 filters: paper
        # Output: (batch, 32, timesteps)

        # --- Output layer ---
        self.fc = nn.Linear(32, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Glorot uniform init for all conv and linear weights; zero biases."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass.

        Args:
            x: tensor of shape (batch, 76, timesteps).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        # Layer 1: apply each sub-cluster conv to its own channel slice
        layer1_out = []
        conv_idx = 0
        for m in range(N_MANIPULATORS):
            ch_offset = 0
            for sc_size in SUBCLUSTERS:
                start = m * BLOCK_SIZE + ch_offset
                end = start + sc_size
                out = F.relu(self.layer1_convs[conv_idx](x[:, start:end, :]))
                layer1_out.append(out)
                ch_offset += sc_size
                conv_idx += 1
        x = torch.cat(layer1_out, dim=1)  # (batch, 160, timesteps)

        # Layer 2: apply each manipulator conv to its 40-channel slice
        layer2_out = []
        manip_channels = N_SUBCLUSTERS * 8  # 40
        for m in range(N_MANIPULATORS):
            start = m * manip_channels
            end = start + manip_channels
            out = F.relu(self.layer2_convs[m](x[:, start:end, :]))
            layer2_out.append(out)
        x = torch.cat(layer2_out, dim=1)  # (batch, 64, timesteps)

        # Layer 3: shared conv
        x = F.relu(self.layer3_conv(x))  # (batch, 32, timesteps)

        # Global average pooling over time
        x = torch.mean(x, dim=2)  # (batch, 32)

        return self.fc(x)  # (batch, num_classes)
