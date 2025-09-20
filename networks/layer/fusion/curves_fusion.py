import torch
import torch.nn as nn


class DualCurveProcessor(nn.Module):
    def __init__(self, hidden_dim=32, output_dim=1):
        super().__init__()

        # Power branch with residual connections and normalization
        self.power_branch = nn.Sequential(
            nn.Conv1d(1, hidden_dim // 2, kernel_size=5, padding=2),
            nn.InstanceNorm1d(hidden_dim // 2),
            nn.SiLU(),
            ResidualBlock(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1)
        )

        # Pitch branch with residual connections and normalization
        self.pitch_branch = nn.Sequential(
            nn.Conv1d(1, hidden_dim // 2, kernel_size=7, padding=3),
            nn.InstanceNorm1d(hidden_dim // 2),
            nn.SiLU(),
            ResidualBlock(hidden_dim // 2, hidden_dim // 2, kernel_size=5, padding=2)
        )

        # Fusion module with residual connections and normalization
        self.fusion = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        )

        # Dynamic weight estimation
        self.dynamic_weight = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(output_dim, output_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(output_dim // 2, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):  # [B, T, 2]
        x = x.transpose(1, 2)  # [B, 2, T]

        # Process each curve separately
        power_processed = self.power_branch(x[:, 0:1, :])  # [B, hidden_dim//2, T]
        pitch_processed = self.pitch_branch(x[:, 1:2, :])  # [B, hidden_dim//2, T]

        # Combine and fuse
        combined = torch.cat([power_processed, pitch_processed], dim=1)  # [B, hidden_dim, T]
        output = self.fusion(combined)  # [B, output_dim, T]

        # Apply dynamic weighting
        weights = self.dynamic_weight(output)  # [B, 2, 1]
        weighted_output = output * weights[:, 0:1, :]

        return weighted_output.transpose(1, 2)  # [B, T, output_dim]


class ResidualBlock(nn.Module):
    """Residual block with instance normalization and optional down/up sampling"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False
        )
        self.norm1 = nn.InstanceNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.norm2 = nn.InstanceNorm1d(out_channels)

        self.activation = nn.SiLU()

        # Skip connection
        self.skip = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += identity
        out = self.activation(out)

        return out


class DualCurveEdgeFusion(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim

        self.curve_processor = DualCurveProcessor(
            hidden_dim=hidden_dim // 2,
            output_dim=hidden_dim
        )

        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.InstanceNorm1d(hidden_dim),
            nn.SiLU()
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.InstanceNorm1d(hidden_dim),
            nn.Sigmoid()
        )

        self.fusion_output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.InstanceNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, curves):
        curve_features = self.curve_processor(curves)  # [B, T, hidden_dim]
        feature_proj = self.feature_proj(x)  # [B, T, hidden_dim]

        gate_value = self.gate(torch.cat([feature_proj, curve_features], dim=-1))
        gated_features = gate_value * feature_proj + (1 - gate_value) * curve_features

        fused_features = torch.cat([gated_features, curve_features], dim=-1)
        edge_enhancement = self.fusion_output(fused_features)

        return edge_enhancement  # [B, T, 1]
