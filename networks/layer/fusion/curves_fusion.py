import torch
import torch.nn as nn


class PowerCurveProcessor(nn.Module):
    def __init__(self, hidden_dim=32, output_dim=1):
        super().__init__()

        self.power_branch = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=5, padding=2),
            nn.InstanceNorm1d(hidden_dim),
            nn.SiLU(),
            ResidualBlock(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        )

        self.fusion = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        )

    def forward(self,
                x,  # [B, C, T]
                ):
        power_processed = self.power_branch(x[:, 0:1, :])  # [B, hidden_dim, T]
        output = self.fusion(power_processed)  # [B, output_dim, T]
        return output.transpose(1, 2)  # [B, T, output_dim]


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


class PowerCurveEdgeFusion(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim

        self.curve_processor = PowerCurveProcessor(
            hidden_dim=hidden_dim,
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

    def forward(self,
                x,  # [B, T, hidden_dims]
                curves,  # [B, 1, T]
                ):
        curve_features = self.curve_processor(curves)  # [B, T, hidden_dim]
        feature_proj = self.feature_proj(x)  # [B, T, hidden_dim]

        gate_value = self.gate(torch.cat([feature_proj, curve_features], dim=-1))
        gated_features = gate_value * feature_proj + (1 - gate_value) * curve_features

        fused_features = torch.cat([gated_features, curve_features], dim=-1)
        edge_enhancement = self.fusion_output(fused_features)
        return edge_enhancement  # [B, T, 1]
