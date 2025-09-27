import torch
import torch.nn as nn


class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class InstanceNorm1dONNX(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_features, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=2, keepdim=True)  # [B, C, 1]
        var = x.var(dim=2, keepdim=True, unbiased=False)  # [B, C, 1]

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized

    def __repr__(self):
        return f"InstanceNorm1dONNX({self.num_features}, eps={self.eps}, affine={self.affine})"


class PowerCurveProcessor(nn.Module):
    def __init__(self, hidden_dim=32, output_dim=1):
        super().__init__()

        self.power_branch = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=5, padding=2),
            InstanceNorm1dONNX(hidden_dim),
            nn.SiLU(),
            ResidualBlock(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        )

        self.fusion = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        )

    def forward(self, x):
        power_processed = self.power_branch(x[:, 0:1, :])  # [B, 1, T] -> [B, hidden_dim, T]
        output = self.fusion(power_processed)  # [B, output_dim, T]
        return output.transpose(1, 2)  # [B, T, output_dim]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False
        )
        self.norm1 = InstanceNorm1dONNX(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.norm2 = InstanceNorm1dONNX(out_channels)

        self.activation = nn.SiLU()

        self.skip = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                InstanceNorm1dONNX(out_channels)
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
            Transpose(1, 2),
            InstanceNorm1dONNX(hidden_dim),
            Transpose(1, 2),
            nn.SiLU()
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            Transpose(1, 2),
            InstanceNorm1dONNX(hidden_dim),
            Transpose(1, 2),
            nn.Sigmoid()
        )

        self.fusion_output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            Transpose(1, 2),
            InstanceNorm1dONNX(hidden_dim),
            Transpose(1, 2),
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
