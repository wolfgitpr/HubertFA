import torch
import torch.nn as nn


class ChannelAttention1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return out.view(b, c, 1)


class PowerCurveEdgeFusion(nn.Module):
    def __init__(self, feature_dim, power_dim=1, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim

        self.power_processor = nn.Sequential(
            nn.Conv1d(power_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

        self.attention = ChannelAttention1D(1, reduction=1)

        self.gate = nn.Sequential(
            nn.Linear(feature_dim + 1, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, power_curve):
        power_features = self.power_processor(power_curve.transpose(1, 2))  # [B, hidden_dim, T]
        power_features = self.attention(power_features) * power_features
        power_features = power_features.transpose(1, 2)  # [B, T, 1]

        gate_input = torch.cat([x, power_features], dim=-1)
        gate_value = self.gate(gate_input)  # [B, T, 1]

        edge_enhancement = gate_value * power_features

        return edge_enhancement
