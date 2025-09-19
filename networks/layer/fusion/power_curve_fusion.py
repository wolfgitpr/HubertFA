import torch
import torch.nn as nn


class PowerProcessor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        self.activation = nn.SiLU()

        attention_mid_dim = max(1, output_dim // 4)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(output_dim, attention_mid_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(attention_mid_dim, output_dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self,
                x  # [B, T, 1]
                ):
        x = x.transpose(1, 2)  # [B, 1, T]
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.conv3(x)

        attn = self.attention(x)
        x = x * attn

        return x.transpose(1, 2)  # [B, T, output_dim]


class PowerCurveEdgeFusion(nn.Module):
    def __init__(self, feature_dim, power_dim=1, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim

        self.power_processor = PowerProcessor(
            input_dim=power_dim,
            hidden_dim=32,
            output_dim=hidden_dim
        )

        self.feature_proj = nn.Linear(feature_dim, hidden_dim)

        self.edge_output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x,  # [B, T, feature_dim]
                power_curve  # [B, T, 1]
                ):
        power_features = self.power_processor(power_curve)  # [B, T, hidden_dim]
        feature_proj = self.feature_proj(x)  # [B, T, hidden_dim]

        fused_features = torch.cat([feature_proj, power_features], dim=-1)  # [B, T, hidden_dim*2]
        edge_enhancement = self.edge_output(fused_features)  # [B, T, 1]

        return self.dropout(edge_enhancement)  # [B, T, 1]
