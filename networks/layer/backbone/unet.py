import math

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from networks.layer.block.resnet_block import ResidualBasicBlock
from networks.layer.scaling.base import BaseDowmSampling, BaseUpSampling
from networks.layer.scaling.stride_conv import DownSampling, UpSampling


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerBottleneck(nn.Module):
    def __init__(self, d_model, nhead=2, num_layers=2, dim_feedforward=64, dropout=0.1):
        super().__init__()
        self.norm_in = nn.LayerNorm(d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm_in(x)
        x = x + self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.norm_out(self.dropout(x))


class UNetBackbone(nn.Module):
    def __init__(
            self,
            input_dims: int,
            output_dims: int,
            hidden_dims: int,
            block,
            down_sampling,
            up_sampling,
            down_sampling_factor: int = 2,
            down_sampling_times: int = 5,
            channels_scaleup_factor: int = 2,
            use_trans: bool = False,
            transformer_nhead: int = 2,
            transformer_dim_feedforward: int = 512,
            transformer_num_layers: int = 2,
            transformer_dropout: float = 0.1,
    ):
        """_summary_

        Args:
            input_dims (int):
            output_dims (int):
            hidden_dims (int):
            block (nn.Module): shape: (B, T, C) -> shape: (B, T, C)
            down_sampling (nn.Module): shape: (B, T, C) -> shape: (B, T/down_sampling_factor, C*2)
            up_sampling (nn.Module): shape: (B, T, C) -> shape: (B, T*down_sampling_factor, C/2)
        """
        super(UNetBackbone, self).__init__()
        assert issubclass(block, nn.Module)
        assert issubclass(down_sampling, BaseDowmSampling)
        assert issubclass(up_sampling, BaseUpSampling)

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims

        self.divisible_factor = down_sampling_factor ** down_sampling_times

        self.encoders = nn.ModuleList()
        self.encoders.append(block(input_dims, hidden_dims))
        for i in range(down_sampling_times - 1):
            i += 1
            self.encoders.append(
                nn.Sequential(
                    down_sampling(
                        int(channels_scaleup_factor ** (i - 1)) * hidden_dims,
                        int(channels_scaleup_factor ** i) * hidden_dims,
                        down_sampling_factor,
                    ),
                    block(
                        int(channels_scaleup_factor ** i) * hidden_dims,
                        int(channels_scaleup_factor ** i) * hidden_dims,
                    ),
                )
            )

        self.bottle_neck = nn.Sequential(
            down_sampling(
                int(channels_scaleup_factor ** (down_sampling_times - 1)) * hidden_dims,
                int(channels_scaleup_factor ** down_sampling_times) * hidden_dims,
                down_sampling_factor,
            ),
            TransformerBottleneck(
                d_model=int(channels_scaleup_factor ** down_sampling_times) * hidden_dims,
                nhead=transformer_nhead,
                num_layers=transformer_num_layers,
                dim_feedforward=transformer_dim_feedforward,
                dropout=transformer_dropout
            ) if use_trans else
            block(
                int(channels_scaleup_factor ** down_sampling_times) * hidden_dims,
                int(channels_scaleup_factor ** down_sampling_times) * hidden_dims,
            ),
            up_sampling(
                int(channels_scaleup_factor ** down_sampling_times) * hidden_dims,
                int(channels_scaleup_factor ** (down_sampling_times - 1)) * hidden_dims,
                down_sampling_factor,
            ),
        )

        self.decoders = nn.ModuleList()
        for i in range(down_sampling_times - 1):
            i += 1
            self.decoders.append(
                nn.Sequential(
                    block(
                        int(channels_scaleup_factor ** (down_sampling_times - i))
                        * hidden_dims,
                        int(channels_scaleup_factor ** (down_sampling_times - i))
                        * hidden_dims,
                    ),
                    up_sampling(
                        int(channels_scaleup_factor ** (down_sampling_times - i))
                        * hidden_dims,
                        int(channels_scaleup_factor ** (down_sampling_times - i - 1))
                        * hidden_dims,
                        down_sampling_factor,
                    ),
                )
            )
        self.decoders.append(block(hidden_dims, output_dims))

    def forward(self,
                x,  # [B, T, C]
                ):
        T = x.shape[1]
        padding_len = T % self.divisible_factor
        if padding_len != 0:
            x = nn.functional.pad(x, (0, 0, 0, self.divisible_factor - padding_len))

        h = [x]
        for encoder in self.encoders:
            h.append(encoder(h[-1]))

        h_ = [self.bottle_neck(h[-1])]
        for i, decoder in enumerate(self.decoders):
            h_.append(decoder(h_[-1] + h[-1 - i]))

        out = h_[-1]
        out = out[:, :T, :]

        return out


if __name__ == "__main__":
    # pass
    model = UNetBackbone(1, 2, 64, ResidualBasicBlock, DownSampling, UpSampling)
    print(model)
    input_feature = torch.randn(16, 320, 1)
    output = model(input_feature)
    print(input_feature.shape, output.shape)
