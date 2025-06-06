import torch
import torch.nn as nn

from networks.layer.block.resnet_block import ResidualBasicBlock
from networks.layer.scaling.base import BaseDowmSampling, BaseUpSampling
from networks.layer.scaling.stride_conv import DownSampling, UpSampling


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
            ), block(
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
        padding_len = (-T) % self.divisible_factor
        x = nn.functional.pad(x, (0, 0, 0, padding_len))

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
