import torch.nn as nn
import torch.nn.functional as F


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
            channels_scaleup_factor: float = 2.0,
            dropout: float = 0.1,
    ):
        """_summary_

        Args:
            input_dims (int):
            output_dims (int):
            hidden_dims (int):
            block (nn.Module): shape: (B, C, T) -> shape: (B, C, T)
            down_sampling (nn.Module): shape: (B, C, T) -> shape: (B, C*2, T/down_sampling_factor)
            up_sampling (nn.Module): shape: (B, C, T) -> shape: (B, C/2, T*down_sampling_factor)
        """
        super(UNetBackbone, self).__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims

        self.divisible_factor = down_sampling_factor ** down_sampling_times

        self.encoders = nn.ModuleList()
        self.encoders.append(block(input_dims, hidden_dims, dropout=dropout))
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
                        dropout=dropout,
                    ),
                )
            )

        self.bottle_neck = nn.Sequential(
            down_sampling(
                int(channels_scaleup_factor ** (down_sampling_times - 1)) * hidden_dims,
                int(channels_scaleup_factor ** down_sampling_times) * hidden_dims,
                down_sampling_factor,
            ),
            block(
                int(channels_scaleup_factor ** down_sampling_times) * hidden_dims,
                int(channels_scaleup_factor ** down_sampling_times) * hidden_dims,
                dropout=dropout,
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
                        dropout=dropout,
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
                x,  # [B, C, T]
                ):
        B, C, T = x.shape
        padding_len = (-T) % self.divisible_factor
        x = F.pad(x, (0, padding_len))

        current = x
        encoder_outputs = []

        for encoder in self.encoders:
            current = encoder(current)
            encoder_outputs.append(current)

        bottleneck_out = self.bottle_neck(encoder_outputs[-1])

        current = bottleneck_out
        for i, decoder in enumerate(self.decoders):
            skip_connection = encoder_outputs[-1 - i]
            if current.shape != skip_connection.shape:
                skip_connection = F.interpolate(skip_connection, size=current.shape[2:])
            current = decoder(current + skip_connection)

        return current[:, :, :T]
