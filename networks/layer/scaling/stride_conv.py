import torch.nn as nn

from networks.layer.scaling.base import BaseDowmSampling, BaseUpSampling


class DownSampling(BaseDowmSampling):
    def __init__(self, input_dims: int, output_dims: int, down_sampling_factor: int = 2):
        super(DownSampling, self).__init__(
            input_dims, output_dims, down_sampling_factor=2
        )

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.down_sampling_factor = down_sampling_factor

        self.conv = nn.Conv1d(
            self.input_dims,
            self.output_dims,
            kernel_size=down_sampling_factor,
            stride=down_sampling_factor,
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        T = x.shape[-1]
        padding_len = (-T) % self.down_sampling_factor
        x = nn.functional.pad(x, (0, padding_len))
        return self.conv(x).transpose(1, 2)


class UpSampling(BaseUpSampling):
    def __init__(self, input_dims: int, output_dims: int, up_sampling_factor: int = 2):
        super(UpSampling, self).__init__(input_dims, output_dims, up_sampling_factor=2)

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.up_sampling_factor = up_sampling_factor

        self.conv = nn.ConvTranspose1d(
            self.input_dims,
            self.output_dims,
            kernel_size=up_sampling_factor,
            stride=up_sampling_factor,
        )

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)
