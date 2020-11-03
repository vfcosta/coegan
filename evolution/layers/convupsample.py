import torch.nn as nn


class ConvUpsample(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, output_size=None, padding=1, output_padding=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.output_size = output_size
        self.padding = padding
        self.output_padding = output_padding
        self.has_bias = bias
        self.module = nn.ConvTranspose2d(self.in_channels, self.out_channels, self.kernel_size, self.stride,
                                         padding=self.padding, output_padding=self.output_padding, bias=self.has_bias)

    @property
    def weight(self):
        return self.module.weight

    @property
    def bias(self):
        return self.module.bias

    @bias.setter
    def bias(self, bias):
        self.module.bias = bias

    def forward(self, input_data):
        return self.module(input_data, output_size=self.output_size)

    def __repr__(self):
        return self.__class__.__name__ + f"(in_channels={self.in_channels}, out_channels={self.out_channels}, " \
                                         f"kernel_size={self.kernel_size}, stride={self.stride}, " \
                                         f"output_size={self.output_size}, padding={self.padding}, " \
                                         f"output_padding={self.output_padding})"
