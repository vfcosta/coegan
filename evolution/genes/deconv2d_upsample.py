import torch.nn as nn
from .conv2d import Conv2d


class Deconv2dUpsample(Conv2d):
    """Represents a convolution layer."""

    def __init__(self, out_channels=None, kernel_size=4, stride=1, activation_type="random", normalize=True, size=None):
        super().__init__(out_channels, kernel_size, stride, activation_type=activation_type, normalize=normalize)
        self._current_input_shape = None
        self.size = size
        self.padding = kernel_size//2

    def changed(self):
        return False

    def _create_phenotype(self, input_shape):
        upsample = nn.Upsample(size=self.size, align_corners=True)
        pad = nn.ZeroPad2d((1, 2, 1, 2))
        conv2d = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride)
        return nn.Sequential(upsample, pad, conv2d)

    def first_deconv(self):
        return not self.previous_layer or not isinstance(self.previous_layer, Deconv2dUpsample)

    def is_upsample(self):
        return self.out_channels < self.in_channels
