from .layer import Layer
from ..config import config
from torch import nn


class Layer2D(Layer):
    """Represents an generic 2d layer in the evolving model."""

    def __init__(self, activation_params={}, activation_type="random", normalize=True, use_dropout=False):
        super().__init__(activation_params=activation_params, activation_type=activation_type, normalize=normalize, use_dropout=use_dropout)

    def is_linear(self):
        return False

    def _create_normalization(self):
        if config.gan.batch_normalization:
            return nn.BatchNorm2d(self.out_channels)
