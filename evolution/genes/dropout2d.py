import torch.nn as nn
from .dropout import Dropout


class Dropout2d(Dropout):
    """Represents a max pooling layer."""

    def __init__(self, p=None):
        super().__init__(p)

    def _create_phenotype(self, input_shape):
        return nn.Dropout2d(p=self.p)

    def is_linear(self):
        return False
