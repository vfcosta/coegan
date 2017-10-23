import torch.nn as nn
from .layer import Layer
import numpy as np


# FIXME should not inherit from Layer anymore
class Dropout(Layer):
    """Represents a max pooling layer."""

    def __init__(self, p=None):
        super().__init__()
        self.p = p

    def setup(self):
        super().setup()
        if self.p is None:
            self.p = np.random.randint(0, 7) * 0.1

    def _create_phenotype(self, input_shape):
        return nn.Dropout(p=self.p)

    def apply_mutation(self):
        self.p = None  # reset p
        self.setup()  # restart p

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + str(self.p) + ')'
