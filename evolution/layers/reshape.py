import torch.nn as nn


class Reshape(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input_data):
        return input_data.view(self.shape)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'shape=' + str(self.shape) + ')'
