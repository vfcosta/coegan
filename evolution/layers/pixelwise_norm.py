import torch
from torch import nn


class PixelwiseNorm(nn.Module):

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(eps = %s)' % (self.eps)
