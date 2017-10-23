import torch
from torch import nn


class FadeIn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.alpha = 0.0

    def update_alpha(self, delta):
        self.alpha = self.alpha + delta
        self.alpha = max(0.0, min(self.alpha, 1.0))

    # input : [x_low, x_high] from ConcatTable()
    def forward(self, x):
        return torch.add(x[0].mul(1.0-self.alpha), x[1].mul(self.alpha))
