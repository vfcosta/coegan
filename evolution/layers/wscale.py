import torch
from torch import nn
from util import tools
import math


class WScaleLayer(nn.Module):
    """
    Applies equalized learning rate to the preceding layer.
    """
    def __init__(self, incoming):
        super(WScaleLayer, self).__init__()
        self.incoming = incoming

        # self.scale = torch.sqrt(torch.mean(self.incoming.weight.data ** 2))
        # self.incoming.weight.data.copy_(self.incoming.weight.data / self.scale)

        fan = nn.init._calculate_correct_fan(self.incoming.weight, "fan_in")
        gain = nn.init.calculate_gain("relu")  # TODO: make calculate gain dependent of the activation
        self.scale = torch.tensor(gain / math.sqrt(fan))

        self.bias = None
        if self.incoming.bias is not None:
            self.bias = self.incoming.bias
            self.incoming.bias = None

    def forward(self, x):
        self.scale = tools.cuda(self.scale) if x.is_cuda else self.scale.cpu()
        x = x.mul(self.scale)
        if self.bias is not None:
            dims = [1, 1] if len(x.size()) == 4 else []
            x += self.bias.view(1, -1, *dims).expand_as(x)
        return x

    def __repr__(self):
        param_str = '(incoming = %s)' % self.incoming.__class__.__name__
        return self.__class__.__name__ + param_str
