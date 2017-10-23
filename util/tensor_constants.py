import torch
from util import tools

ONE = tools.cuda(torch.tensor(1.0))
MONE = tools.cuda(torch.tensor(-1.0))
