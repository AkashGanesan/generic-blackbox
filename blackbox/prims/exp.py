import torch.nn as nn

class Exp(nn.Module):
    def __init__(self):
        super(Exp, self).__init__()

    def forward(self, x):
        return x.exp() * 0.5

