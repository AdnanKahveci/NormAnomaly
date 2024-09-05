# models/layers/invertible_conv.py

import torch
import torch.nn as nn

class InvertibleConv1x1(nn.Module):
    def __init__(self, input_dim):
        super(InvertibleConv1x1, self).__init__()
        self.conv = nn.Conv1d(input_dim, input_dim, kernel_size=1)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.orthogonal_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)
    
    def reverse(self, x):
        return self.conv(x)
