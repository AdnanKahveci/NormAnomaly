# models/layers/affine_coupling.py

import torch
import torch.nn as nn

class AffineCoupling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AffineCoupling, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        s, t = self.net(x1).chunk(2, dim=1)
        return torch.cat([x1, x2 * torch.exp(s) + t], dim=1)
    
    def reverse(self, x):
        x1, x2 = x.chunk(2, dim=1)
        s, t = self.net(x1).chunk(2, dim=1)
        return torch.cat([x1, (x2 - t) * torch.exp(-s)], dim=1)
