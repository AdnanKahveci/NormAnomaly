# models/layers/batch_norm.py

import torch
import torch.nn as nn

class BatchNorm(nn.Module):
    def __init__(self, num_features):
        super(BatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)
    
    def forward(self, x):
        return self.bn(x)
    
    def reverse(self, x):
        return self.bn(x)
