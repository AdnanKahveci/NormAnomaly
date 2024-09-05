# models/normalizing_flow.py

import torch
import torch.nn as nn
from .layers.affine_coupling import AffineCoupling
from .layers.batch_norm import BatchNorm
from .layers.invertible_conv import InvertibleConv1x1

import torch.nn as nn

class NormalizingFlowModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(NormalizingFlowModel, self).__init__()
        self.layers = nn.ModuleList()
        
        for _ in range(num_layers):
            # Katmanları oluştururken uygun giriş ve çıkış boyutlarını kullanın
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(hidden_dim, input_dim))  # input_dim ile eşleşmeli
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def reverse(self, x):
        for layer in reversed(self.layers):
            x = layer.reverse(x)
        return x
