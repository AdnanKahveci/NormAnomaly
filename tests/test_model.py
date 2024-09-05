import pytest
import torch
import sys
import os

# Proje kök dizinini Python yoluna ekleyin
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from models.normalizing_flow import NormalizingFlowModel
from models.layers.affine_coupling import AffineCoupling
from models.layers.batch_norm import BatchNorm
from models.layers.invertible_conv import InvertibleConv1x1

def test_normalizing_flow():
    model = NormalizingFlowModel(input_dim=2, hidden_dim=4, num_layers=2)
    x = torch.randn(1, 2)  # Linear katmanlar için uygun giriş boyutu (batch_size, input_dim)
    output = model(x)
    assert output.shape == x.shape, "Model çıktısı girişle aynı boyutta olmalıdır"

def test_affine_coupling():
    layer = AffineCoupling(input_dim=2, hidden_dim=4)
    x = torch.randn(1, 2)
    output = layer(x)
    assert output.shape == x.shape, "Affine Coupling çıktısı girişle aynı boyutta olmalıdır"

def test_batch_norm():
    layer = BatchNorm(num_features=2)
    x = torch.randn(4, 2)  # BatchNorm için uygun giriş boyutu
    output = layer(x)
    assert output.shape == x.shape, "BatchNorm çıktısı girişle aynı boyutta olmalıdır"

def test_invertible_conv():
    layer = InvertibleConv1x1(input_dim=2)
    x = torch.randn(1, 2, 10)  # Conv1d giriş boyutu [batch_size, in_channels, sequence_length]
    output = layer(x)
    assert output.shape == x.shape, "InvertibleConv1x1 çıktısı girişle aynı boyutta olmalıdır"
