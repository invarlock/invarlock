"""Shared mock models for MI probe tests."""

from unittest.mock import Mock

import torch
import torch.nn as nn


class MockMLP(nn.Module):
    """Mock MLP module for testing."""

    def __init__(self, mlp_dim: int = 3072):
        super().__init__()
        self.c_fc = nn.Linear(768, mlp_dim)
        self.c_proj = nn.Linear(mlp_dim, 768)

    def forward(self, x):
        return self.c_proj(torch.relu(self.c_fc(x)))


class MockTransformerBlock(nn.Module):
    """Mock transformer block with MLP."""

    def __init__(self, mlp_dim: int = 3072):
        super().__init__()
        self.mlp = MockMLP(mlp_dim)
        self.ln_1 = nn.LayerNorm(768)
        self.ln_2 = nn.LayerNorm(768)

    def forward(self, x):
        return x + self.mlp(self.ln_2(x))


class MockGPT2Model(nn.Module):
    """Mock GPT-2 model for testing."""

    def __init__(self, n_layers: int = 2, mlp_dim: int = 3072):
        super().__init__()
        self.config = Mock()
        self.config.n_layer = n_layers
        self.transformer = Mock()
        self.transformer.h = nn.ModuleList(
            [MockTransformerBlock(mlp_dim) for _ in range(n_layers)]
        )
        self.wte = nn.Embedding(50257, 768)
        self.wpe = nn.Embedding(1024, 768)

    def forward(self, input_ids, **kwargs):
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, 50257, requires_grad=True)
        outputs = Mock()
        outputs.logits = logits
        return outputs


class MockAlternativeModel(nn.Module):
    """Alternative model structure without transformer attribute."""

    def __init__(self, n_layers: int = 2):
        super().__init__()
        self.config = Mock()
        self.config.n_layer = n_layers
        self.h = nn.ModuleList([MockTransformerBlock() for _ in range(n_layers)])

    def forward(self, input_ids, **kwargs):
        batch_size, seq_len = input_ids.shape
        return torch.randn(batch_size, seq_len, 50257, requires_grad=True)


__all__ = [
    "Mock",
    "MockAlternativeModel",
    "MockGPT2Model",
    "MockMLP",
    "MockTransformerBlock",
]
