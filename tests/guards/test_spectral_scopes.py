import torch
import torch.nn as nn

from invarlock.guards.spectral import SpectralGuard


class LinearModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))


def test_spectral_should_check_module_scopes():
    m = LinearModule()
    guard_attn = SpectralGuard(scope="attn")
    assert guard_attn._should_check_module("transformer.h.0.attn.c_proj", m)
    assert not guard_attn._should_check_module("transformer.h.0.mlp.c_fc", m)

    guard_ffn = SpectralGuard(scope="ffn")
    assert guard_ffn._should_check_module("transformer.h.0.mlp.c_fc", m)
    assert not guard_ffn._should_check_module("transformer.h.0.attn.c_proj", m)

    guard_all = SpectralGuard(scope="all")
    assert guard_all._should_check_module("anything", m)
