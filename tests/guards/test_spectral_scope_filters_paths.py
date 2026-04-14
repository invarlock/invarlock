import torch

from invarlock.guards.spectral import SpectralGuard


def test_spectral_scope_filters_attn_and_ffn_aliases():
    guard_attn = SpectralGuard(scope="attn")
    guard_ffn = SpectralGuard(scope="ffn")
    lin = torch.nn.Linear(4, 4)
    assert guard_attn._should_check_module("layer.self_attn.out_proj", lin) is True
    assert guard_ffn._should_check_module("layer.mlp.fc_in", lin) is True
