from types import SimpleNamespace

import torch
import torch.nn as nn

from invarlock.eval.metrics import (
    MetricsConfig,
    _collect_activations,
    _locate_transformer_blocks_enhanced,
    _perform_pre_eval_checks,
)


def test_collect_activations_hidden_states_short():
    class M(nn.Module):
        def forward(self, input_ids, output_hidden_states=False):
            B, T = input_ids.shape
            # Exactly 2 hidden states (edge branch: not > 2)
            hs = [torch.randn(B, T, 4) for _ in range(2)]
            return SimpleNamespace(hidden_states=hs)

    cfg = MetricsConfig(oracle_windows=1, progress_bars=False)
    dl = [{"input_ids": torch.ones(1, 4, dtype=torch.long)}]
    out = _collect_activations(M(), dl, cfg, torch.device("cpu"))
    assert isinstance(out, dict)


def test_locate_transformer_blocks_enhanced_base_model_path():
    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.h = [nn.Module()]
            self.h[0].attn = nn.Module()
            self.h[0].mlp = nn.Module()

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = Base()

    blocks = _locate_transformer_blocks_enhanced(Wrapper())
    assert isinstance(blocks, list) and len(blocks) == 1


def test_pre_eval_checks_context_length_try_except():
    # No config attributes -> context length check falls to except path
    class NoCfg(nn.Module):
        def forward(self, *a, **k):
            # Return model output compatible dict
            return SimpleNamespace(ok=True)

    dl = [{"input_ids": torch.ones(1, 2, dtype=torch.long)}]
    _perform_pre_eval_checks(NoCfg(), dl, torch.device("cpu"), MetricsConfig())
