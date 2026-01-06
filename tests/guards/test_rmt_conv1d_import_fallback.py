from __future__ import annotations

import builtins

import torch.nn as nn

from invarlock.guards import rmt as R


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Module()
        self.block.attn = nn.Module()
        self.block.attn.c_attn = nn.Linear(2, 2, bias=False)


def test_rmt_conv1d_importerror_fallback_branches(monkeypatch) -> None:
    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "transformers.pytorch_utils":
            raise ImportError("blocked for test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    model = _TinyModel()
    stats = R.capture_baseline_mp_stats(model)
    assert stats  # should still collect linear layers without Conv1D

    guard = R.RMTGuard()
    modules = guard._get_linear_modules(model)
    assert modules and modules[0][0].endswith(".attn.c_attn")
