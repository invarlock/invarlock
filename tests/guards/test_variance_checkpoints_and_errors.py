import torch
import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        blk = nn.Module()
        blk.attn = nn.Module()
        blk.attn.c_proj = nn.Linear(2, 2, bias=False)
        blk.mlp = nn.Module()
        blk.mlp.c_proj = nn.Linear(2, 2, bias=False)
        self.transformer.h = nn.ModuleList([blk])

    def forward(self, x):
        return self.transformer.h[0].mlp.c_proj(self.transformer.h[0].attn.c_proj(x))


def test_checkpoint_pop_and_commit_empty_paths():
    g = VarianceGuard(policy={"scope": "both", "min_gain": 0.0})
    # Pop with empty stack
    assert g._pop_checkpoint(Tiny()) is False
    # Commit with empty stack does not error
    g._commit_checkpoint()


def test_disable_uses_checkpoint_restore():
    model = Tiny()
    g = VarianceGuard(policy={"scope": "both", "min_gain": 0.0})
    targets = g._resolve_target_modules(model, adapter=None)
    g._prepared = True
    g._target_modules = targets
    # Push checkpoint and mark enabled to exercise checkpoint restore in disable()
    g._push_checkpoint(model)
    g._enabled = True
    assert g.disable(model) is True
    assert g._enabled is False


def test_enable_idempotent_and_monitor_only_paths():
    model = Tiny()
    g = VarianceGuard(policy={"scope": "both", "min_gain": 0.0})
    targets = g._resolve_target_modules(model, adapter=None)
    g._prepared = True
    g._target_modules = targets
    # Idempotent enable
    g._enabled = True
    g._scales = {next(iter(targets.keys())): 0.95}
    assert g.enable(model) is True
    # Monitor-only skips enable
    g._enabled = False
    g._monitor_only = True
    g._scales = {next(iter(targets.keys())): 0.95}
    assert g.enable(model) is False
    assert g._enabled is False


def test_prepare_failed_path_returns_error(monkeypatch):
    model = Tiny()
    g = VarianceGuard(policy={"scope": "both", "min_gain": 0.0})
    # Force target resolution to throw
    monkeypatch.setattr(
        g,
        "_resolve_target_modules",
        lambda m, a: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    res = g.prepare(model, adapter=None, calib=None, policy=None)
    assert res.get("ready") is False and "error" in res


class ErrorWeight:
    # Weight-like: supports data clone for checkpoint but fails on device access
    def __init__(self):
        self._t = torch.zeros(1)
        self.dtype = self._t.dtype
        self.data = self._t

    @property
    def device(self):
        raise RuntimeError("bad device")


class BadModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = ErrorWeight()


def test_scale_apply_error_and_revert_error_paths():
    model = Tiny()
    g = VarianceGuard(policy={"scope": "both", "min_gain": 0.0})
    # Inject a bad target module
    g._prepared = True
    g._target_modules = {"transformer.h.0.mlp.c_proj": BadModule()}
    g._scales = {"transformer.h.0.mlp.c_proj": 1.1}
    # Enable should fail with apply error and no modules applied
    assert g.enable(model) is False
    # Mark enabled and attempt disable to exercise revert error path
    g._enabled = True
    assert g.disable(model) is False
