from types import SimpleNamespace

import torch
import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class TinyAttn(nn.Module):
    def __init__(self, d=4, with_proj=True, unsupported=False):
        super().__init__()
        if with_proj:
            if unsupported:
                # Weight with wrong dimensionality to trigger unsupported_type
                self.c_proj = SimpleNamespace(weight=torch.nn.Parameter(torch.randn(8)))
            else:
                self.c_proj = nn.Linear(d, d, bias=True)


class TinyMLP(nn.Module):
    def __init__(self, d=4, with_proj=True):
        super().__init__()
        if with_proj:
            self.c_proj = nn.Linear(d, d, bias=True)


class TinyBlock(nn.Module):
    def __init__(self, d=4, attn_conf=None, mlp_conf=None):
        super().__init__()
        attn_conf = attn_conf or {}
        mlp_conf = mlp_conf or {}
        self.attn = TinyAttn(d=d, **attn_conf) if attn_conf is not None else None
        self.mlp = TinyMLP(d=d, **mlp_conf) if mlp_conf is not None else None


class TinyModel(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        # GPT-2 style container
        self.transformer = SimpleNamespace(h=nn.ModuleList(blocks))

    def forward(self, x, labels=None):  # pragma: no cover - not used here
        return x


def _make_guard(**overrides) -> VarianceGuard:
    policy = {
        "min_gain": 0.01,
        "max_calib": 50,
        "scope": "both",
        "clamp": (0.5, 2.0),
        "deadband": 0.0,
        "seed": 123,
        "calibration": {"windows": 4, "min_coverage": 2, "seed": 123},
        # Tap both branches
        "tap": ["transformer.h.*.mlp.c_proj", "transformer.h.*.attn.c_proj"],
    }
    policy.update(overrides)
    return VarianceGuard(policy=policy)


def test_resolve_targets_rejection_buckets(monkeypatch):
    # Block0: attn missing -> missing_module; mlp present
    # Block1: attn present but tap excludes? We include taps; use unsupported weight
    blocks = [
        TinyBlock(attn_conf={"with_proj": False}, mlp_conf={"with_proj": True}),
        TinyBlock(
            attn_conf={"with_proj": True, "unsupported": True},
            mlp_conf={"with_proj": False},
        ),
    ]
    model = TinyModel(blocks)

    guard = _make_guard()
    result = guard.prepare(model, adapter=None, calib=[], policy=None)
    assert result["ready"] is True

    tr = guard._stats.get("target_resolution", {})
    assert isinstance(tr, dict)
    rejected = tr.get("rejected", {})
    # Expect at least missing_module and unsupported_type buckets
    assert rejected.get("missing_module", {}).get("count", 0) >= 1
    assert rejected.get("unsupported_type", {}).get("count", 0) >= 1


def test_resolve_targets_tap_mismatch_and_adapter_error(monkeypatch):
    # MLP present but tap excludes it -> tap_mismatch; fallback adapter raises
    blocks = [TinyBlock(attn_conf={"with_proj": False}, mlp_conf={"with_proj": True})]
    model = TinyModel(blocks)

    guard = _make_guard(tap=["transformer.h.*.attn.c_proj"])  # exclude mlp

    class BadAdapter:
        def get_layer_modules(self, model, i):  # noqa: D401
            raise RuntimeError("boom")

    _ = guard.prepare(model, adapter=BadAdapter(), calib=[], policy=None)
    rejected = guard._stats.get("target_resolution", {}).get("rejected", {})
    # tap_mismatch bucket
    assert rejected.get("tap_mismatch", {}).get("count", 0) >= 1
    # adapter_error bucket key starts with adapter_error:
    assert any(k.startswith("adapter_error:") for k in rejected.keys())


def test_enable_disable_checkpoint_restores_weights():
    blocks = [TinyBlock(attn_conf={"with_proj": True}, mlp_conf={"with_proj": True})]
    model = TinyModel(blocks)
    mlp_proj = model.transformer.h[0].mlp.c_proj

    guard = _make_guard()
    # Seed target_modules and scales manually to exercise enable/disable
    name = "transformer.h.0.mlp.c_proj"
    guard._target_modules = {name: mlp_proj}
    guard._scales = {name: 0.9}
    guard._prepared = True

    orig = mlp_proj.weight.detach().clone()
    assert guard.enable(model) is True
    assert not torch.allclose(mlp_proj.weight, orig)
    # Disable should restore exact weights
    assert guard.disable(model) is True
    assert torch.allclose(mlp_proj.weight, orig)


def test_enable_fails_on_quantized_weights_and_rolls_back():
    # Create a fake target with int8 weights
    class FakeModule(nn.Module):
        def __init__(self):
            super().__init__()
            # Use plain tensor (not Parameter) to avoid grad constraints
            self.weight = torch.randint(-2, 2, (4, 4), dtype=torch.int8)

    model = TinyModel([TinyBlock(attn_conf={"with_proj": True})])
    fake = FakeModule()
    guard = _make_guard()
    guard._prepared = True
    guard._target_modules = {"transformer.h.0.attn.c_proj": fake}
    guard._scales = {"transformer.h.0.attn.c_proj": 0.9}

    ok = guard.enable(model)
    # No modules scaled; checkpoint popped and enable returns False
    assert ok is False
    assert guard._enabled is False


class _ConstLossTinyModel(TinyModel):
    def __init__(self, blocks, loss_value=2.0):
        super().__init__(blocks)
        self._loss_value = float(loss_value)

    def forward(self, input_ids, labels=None):  # type: ignore[override]
        class Out:
            def __init__(self, loss):
                self.loss = torch.tensor(float(loss))

        return Out(loss=self._loss_value)


def test_finalize_records_ab_provenance_and_point_estimates(monkeypatch):
    # Bypass prepare complexity: directly seed minimal stats for finalize
    guard = _make_guard()

    class ParamTinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList(
                [TinyBlock(attn_conf={"with_proj": True}, mlp_conf={"with_proj": True})]
            )

        def forward(self, input_ids, labels=None):  # noqa: D401
            class Out:
                def __init__(self, loss):
                    self.loss = torch.tensor(float(loss))

            return Out(loss=2.0)

    model = ParamTinyModel()

    guard._stats.setdefault("ab_provenance", {})
    guard._stats.setdefault(
        "ab_point_estimates", {"tag": "test", "ppl_no_ve": 10.0, "ppl_with_ve": 10.0}
    )
    guard._scales = {}
    guard._target_modules = {}
    guard._prepared = True

    out = guard.finalize(model)
    assert isinstance(out, dict) and out.get("metrics")
    metrics = out["metrics"]
    assert "ab_provenance" in metrics
    assert "ab_point_estimates" in metrics
    # no_scaling_required path implies ve_enabled False
    assert metrics.get("ve_enabled") in (False, None)


def test_no_scaling_required_path_sets_point_estimates(monkeypatch):
    # Deterministically hit the no_scales path by calling calibration pass directly
    guard = _make_guard()

    class ParamTinyModel2(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList(
                [TinyBlock(attn_conf={"with_proj": True}, mlp_conf={"with_proj": True})]
            )

        def forward(self, input_ids, labels=None):  # noqa: D401
            class Out:
                def __init__(self, loss):
                    self.loss = torch.tensor(2.0)

            return Out(loss=2.0)

    model = ParamTinyModel2()

    # Seed targets (MLP only is enough) and mark prepared
    mlp_proj = model.transformer.h[0].mlp.c_proj
    guard._target_modules = {"transformer.h.0.mlp.c_proj": mlp_proj}
    guard._prepared = True
    guard._scales = {}  # no scales → should record no_scales path

    # Provide batches and run calibration pass
    calib = [torch.arange(4), torch.arange(4), torch.arange(4), torch.arange(4)]
    guard._calibration_stats = {
        "requested": 0,
        "coverage": 0,
        "min_coverage": 2,
        "seed": 123,
        "status": "uninitialized",
    }
    guard._evaluate_calibration_pass(
        model, calibration_batches=calib, min_coverage=2, calib_seed=123, tag="pre_edit"
    )

    pg = guard._stats.get("predictive_gate", {})
    assert pg.get("reason") in {"no_scales", "insufficient_coverage"}
    ab = guard._stats.get("ab_point_estimates", {})
    assert "ppl_no_ve" in ab

    final = guard.finalize(model)
    assert "ab_provenance" in final.get("metrics", {})


def test_validate_returns_monitor_warning_action():
    guard = _make_guard()
    guard._prepared = True
    guard._monitor_only = True
    res = guard.validate(model=None, adapter=None, context={})
    assert res["action"] in {"warn", "continue"}
