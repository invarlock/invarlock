from __future__ import annotations

import builtins

import pytest
import torch
import torch.nn as nn

import invarlock.guards.rmt as runtime_rmt
import invarlock.guards.rmt_analysis as rmt_analysis
import invarlock.guards.rmt_detection as rmt_detection


def test_runtime_activation_module_and_edge_risk_guardrails(monkeypatch) -> None:
    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "transformers.pytorch_utils":
            raise ImportError("blocked for test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(8, 4)
            self.norm = nn.LayerNorm(4)
            self.attn = nn.Linear(4, 4, bias=False)

    guard = runtime_rmt.RMTGuard()
    modules = guard._get_activation_modules(Model())
    names = [name for name, _module in modules]
    assert "embed" in names
    assert "norm" in names
    assert "attn" in names

    original_vector_norm = torch.linalg.vector_norm
    monkeypatch.setattr(
        torch.linalg,
        "vector_norm",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("norm fail")),
    )
    assert guard._activation_edge_risk(torch.randn(3, 2)) is None

    monkeypatch.setattr(torch.linalg, "vector_norm", original_vector_norm)
    original_sqrt = torch.sqrt
    monkeypatch.setattr(torch, "sqrt", lambda *_a, **_k: torch.tensor(float("nan")))
    guard = runtime_rmt.RMTGuard()
    assert guard._activation_edge_risk(torch.randn(3, 2)) is None

    monkeypatch.setattr(torch, "sqrt", original_sqrt)
    original_mp_bulk_edge = rmt_analysis.mp_bulk_edge
    monkeypatch.setattr(rmt_analysis, "mp_bulk_edge", lambda *_a, **_k: float("nan"))
    assert guard._activation_edge_risk(torch.randn(3, 2)) is None

    monkeypatch.setattr(rmt_analysis, "mp_bulk_edge", original_mp_bulk_edge)
    guard.estimator = {"iters": "bad", "init": "bogus"}
    assert guard._activation_edge_risk(torch.randn(3, 2)) is not None

    guard.estimator = {"iters": 1, "init": "e0"}
    assert guard._activation_edge_risk(torch.randn(3, 2)) is not None


def test_runtime_activation_collection_handles_bad_hooks() -> None:
    guard = runtime_rmt.RMTGuard()
    assert guard._compute_activation_edge_risk(nn.Linear(2, 2), []) is None
    assert guard._compute_activation_edge_risk(nn.Module(), [object()]) is None

    class RaisingHookLinear(nn.Linear):
        def register_forward_hook(self, hook):  # noqa: ANN001
            raise RuntimeError("cannot hook")

    class RaisingHookModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = RaisingHookLinear(2, 2, bias=False)

        def forward(self, input_ids, attention_mask=None):  # noqa: ANN001
            return self.attn(input_ids.float())

    assert (
        guard._compute_activation_edge_risk(
            RaisingHookModel(),
            [{"input_ids": None}, {"input_ids": torch.ones(1, 2)}],
        )
        is None
    )

    class BadHandle:
        def remove(self) -> None:
            raise RuntimeError("cannot remove")

    class BadHandleLinear(nn.Linear):
        def register_forward_hook(self, hook):  # noqa: ANN001
            super().register_forward_hook(hook)
            return BadHandle()

    class BadHandleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = BadHandleLinear(2, 2, bias=False)

        def forward(self, input_ids, attention_mask=None):  # noqa: ANN001
            return self.attn(input_ids.float())

    result = guard._compute_activation_edge_risk(
        BadHandleModel(), [{"input_ids": torch.ones(1, 2)}]
    )
    assert result is not None
    assert result["analysis_source"] == "activations_edge_risk"
    assert result["batches_used"] == 1


def test_runtime_detection_logs_correction_failure(monkeypatch) -> None:
    guard = runtime_rmt.RMTGuard(correct=True)
    guard.baseline_sigmas = {"layer": 1.0}
    guard.baseline_mp_stats = {"layer": {"sigma_base": 1.0, "mp_bulk_edge_base": 1.0}}
    layer = nn.Linear(2, 2, bias=False)

    monkeypatch.setattr(guard, "_get_linear_modules", lambda _model: [("layer", layer)])
    monkeypatch.setattr(
        runtime_rmt.rmt_analysis,
        "layer_svd_stats",
        lambda *_a, **_k: {
            "sigma_min": 0.0,
            "sigma_max": 10.0,
            "worst_ratio": 10.0,
            "worst_details": {"name": "weight"},
        },
    )
    monkeypatch.setattr(
        rmt_detection,
        "_apply_rmt_correction",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = guard._apply_rmt_detection_and_correction(nn.Identity())

    assert result["has_outliers"] is True
    assert any(
        event["kind"] == "rmt_correct_failed" for event in guard.diagnostic_records
    )


def test_prepare_rejects_legacy_epsilon_parameter() -> None:
    from invarlock.core.exceptions import ValidationError

    guard = runtime_rmt.RMTGuard()

    with pytest.raises(ValidationError):
        guard.prepare(nn.Linear(2, 2, bias=False), policy={"epsilon": 0.1})
