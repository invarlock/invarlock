from __future__ import annotations

import builtins
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import invarlock.guards.rmt as runtime_rmt


def test_finalize_returns_plain_dict_when_guardoutcome_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(runtime_rmt, "HAS_GUARD_OUTCOME", False)
    monkeypatch.setattr(runtime_rmt, "GuardOutcome", dict, raising=False)

    result = runtime_rmt.RMTGuard().finalize(nn.Linear(2, 2, bias=False), adapter=None)

    assert result["passed"] is False
    assert result["metrics"]["prepared"] is False
    assert result["errors"] == ["RMT guard not properly prepared"]


def test_finalize_activation_required_failure_returns_plain_dict(
    monkeypatch,
) -> None:
    monkeypatch.setattr(runtime_rmt, "HAS_GUARD_OUTCOME", False)
    monkeypatch.setattr(runtime_rmt, "GuardOutcome", dict, raising=False)

    guard = runtime_rmt.RMTGuard()
    guard.prepared = True
    guard._require_activation = True
    guard._activation_required_failed = True
    guard._activation_required_reason = "activation_required"

    result = guard.finalize(nn.Linear(2, 2, bias=False), adapter=None)

    assert result["passed"] is False
    assert result["metrics"]["activation_ready"] is False
    assert result["metrics"]["activation_reason"] == "activation_required"


def test_finalize_hydrates_edge_risk_and_returns_plain_dict(monkeypatch) -> None:
    monkeypatch.setattr(runtime_rmt, "HAS_GUARD_OUTCOME", False)
    monkeypatch.setattr(runtime_rmt, "GuardOutcome", dict, raising=False)

    guard = runtime_rmt.RMTGuard()
    guard.prepared = True
    guard._calibration_batches = [object()]

    monkeypatch.setattr(
        guard,
        "_compute_activation_edge_risk",
        lambda *_a, **_k: {
            "edge_risk_by_family": {"attn": 0.2},
            "edge_risk_by_module": {"layer": 0.2},
        },
    )
    monkeypatch.setattr(guard, "_compute_epsilon_violations", lambda: [])

    result = guard.finalize(nn.Linear(2, 2, bias=False), adapter=None)

    assert result["passed"] is True
    assert result["decision"] == "allow"
    assert result["metrics"]["edge_risk_by_family"]["attn"] == 0.2
    assert guard.edge_risk_by_module["layer"] == 0.2


def test_validate_uses_dict_finalize_path() -> None:
    guard = runtime_rmt.RMTGuard()
    guard.finalize = lambda *_a, **_k: {
        "passed": False,
        "metrics": {"prepared": True},
        "errors": ["boom"],
    }

    result = guard.validate(model=None, adapter=None, context={})

    assert result["passed"] is False
    assert result["decision"] == "monitor"
    assert result["violations"] == [
        {"type": "rmt_error", "severity": "error", "message": "boom"}
    ]


def test_set_run_context_and_epsilon_setters_ignore_invalid_values() -> None:
    guard = runtime_rmt.RMTGuard()

    guard.set_run_context(
        SimpleNamespace(context={"profile": "CI", "auto": {"tier": "Conservative"}})
    )
    default_before = guard.epsilon_default

    guard._set_epsilon_default("bad-float")
    guard._set_epsilon_by_family({"attn": "bad-float", "ffn": 0.25})

    assert guard._run_profile == "ci"
    assert guard._run_tier == "conservative"
    assert guard._require_activation is True
    assert guard.epsilon_default == default_before
    assert guard.epsilon_by_family["ffn"] == 0.25


def test_runtime_collection_and_tensor_fallbacks(monkeypatch) -> None:
    guard = runtime_rmt.RMTGuard()

    class IndexedSource:
        def __len__(self) -> int:
            return 3

        def __getitem__(self, idx: int) -> int:
            if idx == 1:
                raise RuntimeError("skip")
            return idx

    guard.activation_sampling["windows"]["indices_policy"] = "unknown"
    assert guard._collect_calibration_batches(IndexedSource(), 3) == [0, 2]

    class IterableOnly:
        def __iter__(self):
            return iter([10, 20, 30])

    assert guard._collect_calibration_batches(IterableOnly(), 2) == [10, 20]
    assert guard._collect_calibration_batches(object(), 2) == []

    monkeypatch.setattr(
        torch.Tensor,
        "to",
        lambda self, device: (_ for _ in ()).throw(RuntimeError("no device")),
    )
    input_ids, attention_mask = guard._prepare_activation_inputs(
        {
            "input_ids": torch.tensor([1, 2]),
            "attention_mask": torch.tensor([1, 1]),
        },
        torch.device("cpu"),
    )
    assert input_ids is not None and attention_mask is not None
    assert tuple(input_ids.shape) == (1, 2)
    assert tuple(attention_mask.shape) == (1, 2)

    monkeypatch.setattr(
        torch.Tensor,
        "sum",
        lambda self: (_ for _ in ()).throw(RuntimeError("bad sum")),
    )
    assert guard._batch_token_weight(torch.ones(2, 2), torch.ones(2, 2)) == 4

    monkeypatch.setattr(
        torch.Tensor,
        "numel",
        lambda self: (_ for _ in ()).throw(RuntimeError("bad numel")),
    )
    assert guard._batch_token_weight(torch.ones(2, 2), None) == 1


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
    original_mp_bulk_edge = runtime_rmt.rmt_math.mp_bulk_edge
    monkeypatch.setattr(
        runtime_rmt.rmt_math, "mp_bulk_edge", lambda *_a, **_k: float("nan")
    )
    assert guard._activation_edge_risk(torch.randn(3, 2)) is None

    monkeypatch.setattr(runtime_rmt.rmt_math, "mp_bulk_edge", original_mp_bulk_edge)
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
        runtime_rmt.rmt_detection,
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
