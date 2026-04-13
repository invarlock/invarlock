from __future__ import annotations

import builtins
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import invarlock.guards.rmt as runtime_rmt
import invarlock.guards.rmt_runtime as runtime_helpers


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
    assert result["decision"] == "block"
    assert result["violations"] == [
        {"type": "rmt_error", "severity": "error", "message": "boom"}
    ]


def test_runtime_helpers_cover_rollback_unknown_events_and_object_validate_path(
    monkeypatch,
) -> None:
    guard = runtime_rmt.RMTGuard()
    logged: list[str] = []
    monkeypatch.setattr(guard, "_get_linear_modules", lambda _model: [])
    monkeypatch.setattr(
        guard,
        "_log_event",
        lambda operation, **_kwargs: logged.append(operation),
    )
    monkeypatch.setattr(
        runtime_helpers.rmt_detection,
        "step5_detect_and_correct_modules",
        lambda *_a, **_k: {
            "events": [
                {"operation": "unknown", "module_name": "layer"},
                {
                    "operation": "rmt_correct_failed",
                    "module_name": "layer",
                    "error": "boom",
                },
            ],
            "passed": False,
        },
    )

    out = runtime_helpers.apply_rmt_detection_and_correction(guard, nn.Identity())
    assert out == {"passed": False}
    assert logged == ["rmt_correction", "rmt_correct_failed"]

    result = runtime_helpers.validate_rmt_guard(
        SimpleNamespace(
            finalize=lambda *_a, **_k: SimpleNamespace(
                passed=False,
                decision="rollback",
                metrics={"prepared": True},
                violations=[],
            )
        ),
        model=None,
        adapter=None,
        context={},
    )
    assert result.passed is False
    assert result.decision == "rollback"
    assert result.violations == ()


def test_apply_rmt_detection_and_correction_logs_success_event(monkeypatch) -> None:
    guard = runtime_rmt.RMTGuard()
    logged: list[tuple[str, dict[str, object]]] = []
    layer = nn.Linear(2, 2, bias=False)

    monkeypatch.setattr(guard, "_get_linear_modules", lambda _model: [("layer", layer)])
    monkeypatch.setattr(
        guard,
        "_log_event",
        lambda operation, **kwargs: logged.append((operation, kwargs)),
    )
    monkeypatch.setattr(
        runtime_helpers.rmt_detection,
        "step5_detect_and_correct_modules",
        lambda *_a, **_k: {
            "events": [
                {
                    "operation": "rmt_correct",
                    "module_name": "layer",
                    "pre_ratio": 1.2,
                    "threshold": 1.0,
                }
            ],
            "passed": True,
        },
    )

    out = runtime_helpers.apply_rmt_detection_and_correction(guard, nn.Identity())

    assert out == {"passed": True}
    assert logged[0][0] == "rmt_correction"
    assert logged[1] == (
        "rmt_correct",
        {
            "message": "Applied correction to layer",
            "module_name": "layer",
            "pre_ratio": 1.2,
            "threshold": 1.0,
        },
    )


def test_finalize_rmt_guard_allows_empty_hydration_result() -> None:
    guard = runtime_rmt.RMTGuard()
    guard.prepared = True
    guard._calibration_batches = [object()]
    guard._compute_activation_edge_risk = lambda *_a, **_k: None

    result = runtime_helpers.finalize_rmt_guard(
        guard,
        nn.Linear(2, 2, bias=False),
        has_guard_outcome=False,
        guard_outcome_type=dict,
    )

    assert result["passed"] is True
    assert result["metrics"]["edge_risk_by_family"] == {}
    assert result["violations"] == []


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


def test_prepare_rmt_guard_covers_invalid_window_count_and_activation_requirement(
    monkeypatch,
) -> None:
    guard = runtime_rmt.RMTGuard()
    guard._require_activation = True
    guard.activation_sampling["windows"]["count"] = "not-an-int"
    logged: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        runtime_helpers, "apply_rmt_policy_overrides", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        runtime_helpers.rmt_result_contract,
        "build_prepare_result",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(
        guard,
        "_collect_calibration_batches",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("unexpected collection")
        ),
    )
    monkeypatch.setattr(
        guard,
        "_log_event",
        lambda event, **kwargs: logged.append((event, kwargs)),
    )

    result = runtime_helpers.prepare_rmt_guard(
        guard,
        nn.Linear(2, 2, bias=False),
        adapter=None,
        calib=[object()],
        policy={"mode": "ci"},
    )

    assert result["ready"] is False
    assert result["error"] == "Activation batches required but unavailable"
    assert guard._activation_required_failed is True
    assert guard._activation_required_reason == "activation_required"
    assert [event for event, _kwargs in logged] == ["prepare"]


def test_prepare_rmt_guard_returns_ready_without_activation_baseline(
    monkeypatch,
) -> None:
    guard = runtime_rmt.RMTGuard()
    guard._require_activation = False

    monkeypatch.setattr(
        runtime_helpers, "apply_rmt_policy_overrides", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        runtime_helpers.rmt_result_contract,
        "build_prepare_result",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(guard, "_log_event", lambda *_a, **_k: None)

    result = runtime_helpers.prepare_rmt_guard(
        guard,
        nn.Linear(2, 2, bias=False),
        adapter=None,
        calib=None,
        policy={},
    )

    assert result["ready"] is True
    assert result["baseline_metrics"] == {}
    assert guard.prepared is True
    assert guard._activation_ready is False


def test_prepare_rmt_guard_handles_required_baseline_and_exception_paths(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime_helpers, "apply_rmt_policy_overrides", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        runtime_helpers.rmt_result_contract,
        "build_prepare_result",
        lambda **kwargs: kwargs,
    )

    require_guard = runtime_rmt.RMTGuard()
    require_guard._require_activation = True
    require_logged: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        require_guard,
        "_collect_calibration_batches",
        lambda *_a, **_k: [object()],
    )
    monkeypatch.setattr(
        require_guard,
        "_compute_activation_edge_risk",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        require_guard,
        "_log_event",
        lambda event, **kwargs: require_logged.append((event, kwargs)),
    )

    missing_result = runtime_helpers.prepare_rmt_guard(
        require_guard,
        nn.Linear(2, 2, bias=False),
        adapter=None,
        calib=[object()],
        policy={"mode": "ci"},
    )

    assert missing_result["ready"] is False
    assert missing_result["error"] == "Activation baseline unavailable"
    assert require_guard._activation_required_failed is True
    assert (
        require_guard._activation_required_reason == "activation_baseline_unavailable"
    )
    assert require_guard._activation_ready is False
    assert require_guard.prepared is False
    assert require_logged[0][0] == "prepare"

    success_guard = runtime_rmt.RMTGuard()
    success_guard.estimator = {"type": "power_iter"}
    success_guard.activation_sampling = {"windows": {"count": 1}}
    monkeypatch.setattr(
        success_guard,
        "_collect_calibration_batches",
        lambda *_a, **_k: [object()],
    )
    monkeypatch.setattr(
        success_guard,
        "_compute_activation_edge_risk",
        lambda *_a, **_k: {
            "edge_risk_by_family": {"attn": 0.2},
            "edge_risk_by_module": {"layer": 0.2},
        },
    )

    success_result = runtime_helpers.prepare_rmt_guard(
        success_guard,
        nn.Linear(2, 2, bias=False),
        adapter="adapter",
        calib=[object()],
        policy={"mode": "ci"},
    )

    assert success_result["ready"] is True
    assert success_result["baseline_metrics"]["edge_risk_by_family"] == {"attn": 0.2}
    assert success_result["baseline_metrics"]["measurement_contract"]["kind"] == (
        "activation_edge_risk"
    )
    assert success_guard.baseline_edge_risk_by_family == {"attn": 0.2}
    assert success_guard.baseline_edge_risk_by_module == {"layer": 0.2}
    assert success_guard._activation_ready is True
    assert success_guard.prepared is True
    assert success_guard.adapter == "adapter"

    failing_guard = runtime_rmt.RMTGuard()
    failing_logged: list[tuple[str, dict[str, object]]] = []
    failing_guard.activation_sampling = {"windows": {"count": 1}}
    monkeypatch.setattr(
        failing_guard,
        "_collect_calibration_batches",
        lambda *_a, **_k: [object()],
    )
    monkeypatch.setattr(
        failing_guard,
        "_compute_activation_edge_risk",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("baseline boom")),
    )
    monkeypatch.setattr(
        failing_guard,
        "_log_event",
        lambda event, **kwargs: failing_logged.append((event, kwargs)),
    )

    failure_result = runtime_helpers.prepare_rmt_guard(
        failing_guard,
        nn.Linear(2, 2, bias=False),
        adapter=None,
        calib=[object()],
        policy=None,
    )

    assert failure_result["ready"] is False
    assert failure_result["error"] == "baseline boom"
    assert failing_guard.prepared is False
    assert failing_logged[-1][0] == "prepare_failed"


def test_validate_rmt_guard_covers_object_and_dict_result_paths() -> None:
    object_result = SimpleNamespace(
        passed=True,
        decision="monitor",
        metrics={"prepared": True},
        violations=[
            {"type": "custom", "severity": "warning", "message": "warn"},
            "raw",
        ],
    )
    guard = SimpleNamespace(finalize=lambda *_a, **_k: object_result)

    typed = runtime_helpers.validate_rmt_guard(
        guard, model=None, adapter=None, context={}
    )
    assert typed.passed is True
    assert typed.decision == "monitor"
    assert typed.violations[0]["type"] == "custom"
    assert typed.violations[1]["message"] == "raw"

    guard.finalize = lambda *_a, **_k: {
        "passed": False,
        "decision": "block",
        "metrics": {"prepared": False},
        "errors": ["boom"],
    }
    fallback = runtime_helpers.validate_rmt_guard(
        guard, model=None, adapter=None, context={}
    )
    assert fallback.passed is False
    assert fallback.decision == "block"
    assert fallback.violations[0]["message"] == "boom"


def test_before_and_after_edit_rmt_guard_helper_paths(monkeypatch) -> None:
    before_events: list[tuple[str, dict[str, object]]] = []
    before_guard = SimpleNamespace(
        prepared=True,
        _log_event=lambda event, **kwargs: before_events.append((event, kwargs)),
    )

    runtime_helpers.before_edit_rmt_guard(before_guard, nn.Identity())

    assert before_events == [
        (
            "before_edit",
            {"message": "RMT guard ready for post-edit detection and correction"},
        )
    ]

    quiet_events: list[tuple[str, dict[str, object]]] = []
    quiet_guard = SimpleNamespace(
        prepared=False,
        _log_event=lambda event, **kwargs: quiet_events.append((event, kwargs)),
    )

    runtime_helpers.before_edit_rmt_guard(quiet_guard, nn.Identity())

    assert quiet_events == []

    skipped_events: list[tuple[str, dict[str, object]]] = []
    skipped_guard = SimpleNamespace(
        prepared=False,
        _log_event=lambda event, **kwargs: skipped_events.append((event, kwargs)),
    )

    runtime_helpers.after_edit_rmt_guard(skipped_guard, nn.Identity())

    assert skipped_events == [
        (
            "after_edit_skipped",
            {
                "level": "WARN",
                "message": "RMT guard not prepared, skipping post-edit detection",
            },
        )
    ]

    monkeypatch.setattr(
        runtime_helpers.rmt_result_contract,
        "build_after_edit_result",
        lambda: {"analysis_source": "activations_edge_risk"},
    )

    idle_guard = SimpleNamespace(
        prepared=True,
        _require_activation=False,
        _calibration_batches=[object()],
        _compute_activation_edge_risk=lambda *_a, **_k: None,
        _log_event=lambda *_a, **_k: None,
    )

    runtime_helpers.after_edit_rmt_guard(idle_guard, nn.Identity())

    assert idle_guard._last_result == {"analysis_source": "activations_edge_risk"}

    updated_guard = SimpleNamespace(
        prepared=True,
        _require_activation=False,
        _calibration_batches=[object()],
        _compute_activation_edge_risk=lambda *_a, **_k: {
            "edge_risk_by_family": {"attn": 0.3},
            "edge_risk_by_module": {"layer": 0.3},
        },
        _log_event=lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        runtime_helpers,
        "compute_epsilon_violations",
        lambda _guard: [{"family": "attn"}],
    )

    runtime_helpers.after_edit_rmt_guard(updated_guard, nn.Identity())

    assert updated_guard.edge_risk_by_family == {"attn": 0.3}
    assert updated_guard.edge_risk_by_module == {"layer": 0.3}
    assert updated_guard._last_result == {
        "edge_risk_by_family": {"attn": 0.3},
        "edge_risk_by_module": {"layer": 0.3},
    }
    assert updated_guard.epsilon_violations == [{"family": "attn"}]

    failure_events: list[tuple[str, dict[str, object]]] = []
    failing_guard = SimpleNamespace(
        prepared=True,
        _require_activation=False,
        _calibration_batches=[object()],
        _compute_activation_edge_risk=lambda *_a, **_k: (_ for _ in ()).throw(
            RuntimeError("after boom")
        ),
        _log_event=lambda event, **kwargs: failure_events.append((event, kwargs)),
    )

    runtime_helpers.after_edit_rmt_guard(failing_guard, nn.Identity())

    assert failing_guard._last_result == {"analysis_source": "activations_edge_risk"}
    assert failing_guard.epsilon_violations == []
    assert failure_events[-1][0] == "after_edit_failed"


def test_finalize_rmt_guard_covers_prepared_false_and_stable_paths(monkeypatch) -> None:
    guard = runtime_rmt.RMTGuard()
    guard.name = "rmt"
    guard.prepared = False

    monkeypatch.setattr(guard, "_log_event", lambda *_a, **_k: None)

    prepared_result = runtime_helpers.finalize_rmt_guard(
        guard,
        nn.Linear(2, 2, bias=False),
        has_guard_outcome=True,
        guard_outcome_type=SimpleNamespace,
    )
    assert prepared_result.passed is False
    assert prepared_result.decision == "block"

    plain_result = runtime_helpers.finalize_rmt_guard(
        guard,
        nn.Linear(2, 2, bias=False),
        has_guard_outcome=False,
        guard_outcome_type=SimpleNamespace,
    )
    assert plain_result["passed"] is False
    assert plain_result["metrics"]["prepared"] is False


def test_finalize_rmt_guard_hydrates_and_emits_epsilon_violation(
    monkeypatch,
) -> None:
    guard = runtime_rmt.RMTGuard()
    guard.prepared = True
    guard._calibration_batches = [object()]
    guard._require_activation = False
    guard._policy = {"seed": 123}
    guard._enabled = True
    guard._log_event = lambda *_a, **_k: None
    guard._compute_activation_edge_risk = lambda *_a, **_k: {
        "edge_risk_by_family": {"attn": 0.25},
        "edge_risk_by_module": {"layer": 0.25},
    }
    monkeypatch.setattr(
        runtime_helpers,
        "compute_epsilon_violations",
        lambda _guard: [
            {
                "family": "attn",
                "edge_base": 0.1,
                "edge_cur": 0.25,
                "allowed": 0.2,
                "epsilon": 0.1,
                "delta": 0.05,
            }
        ],
    )

    result = runtime_helpers.finalize_rmt_guard(
        guard,
        nn.Linear(2, 2, bias=False),
        has_guard_outcome=False,
        guard_outcome_type=SimpleNamespace,
    )

    assert result["passed"] is False
    assert result["decision"] == "block"
    assert result["violations"][0]["type"] == "epsilon_band"
    assert guard.edge_risk_by_family["attn"] == 0.25


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
