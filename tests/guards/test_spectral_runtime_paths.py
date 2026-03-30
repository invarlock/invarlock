from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from invarlock.guards import spectral_runtime


class _RuntimeGuard:
    def __init__(self) -> None:
        self.scope = "all"
        self.sigma_quantile = 0.8
        self.deadband = 0.1
        self.max_caps = 1
        self.max_spectral_norm = None
        self.correction_enabled = False
        self.ignore_preview_inflation = False
        self.family_caps = {"ffn": {"kappa": 2.5}}
        self.multiple_testing = {"method": "bh", "alpha": 0.05, "m": 4}
        self.estimator = {"type": "power_iter", "iters": 2, "init": "ones"}
        self.degeneracy = {
            "enabled": False,
            "stable_rank": {"warn_ratio": 0.5, "fatal_ratio": 0.25},
            "norm_collapse": {"warn_ratio": 0.25, "fatal_ratio": 0.1},
        }
        self.config: dict[str, object] = {}
        self.baseline_sigmas: dict[str, float] = {}
        self.baseline_family_stats: dict[str, dict[str, float]] = {}
        self.module_family_map: dict[str, str] = {}
        self.baseline_degeneracy: dict[str, dict[str, float]] = {}
        self.baseline_metrics: dict[str, object] = {}
        self.pre_edit_metrics: dict[str, float] = {}
        self.pre_edit_z_scores: dict[str, float] = {}
        self.current_metrics: dict[str, float] = {}
        self.latest_z_scores: dict[str, float] = {}
        self.violations: list[dict[str, object]] = []
        self.target_sigma = 0.0
        self.prepared = False
        self.logs: list[tuple[str, dict[str, object]]] = []

    def _log_event(self, event: str, **kwargs: object) -> None:
        self.logs.append((event, dict(kwargs)))

    def _serialize_policy(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "sigma_quantile": self.sigma_quantile,
            "deadband": self.deadband,
        }

    def _select_budgeted_violations(
        self, budgeted_violations: list[dict[str, object]]
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        return list(budgeted_violations), {"selected": len(budgeted_violations)}

    def prepare(
        self, model: object, adapter: object, calib: object, policy: dict[str, object]
    ) -> dict[str, object]:
        return spectral_runtime.prepare_guard(
            self,
            model,
            adapter,
            calib,
            policy,
            classify_model_families_fn=lambda *_a, **_k: {"module": "ffn"},
            compute_family_stats_fn=lambda *_a, **_k: {"ffn": {"mean": 1.0, "std": 0.1}},
            summarize_sigmas_fn=lambda sigmas: {"count": len(sigmas)},
            percentile_fn=lambda *_a, **_k: 1.0,
        )


def test_raise_prepare_failure_plain_and_chained() -> None:
    with pytest.raises(RuntimeError, match="plain failure"):
        spectral_runtime._raise_prepare_failure("plain failure")

    cause = ValueError("boom")
    with pytest.raises(RuntimeError, match="wrapped") as excinfo:
        spectral_runtime._raise_prepare_failure("wrapped", error=cause)

    assert excinfo.value.__cause__ is cause


def test_prepare_guard_success_without_degeneracy() -> None:
    guard = _RuntimeGuard()
    guard._get_scoped_modules = lambda _model: [("module", torch.nn.Linear(2, 2))]
    guard._capture_sigmas = lambda _model, phase: {"module": 1.5}

    result = spectral_runtime.prepare_guard(
        guard,
        model=object(),
        adapter=None,
        calib=None,
        policy={"ignore_preview_inflation": True},
        classify_model_families_fn=lambda *_a, **_k: {"module": "ffn"},
        compute_family_stats_fn=lambda *_a, **_k: {"ffn": {"mean": 1.0, "std": 0.1}},
        summarize_sigmas_fn=lambda sigmas: {"modules_checked": len(sigmas)},
        percentile_fn=lambda *_a, **_k: 1.25,
    )

    assert result["ready"] is True
    assert guard.prepared is True
    assert guard.target_sigma == pytest.approx(1.25)
    assert guard.baseline_metrics["modules_checked"] == 1
    assert guard.baseline_metrics["measurement_contract"]["degeneracy"] == guard.degeneracy


def test_prepare_guard_records_degeneracy_diagnostics_and_skip_conditions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = _RuntimeGuard()
    guard.degeneracy["enabled"] = True
    modules = [
        ("bad_weight", SimpleNamespace(weight="not-a-tensor")),
        ("missing_sigma", torch.nn.Linear(2, 2, bias=False)),
        ("explode", torch.nn.Linear(2, 2, bias=False)),
    ]
    guard._get_scoped_modules = lambda _model: modules
    guard._capture_sigmas = lambda _model, phase: {"explode": 2.0}

    monkeypatch.setattr(
        spectral_runtime,
        "row_col_norm_extrema",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad norms")),
    )

    result = spectral_runtime.prepare_guard(
        guard,
        model=object(),
        adapter=None,
        calib=None,
        policy={},
        classify_model_families_fn=lambda *_a, **_k: {
            "bad_weight": "ffn",
            "missing_sigma": "ffn",
            "explode": "ffn",
        },
        compute_family_stats_fn=lambda *_a, **_k: {"ffn": {"mean": 1.0, "std": 0.1}},
        summarize_sigmas_fn=lambda sigmas: {"captured": len(sigmas)},
        percentile_fn=lambda *_a, **_k: 2.0,
    )

    assert result["ready"] is True
    diagnostics = guard.baseline_metrics["degeneracy_diagnostics"]
    assert diagnostics[0]["kind"] == "spectral_degeneracy_unavailable"
    assert guard.baseline_metrics["baseline_degeneracy"] == {}


def test_prepare_guard_percentile_failure_logs_and_raises() -> None:
    guard = _RuntimeGuard()
    guard._get_scoped_modules = lambda _model: []
    guard._capture_sigmas = lambda _model, phase: {"module": 1.0}

    with pytest.raises(RuntimeError, match="Failed to prepare spectral guard."):
        spectral_runtime.prepare_guard(
            guard,
            model=object(),
            adapter=None,
            calib=None,
            policy={},
            classify_model_families_fn=lambda *_a, **_k: {"module": "ffn"},
            compute_family_stats_fn=lambda *_a, **_k: {"ffn": {"mean": 1.0, "std": 0.1}},
            summarize_sigmas_fn=lambda sigmas: {"captured": len(sigmas)},
            percentile_fn=lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad percentile")),
        )

    assert guard.prepared is False
    assert any(event == "prepare_failed" for event, _details in guard.logs)


def test_before_edit_guard_skips_when_not_prepared() -> None:
    guard = _RuntimeGuard()

    spectral_runtime.before_edit_guard(guard, model=object())

    assert guard.logs[-1][0] == "before_edit_skipped"


def test_before_edit_guard_captures_state_when_prepared() -> None:
    guard = _RuntimeGuard()
    guard.prepared = True
    guard.baseline_family_stats = {"ffn": {"mean": 0.0, "std": 1.0}}
    guard.module_family_map = {"module": "ffn"}
    guard.baseline_sigmas = {"module": 1.0}
    guard._capture_sigmas = lambda _model, phase: {"module": 1.2}

    spectral_runtime.before_edit_guard(
        guard,
        model=object(),
        compute_z_scores_fn=lambda *_a, **_k: {"module": 0.2},
    )

    assert guard.pre_edit_metrics == {"module": 1.2}
    assert guard.pre_edit_z_scores == {"module": 0.2}
    assert guard.logs[-1][0] == "before_edit"


def test_after_edit_guard_skips_when_not_prepared() -> None:
    guard = _RuntimeGuard()

    spectral_runtime.after_edit_guard(guard, model=object())

    assert guard.logs[-1][0] == "after_edit_skipped"


def test_after_edit_guard_applies_control_when_enabled() -> None:
    guard = _RuntimeGuard()
    guard.prepared = True
    guard.correction_enabled = True
    guard.baseline_sigmas = {"module": 1.0}
    guard.target_sigma = 0.95
    guard._capture_sigmas = lambda _model, phase: {"module": 1.4}
    guard._detect_spectral_violations = lambda _model, metrics, phase: [
        {
            "type": "sigma_drift",
            "severity": "warning",
            "message": "module drifted",
            "family": "ffn",
            "module": "module",
        }
    ]
    seen: dict[str, object] = {}

    def fake_apply_spectral_control(model: object, policy: dict[str, object]) -> dict[str, object]:
        seen["model"] = model
        seen["policy"] = dict(policy)
        return {"applied": True}

    spectral_runtime.after_edit_guard(
        guard,
        model=object(),
        apply_spectral_control_fn=fake_apply_spectral_control,
    )

    assert guard.current_metrics == {"module": 1.4}
    assert guard.violations[0]["type"] == "sigma_drift"
    assert seen["policy"] == {
        "sigma_quantile": guard.sigma_quantile,
        "scope": guard.scope,
        "baseline_sigmas": guard.baseline_sigmas,
        "target_sigma": guard.target_sigma,
    }
    assert any(event == "spectral_control_applied" for event, _details in guard.logs)
    assert guard.logs[-1][0] == "after_edit"


def test_after_edit_guard_logs_and_raises_on_failure() -> None:
    guard = _RuntimeGuard()
    guard.prepared = True
    guard._capture_sigmas = lambda _model, phase: (_ for _ in ()).throw(ValueError("bad capture"))

    with pytest.raises(RuntimeError, match="Post-edit spectral analysis failed."):
        spectral_runtime.after_edit_guard(guard, model=object())

    assert guard.logs[-1][0] == "after_edit_failed"


def test_finalize_guard_returns_error_payload_when_not_prepared() -> None:
    result = spectral_runtime.finalize_guard(_RuntimeGuard(), model=object())

    assert result["passed"] is False
    assert result["errors"] == ["Preparation failed or not called"]
    assert result["diagnostics"][0]["kind"] == "spectral_preparation"
