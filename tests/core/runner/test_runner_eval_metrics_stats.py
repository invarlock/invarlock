from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

import invarlock.core.runner_runtime.eval_metrics_stats as mod
from invarlock.core.exceptions import InvarlockError


class _FakeRunner:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, str, dict[str, Any]]] = []

    def _log_event(
        self,
        component: str,
        operation: str,
        level: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.events.append((component, operation, level, data or {}))


def _runtime(**overrides: Any) -> Any:
    payload = {
        "bootstrap_enabled": True,
        "bootstrap_replicates": 8,
        "bootstrap_alpha": 0.05,
        "bootstrap_seed": 13,
        "single_method": "bca",
        "delta_method": "paired",
        "pairing_context": {"source": "test"},
        "profile_label": "dev",
        "bootstrap_method": "bca",
        "ci_band": 0.95,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _slices(**overrides: Any) -> Any:
    payload = {
        "preview_mean_log": 0.1,
        "final_mean_log": 0.2,
        "delta_mean_log": 0.1,
        "ppl_ratio": 1.1,
        "pm_invalid": False,
        "preview_log_losses": [0.1, 0.2],
        "final_log_losses": [0.3, 0.45],
        "preview_token_counts": [2, 3],
        "final_token_counts": [5, 7],
        "pm_preview": 1.1,
        "pm_final": 1.2,
        "preview_window_ids": [1, 2],
        "preview_tokens": [[1], [2]],
        "final_window_ids": [1, 2],
        "final_tokens": [[1], [2]],
        "preview_batches_ct": 2,
        "final_batches_ct": 2,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def test_compute_bootstrap_delta_stats_marks_inconsistent_ratio_ci_invalid() -> None:
    runner = _FakeRunner()

    result = mod._compute_bootstrap_delta_stats(
        runner,
        _runtime(),
        _slices(),
        compute_independent_delta_log_ci_fn=lambda *_args, **_kwargs: (0.0, 0.5),
        logspace_to_ratio_ci_fn=lambda _delta_ci: (999.0, 999.0),
    )

    assert result.pm_invalid is True
    assert result.ratio_ci == pytest.approx((1.0, 1.6487212707))
    assert result.degraded_reason == "primary_metric_invalid"
    assert any(
        operation == "ratio_ci_inconsistent" for _, operation, _, _ in runner.events
    )


def test_compute_bootstrap_delta_stats_records_collapsed_independent_interval() -> None:
    runner = _FakeRunner()

    result = mod._compute_bootstrap_delta_stats(
        runner,
        _runtime(),
        _slices(
            preview_log_losses=[0.1],
            final_log_losses=[0.3],
            preview_token_counts=[2],
            final_token_counts=[4],
        ),
        compute_independent_delta_log_ci_fn=lambda *_args, **_kwargs: (0.0, 0.0),
        logspace_to_ratio_ci_fn=lambda _delta_ci: (1.0, 1.0),
    )

    assert result.degenerate_delta is True
    assert result.degenerate_reason == "constant_bootstrap_distribution"
    assert result.pm_invalid is False
    assert result.degraded_reason is None
    assert any(
        operation == "independent_slice_delta_degenerate"
        for _, operation, _, _ in runner.events
    )


def test_compute_bootstrap_delta_stats_marks_non_finite_delta_invalid() -> None:
    result = mod._compute_bootstrap_delta_stats(
        _FakeRunner(),
        _runtime(),
        _slices(delta_mean_log=float("nan"), ppl_ratio=float("inf")),
        compute_independent_delta_log_ci_fn=lambda *_args, **_kwargs: (0.0, 0.5),
        logspace_to_ratio_ci_fn=lambda _delta_ci: (1.0, 1.6487212707),
    )

    assert result.pm_invalid is True
    assert result.degraded_reason == "non_finite_delta"


def test_compute_bootstrap_delta_stats_marks_missing_slice_losses_invalid() -> None:
    result = mod._compute_bootstrap_delta_stats(
        _FakeRunner(),
        _runtime(bootstrap_enabled=False),
        _slices(
            preview_log_losses=[],
            final_log_losses=[],
            preview_token_counts=[],
            final_token_counts=[],
        ),
        compute_independent_delta_log_ci_fn=lambda *_args, **_kwargs: (0.0, 0.0),
        logspace_to_ratio_ci_fn=lambda _delta_ci: (1.0, 1.0),
    )

    assert result.degenerate_delta is True
    assert result.degenerate_reason == "missing_slice_losses"
    assert result.delta_ci_method == "none"
    assert result.delta_ci_reason == "missing_slice_losses"
    assert result.pm_invalid is True


def test_compute_bootstrap_delta_stats_allows_missing_pair_weights() -> None:
    result = mod._compute_bootstrap_delta_stats(
        _FakeRunner(),
        _runtime(bootstrap_enabled=False),
        _slices(
            preview_token_counts=[],
            final_token_counts=[5],
            preview_log_losses=[0.1, 0.2],
            final_log_losses=[0.3, 0.45],
        ),
        compute_independent_delta_log_ci_fn=lambda *_args, **_kwargs: (0.0, 0.0),
        logspace_to_ratio_ci_fn=lambda _delta_ci: (1.0, 1.0),
    )

    assert result.degenerate_delta is False
    assert result.delta_ci_method == "none"
    assert result.delta_ci_reason == "bootstrap_disabled"
    assert result.degraded_reason is None


def test_compute_bootstrap_delta_stats_passes_disjoint_arms_and_weights_separately() -> (
    None
):
    captured: dict[str, Any] = {}

    def independent_ci(final, preview, **kwargs):  # noqa: ANN001
        captured["final"] = list(final)
        captured["preview"] = list(preview)
        captured.update(kwargs)
        return (-0.1, 0.2)

    result = mod._compute_bootstrap_delta_stats(
        _FakeRunner(),
        _runtime(),
        _slices(
            preview_log_losses=[0.1, 0.2],
            final_log_losses=[0.3, 0.4, 0.5],
            preview_token_counts=[2, 3],
            final_token_counts=[5, 7, 11],
        ),
        compute_independent_delta_log_ci_fn=independent_ci,
        logspace_to_ratio_ci_fn=lambda ci: (math.exp(ci[0]), math.exp(ci[1])),
    )

    assert captured["preview"] == [0.1, 0.2]
    assert captured["final"] == [0.3, 0.4, 0.5]
    assert captured["preview_weights"] == [2.0, 3.0]
    assert captured["final_weights"] == [5.0, 7.0, 11.0]
    assert captured["method"] == "percentile"
    assert captured["seed"] == 110
    assert result.delta_ci_method == "independent_percentile_delta_log"


@pytest.mark.parametrize(
    ("overrides", "expected_event"),
    [
        ({"preview_token_counts": [2]}, "preview_slice_weight_mismatch"),
        ({"final_token_counts": [5]}, "final_slice_weight_mismatch"),
    ],
)
def test_compute_bootstrap_delta_stats_rejects_misaligned_slice_weights(
    overrides: dict[str, Any], expected_event: str
) -> None:
    runner = _FakeRunner()

    result = mod._compute_bootstrap_delta_stats(
        runner,
        _runtime(),
        _slices(**overrides),
        compute_independent_delta_log_ci_fn=lambda *_args, **_kwargs: (0.0, 0.1),
        logspace_to_ratio_ci_fn=lambda ci: (math.exp(ci[0]), math.exp(ci[1])),
    )

    assert result.pm_invalid is True
    assert result.degraded_reason == "primary_metric_invalid"
    assert any(operation == expected_event for _, operation, _, _ in runner.events)


def test_compute_bootstrap_delta_stats_fails_closed_on_bootstrap_error() -> None:
    runner = _FakeRunner()

    def raise_invalid(*_args: Any, **_kwargs: Any) -> tuple[float, float]:
        raise ValueError("invalid independent slice")

    result = mod._compute_bootstrap_delta_stats(
        runner,
        _runtime(),
        _slices(),
        compute_independent_delta_log_ci_fn=raise_invalid,
        logspace_to_ratio_ci_fn=lambda ci: (math.exp(ci[0]), math.exp(ci[1])),
    )

    assert result.pm_invalid is True
    assert result.delta_ci_reason == "independent_slice_bootstrap_error"
    assert result.degraded_reason == "primary_metric_invalid"
    assert any(
        operation == "independent_slice_delta_error"
        and data["reason"] == "invalid independent slice"
        for _, operation, _, data in runner.events
    )


def test_evaluate_pairing_and_coverage_raises_in_ci_release_mismatch() -> None:
    with pytest.raises(RuntimeError, match="Window pairing mismatch"):
        mod._evaluate_pairing_and_coverage(
            _FakeRunner(),
            _runtime(profile_label="ci"),
            _slices(),
            config=SimpleNamespace(context={"auto": {"tier": "balanced"}}),
            coverage_requirements={},
            compute_window_pairing_metrics_fn=lambda **_kwargs: {
                "preview": {"matched": 1, "expected": 2, "reason": "missing"},
                "final": {"matched": 1, "expected": 2, "reason": "missing"},
                "match_fraction": 0.5,
                "overlap_fraction": 0.0,
                "duplicate_fraction": 0.0,
                "count_mismatch": False,
                "reason": "missing",
            },
            assess_bootstrap_coverage_fn=lambda **_kwargs: {
                "preview_required": 1,
                "final_required": 1,
                "replicates_required": 1,
                "preview_ok": True,
                "final_ok": True,
                "replicates_ok": True,
                "coverage": {},
            },
        )


def test_evaluate_pairing_and_coverage_raises_in_ci_release_for_bootstrap_floor() -> (
    None
):
    with pytest.raises(InvarlockError, match="INSUFFICIENT-SAMPLE"):
        mod._evaluate_pairing_and_coverage(
            _FakeRunner(),
            _runtime(profile_label="release"),
            _slices(),
            config=SimpleNamespace(context={"auto": {"tier": "balanced"}}),
            coverage_requirements={},
            compute_window_pairing_metrics_fn=lambda **_kwargs: {
                "preview": {"matched": 2, "expected": 2, "reason": None},
                "final": {"matched": 2, "expected": 2, "reason": None},
                "match_fraction": 1.0,
                "overlap_fraction": 0.0,
                "duplicate_fraction": 0.0,
                "count_mismatch": False,
                "reason": None,
            },
            assess_bootstrap_coverage_fn=lambda **_kwargs: {
                "preview_required": 8,
                "final_required": 8,
                "replicates_required": 32,
                "preview_ok": False,
                "final_ok": True,
                "replicates_ok": False,
                "coverage": {},
            },
        )


def test_evaluate_pairing_and_coverage_records_release_profile_floors() -> None:
    captured: dict[str, Any] = {}

    def assess(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "preview_required": 200,
            "final_required": 200,
            "replicates_required": 3200,
            "preview_ok": True,
            "final_ok": True,
            "replicates_ok": True,
            "coverage": {},
        }

    mod._evaluate_pairing_and_coverage(
        _FakeRunner(),
        _runtime(profile_label="release"),
        _slices(),
        config=SimpleNamespace(context={"auto": {"tier": "balanced"}}),
        coverage_requirements={
            "balanced": {"preview": 180, "final": 180, "replicates": 1200}
        },
        compute_window_pairing_metrics_fn=lambda **_kwargs: {
            "preview": {"matched": 2, "expected": 2, "reason": None},
            "final": {"matched": 2, "expected": 2, "reason": None},
            "match_fraction": 1.0,
            "overlap_fraction": 0.0,
            "duplicate_fraction": 0.0,
            "count_mismatch": False,
            "reason": None,
        },
        assess_bootstrap_coverage_fn=assess,
    )

    assert captured["requirements"]["balanced"] == {
        "preview": 200,
        "final": 200,
        "replicates": 3200,
    }


def test_evaluate_pairing_and_coverage_logs_overlap_warning_and_uses_balanced_fallback() -> (
    None
):
    runner = _FakeRunner()

    result = mod._evaluate_pairing_and_coverage(
        runner,
        _runtime(profile_label="dev"),
        _slices(),
        config=SimpleNamespace(context={"auto": []}),
        coverage_requirements={},
        compute_window_pairing_metrics_fn=lambda **_kwargs: {
            "preview": {"matched": 2, "expected": 2, "reason": None},
            "final": {"matched": 2, "expected": 2, "reason": None},
            "match_fraction": 1.0,
            "overlap_fraction": 0.25,
            "duplicate_fraction": 0.25,
            "count_mismatch": False,
            "reason": None,
        },
        assess_bootstrap_coverage_fn=lambda **_kwargs: {
            "preview_required": 1,
            "final_required": 1,
            "replicates_required": 1,
            "preview_ok": True,
            "final_ok": True,
            "replicates_ok": True,
            "coverage": {"preview": {"used": 2, "required": 1, "ok": True}},
        },
    )

    assert result.window_overlap_fraction == 0.25
    assert result.bootstrap_info["coverage"] == {
        "preview": {"used": 2, "required": 1, "ok": True}
    }
    assert any(
        operation == "window_overlap_warning" for _, operation, _, _ in runner.events
    )


def test_evaluate_pairing_and_coverage_raises_on_ci_release_overlap() -> None:
    with pytest.raises(RuntimeError, match="Window overlap detected"):
        mod._evaluate_pairing_and_coverage(
            _FakeRunner(),
            _runtime(profile_label="ci"),
            _slices(),
            config=SimpleNamespace(context={"auto": {"tier": "balanced"}}),
            coverage_requirements={},
            compute_window_pairing_metrics_fn=lambda **_kwargs: {
                "preview": {"matched": 2, "expected": 2, "reason": None},
                "final": {"matched": 2, "expected": 2, "reason": None},
                "match_fraction": 1.0,
                "overlap_fraction": 0.25,
                "duplicate_fraction": 0.25,
                "count_mismatch": False,
                "reason": None,
            },
            assess_bootstrap_coverage_fn=lambda **_kwargs: {
                "preview_required": 1,
                "final_required": 1,
                "replicates_required": 1,
                "preview_ok": True,
                "final_ok": True,
                "replicates_ok": True,
                "coverage": {},
            },
        )


def test_evaluate_pairing_and_coverage_raises_on_ci_release_count_mismatch() -> None:
    with pytest.raises(RuntimeError, match="Window count mismatch detected"):
        mod._evaluate_pairing_and_coverage(
            _FakeRunner(),
            _runtime(profile_label="release"),
            _slices(),
            config=SimpleNamespace(context={"auto": {"tier": "balanced"}}),
            coverage_requirements={},
            compute_window_pairing_metrics_fn=lambda **_kwargs: {
                "preview": {"matched": 2, "expected": 2, "reason": None},
                "final": {"matched": 2, "expected": 2, "reason": None},
                "match_fraction": 1.0,
                "overlap_fraction": 0.0,
                "duplicate_fraction": 0.0,
                "count_mismatch": True,
                "reason": None,
            },
            assess_bootstrap_coverage_fn=lambda **_kwargs: {
                "preview_required": 1,
                "final_required": 1,
                "replicates_required": 1,
                "preview_ok": True,
                "final_ok": True,
                "replicates_ok": True,
                "coverage": {},
            },
        )


def test_pairing_error_result_keeps_runtime_bootstrap_metadata() -> None:
    runtime = _runtime(
        bootstrap_enabled=False,
        bootstrap_method="percentile",
        bootstrap_alpha=0.1,
        bootstrap_replicates=16,
        bootstrap_seed=23,
        ci_band=0.9,
    )

    result = mod._pairing_error_result(runtime)

    assert result.preview_pair_stats == {"matched": 0, "expected": 0}
    assert result.final_pair_stats == {"matched": 0, "expected": 0}
    assert result.bootstrap_info == {
        "enabled": False,
        "method": "percentile",
        "preview_final_delta_basis": "independent_disjoint_slices",
        "preview_final_delta_method": "none",
        "preview_final_delta_seed": None,
        "alpha": 0.1,
        "replicates": 16,
        "seed": 23,
        "ci_band": 0.9,
    }
