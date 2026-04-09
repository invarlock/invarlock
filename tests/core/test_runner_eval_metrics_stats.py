from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import invarlock.core.runner_eval_metrics_stats as mod
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
        compute_paired_delta_log_ci_fn=lambda *_args, **_kwargs: (0.0, 0.5),
        logspace_to_ratio_ci_fn=lambda _delta_ci: (999.0, 999.0),
    )

    assert result.pm_invalid is True
    assert result.ratio_ci == pytest.approx((1.0, 1.6487212707))
    assert result.degraded_reason == "primary_metric_invalid"
    assert any(
        operation == "ratio_ci_inconsistent" for _, operation, _, _ in runner.events
    )


def test_compute_bootstrap_delta_stats_marks_degenerate_single_pair() -> None:
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
        compute_paired_delta_log_ci_fn=lambda *_args, **_kwargs: (0.0, 0.0),
        logspace_to_ratio_ci_fn=lambda _delta_ci: (1.0, 1.0),
    )

    assert result.degenerate_delta is True
    assert result.degenerate_reason == "single_pair"
    assert result.pm_invalid is True
    assert result.degraded_reason == "degenerate_delta:single_pair"
    assert any(
        operation == "degenerate_delta_samples" for _, operation, _, _ in runner.events
    )


def test_compute_bootstrap_delta_stats_marks_non_finite_delta_invalid() -> None:
    result = mod._compute_bootstrap_delta_stats(
        _FakeRunner(),
        _runtime(),
        _slices(delta_mean_log=float("nan"), ppl_ratio=float("inf")),
        compute_paired_delta_log_ci_fn=lambda *_args, **_kwargs: (0.0, 0.5),
        logspace_to_ratio_ci_fn=lambda _delta_ci: (1.0, 1.6487212707),
    )

    assert result.pm_invalid is True
    assert result.degraded_reason == "non_finite_delta"


def test_compute_bootstrap_delta_stats_marks_no_pairs_when_losses_missing() -> None:
    result = mod._compute_bootstrap_delta_stats(
        _FakeRunner(),
        _runtime(bootstrap_enabled=False),
        _slices(
            preview_log_losses=[],
            final_log_losses=[],
            preview_token_counts=[],
            final_token_counts=[],
        ),
        compute_paired_delta_log_ci_fn=lambda *_args, **_kwargs: (0.0, 0.0),
        logspace_to_ratio_ci_fn=lambda _delta_ci: (1.0, 1.0),
    )

    assert result.degenerate_delta is True
    assert result.degenerate_reason == "no_pairs"
    assert result.delta_samples == []
    assert result.delta_weights == []


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
        compute_paired_delta_log_ci_fn=lambda *_args, **_kwargs: (0.0, 0.0),
        logspace_to_ratio_ci_fn=lambda _delta_ci: (1.0, 1.0),
    )

    assert result.degenerate_delta is False
    assert result.delta_weights == []
    assert result.degraded_reason is None


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
        "alpha": 0.1,
        "replicates": 16,
        "seed": 23,
        "ci_band": 0.9,
    }
