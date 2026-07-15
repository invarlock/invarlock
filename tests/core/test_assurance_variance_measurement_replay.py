from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch.nn as nn

from invarlock.core.assurance_contract import (
    build_assurance_section,
    strict_report_policy_errors,
)
from invarlock.core.assurance_guard_validation_variance_measurements import (
    _variance_measurement_errors,
)
from invarlock.core.bootstrap import compute_paired_delta_log_ci
from invarlock.guards.variance import VarianceGuard
from invarlock.guards.variance_evaluation import evaluate_calibration_pass
from invarlock.guards.variance_results import (
    build_finalize_metrics,
    build_finalize_result,
)
from invarlock.reporting.guards_variance import _extract_variance_analysis
from invarlock.utils import (
    bootstrap_mean_statistics,
    percentile_interval_from_statistics,
)
from tests.core._support_assurance_contract import strict_report


def _measurements(
    window_ids: list[str],
    *,
    ppl_a: float,
    ppl_b: float,
    seed: int = 123,
    alpha: float = 0.05,
) -> dict[str, Any]:
    coverage = len(window_ids)
    condition_a = {
        "ppl": [ppl_a] * coverage,
        "log_loss": [math.log(ppl_a)] * coverage,
        "token_counts": [16] * coverage,
    }
    condition_b = {
        "ppl": [ppl_b] * coverage,
        "log_loss": [math.log(ppl_b)] * coverage,
        "token_counts": [16] * coverage,
    }
    ratios = np.asarray([ppl_b / ppl_a] * coverage, dtype=float)
    ratio_ci = percentile_interval_from_statistics(
        bootstrap_mean_statistics(
            ratios,
            n_bootstrap=500,
            random_state=np.random.default_rng(seed),
        ),
        alpha=alpha,
    )
    delta_ci = compute_paired_delta_log_ci(
        condition_b["log_loss"],
        condition_a["log_loss"],
        weights=condition_a["token_counts"],
        method="bca",
        replicates=500,
        alpha=alpha,
        seed=seed + 211,
    )
    return {
        "window_ids": list(window_ids),
        "condition_a": condition_a,
        "condition_b": condition_b,
        "ratio_bootstrap": {
            "method": "percentile_mean_ppl_ratio",
            "replicates": 500,
            "alpha": alpha,
            "seed": seed,
        },
        "delta_log_bootstrap": {
            "method": "bca_paired_delta_log",
            "replicates": 500,
            "alpha": alpha,
            "seed": seed + 211,
            "weights": "condition_a_token_counts",
        },
        "ratio_ci": list(ratio_ci),
        "delta_log_ci": list(delta_ci),
    }


def _variance_guard(report: dict[str, Any]) -> dict[str, Any]:
    return next(entry for entry in report["guards"] if entry["name"] == "variance")


def _strict_noop_report() -> dict[str, Any]:
    report = strict_report()
    guard = _variance_guard(report)
    policy = guard["policy"]
    policy["alpha"] = 0.05
    report["variance"]["policy"]["alpha"] = 0.05
    report["resolved_policy"]["variance"]["alpha"] = 0.05
    guard["details"]["policy"]["alpha"] = 0.05
    window_ids = guard["metrics"]["ab_provenance"]["condition_a"]["window_ids"]
    measurements = _measurements(window_ids, ppl_a=100.0, ppl_b=100.0)
    guard["metrics"]["ab_measurements"] = copy.deepcopy(measurements)
    guard["details"]["stats"]["ab_measurements"] = copy.deepcopy(measurements)
    report["variance"]["ab_test"]["measurements"] = copy.deepcopy(measurements)
    report["assurance"] = build_assurance_section(report)
    return report


def _gain_inputs() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    window_ids = [f"window::{index}" for index in range(8)]
    measurements = _measurements(window_ids, ppl_a=100.0, ppl_b=98.0)
    delta = math.log(98.0) - math.log(100.0)
    provenance = {
        condition: {"window_ids": list(window_ids)}
        for condition in ("condition_a", "condition_b")
    }
    metrics = {
        "ab_measurements": copy.deepcopy(measurements),
        "ab_seed_used": 123,
        "ab_provenance": provenance,
        "ppl_no_ve": 100.0,
        "ppl_with_ve": 98.0,
        "ab_gain": 0.02,
        "ratio_ci": [0.98, 0.98],
        "predictive_gate": {
            "evaluated": True,
            "passed": True,
            "reason": "ci_gain_met",
            "delta_ci": [delta, delta],
            "gain_ci": [-delta, -delta],
            "mean_delta": delta,
        },
    }
    variance = {"ab_test": {"measurements": copy.deepcopy(measurements)}}
    entry = {"details": {"stats": {"ab_measurements": measurements}}}
    policy = {
        "alpha": 0.05,
        "min_effect_lognll": 0.005,
        "predictive_one_sided": True,
    }
    return variance, entry, metrics, policy


def test_producer_preserves_window_measurements_through_report_projection(
    monkeypatch,
) -> None:
    guard = VarianceGuard(
        {
            "scope": "both",
            "min_gain": 0.0,
            "alpha": 0.05,
            "calibration": {"windows": 2, "min_coverage": 2, "seed": 123},
        }
    )
    guard._explicit_noop_no_change = True
    guard._calibration_window_ids = ["window::0", "window::1"]
    monkeypatch.setattr(
        guard,
        "_compute_ppl_for_batches",
        lambda *_args, **_kwargs: (
            [10.0, 20.0],
            [math.log(10.0), math.log(20.0)],
            [8, 12],
        ),
    )
    evaluate_calibration_pass(
        guard,
        nn.Linear(2, 2),
        calibration_batches=[object(), object()],
        min_coverage=2,
        calib_seed=123,
        tag="post_edit",
    )

    metrics = build_finalize_metrics(
        scales={},
        target_modules={},
        stats=guard._stats,
        focus_modules=set(),
        enabled_after_ab=False,
        should_enable=False,
        ab_gain=0.0,
        ab_windows_used=2,
        ab_seed_used=123,
        monitor_only=False,
        policy=guard._policy,
        ppl_no_ve=15.0,
        ppl_with_ve=15.0,
        ratio_ci=(1.0, 1.0),
        calibration_stats=guard._calibration_stats,
        predictive_gate_state=guard._predictive_gate_state,
        raw_scales_pre_edit={},
        raw_scales_post_edit={},
    )
    result = build_finalize_result(
        passed=True,
        metrics=metrics,
        warnings=[],
        errors=[],
        finalize_time=0.0,
        enabled_after_ab=False,
        ppl_no_ve=15.0,
        scales={},
        stats=guard._stats,
        policy=guard._policy,
    )
    variance = _extract_variance_analysis(
        {"guards": [{"name": "variance", "metrics": metrics}]}
    )

    measurements = metrics["ab_measurements"]
    assert measurements["window_ids"] == ["window::0", "window::1"]
    assert measurements["condition_a"]["token_counts"] == [8, 12]
    assert measurements["condition_b"] == measurements["condition_a"]
    assert result["details"]["stats"]["ab_measurements"] == measurements
    assert variance["ab_test"]["measurements"] == measurements


def test_strict_noop_report_accepts_replayable_measurements() -> None:
    errors = strict_report_policy_errors(_strict_noop_report(), require_strict=True)
    assert errors == [], "\n".join(errors)


def test_gain_measurements_replay_to_exact_decision() -> None:
    variance, entry, metrics, policy = _gain_inputs()
    assert (
        _variance_measurement_errors(
            variance,
            entry,
            metrics,
            8,
            policy,
            source="guards[2]",
            no_adjustment=False,
        )
        == []
    )


def test_replay_rejects_coordinated_forged_ratio_interval() -> None:
    report = _strict_noop_report()
    guard = _variance_guard(report)
    for measurements in (
        guard["metrics"]["ab_measurements"],
        guard["details"]["stats"]["ab_measurements"],
        report["variance"]["ab_test"]["measurements"],
    ):
        measurements["ratio_ci"] = [0.5, 0.5]
    guard["metrics"]["ratio_ci"] = [0.5, 0.5]
    report["variance"]["ratio_ci"] = [0.5, 0.5]

    errors = strict_report_policy_errors(report, require_strict=True)
    assert any("deterministic bootstrap replay" in error for error in errors)


def test_replay_rejects_per_window_ppl_log_loss_disagreement() -> None:
    variance, entry, metrics, policy = _gain_inputs()
    for measurements in (
        metrics["ab_measurements"],
        entry["details"]["stats"]["ab_measurements"],
        variance["ab_test"]["measurements"],
    ):
        measurements["condition_b"]["ppl"][0] = 97.0

    errors = _variance_measurement_errors(
        variance,
        entry,
        metrics,
        8,
        policy,
        source="guards[2]",
        no_adjustment=False,
    )
    assert any("must equal exp(log_loss" in error for error in errors)


def test_replay_rejects_coordinated_unsupported_coverage() -> None:
    report = _strict_noop_report()
    guard = _variance_guard(report)
    extra_id = "final::8"
    for condition in ("condition_a", "condition_b"):
        guard["metrics"]["ab_provenance"][condition]["window_ids"].append(extra_id)
        guard["metrics"]["ab_provenance"][condition]["window_count"] = 9
        guard["details"]["stats"]["ab_provenance"][condition]["window_ids"].append(
            extra_id
        )
        guard["details"]["stats"]["ab_provenance"][condition]["window_count"] = 9
        report["variance"]["ab_test"]["provenance"][condition]["window_ids"].append(
            extra_id
        )
        report["variance"]["ab_test"]["provenance"][condition]["window_count"] = 9
    report["variance"]["ab_test"]["provenance"]["window_ids"].append(extra_id)
    for measurements in (
        guard["metrics"]["ab_measurements"],
        guard["details"]["stats"]["ab_measurements"],
        report["variance"]["ab_test"]["measurements"],
    ):
        measurements["window_ids"].append(extra_id)
        for condition in ("condition_a", "condition_b"):
            measurements[condition]["ppl"].append(100.0)
            measurements[condition]["log_loss"].append(math.log(100.0))
            measurements[condition]["token_counts"].append(16)
    for calibration in (
        guard["metrics"]["calibration"],
        report["variance"]["calibration"],
    ):
        calibration.update(coverage=9, requested=9)
    for policy in (
        guard["policy"],
        guard["details"]["policy"],
        report["variance"]["policy"],
        report["resolved_policy"]["variance"],
    ):
        policy["calibration"]["windows"] = 9
    guard["metrics"]["ab_windows_used"] = 9
    report["variance"]["ab_test"]["windows_used"] = 9
    guard["metrics"]["ab_point_estimates"]["coverage"] = 9
    guard["details"]["stats"]["ab_point_estimates"]["coverage"] = 9
    report["variance"]["ab_test"]["point_estimates"]["coverage"] = 9
    guard["details"]["stats"]["calibration"]["window_ids"].append(extra_id)

    errors = strict_report_policy_errors(report, require_strict=True)
    assert any("must match the consumed prefix" in error for error in errors)
