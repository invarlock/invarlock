from __future__ import annotations

from invarlock.guards.spectral_results import (
    _quantile,
    build_spectral_diagnostics,
    build_spectral_finalize_metrics,
    build_spectral_validation_metrics,
    categorize_spectral_messages,
    compute_family_observability,
    evaluate_spectral_outcome,
    partition_spectral_violations,
    spectral_validation_message,
)


def test_quantile_helper_branches() -> None:
    assert _quantile([], 0.5) == 0.0
    assert _quantile([3.0], 0.5) == 3.0
    assert _quantile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert _quantile([1.0, 3.0], 0.25) == 1.5


def test_compute_family_observability_branches() -> None:
    quantiles, top = compute_family_observability(
        {
            "m1": 1.0,
            "m2": -2.0,
            "m3": "bad",
            "m5": object(),
            "m4": 3.0,
        },
        {"m1": "ffn", "m2": "ffn", "m4": "attn", "m5": "ffn"},
    )
    assert quantiles["ffn"]["count"] == 2
    assert quantiles["attn"]["max"] == 3.0
    assert top["ffn"][0]["module"] == "m2"


def test_compute_family_observability_clamps_invalid_top_k() -> None:
    quantiles, top = compute_family_observability(
        {"m1": 1.0, "m2": 2.0},
        {"m1": "ffn", "m2": "ffn"},
        top_k=-1,
    )
    _, default_top = compute_family_observability(
        {"m1": 1.0, "m2": 2.0},
        {"m1": "ffn", "m2": "ffn"},
        top_k="bad",
    )

    assert quantiles["ffn"]["count"] == 2
    assert top["ffn"] == []
    assert len(default_top["ffn"]) == 2


def test_partition_and_evaluate_spectral_outcome() -> None:
    violations = [
        {"type": "family_z_cap", "severity": "budgeted", "message": "warn"},
        {"type": "max_spectral_norm", "severity": "budgeted", "message": "fatal"},
        {"type": "other", "severity": "fatal", "message": "fatal2"},
    ]
    fatal, budgeted = partition_spectral_violations(violations)
    assert len(fatal) == 2
    assert len(budgeted) == 1

    outcome = evaluate_spectral_outcome(
        fatal_violations=[],
        budgeted_violations=budgeted,
        selected_budgeted=budgeted,
        max_caps=0,
    )
    assert outcome["passed"] is False
    assert outcome["action"] == "abort"
    assert outcome["caps_exceeded"] is True

    warn_outcome = evaluate_spectral_outcome(
        fatal_violations=[],
        budgeted_violations=budgeted,
        selected_budgeted=budgeted,
        max_caps=2,
    )
    assert warn_outcome["passed"] is True
    assert warn_outcome["action"] == "warn"

    continue_outcome = evaluate_spectral_outcome(
        fatal_violations=[],
        budgeted_violations=[],
        selected_budgeted=[],
        max_caps=2,
    )
    assert continue_outcome["passed"] is True
    assert continue_outcome["action"] == "continue"


def test_build_spectral_diagnostics_marks_fatal_and_budgeted_entries() -> None:
    diagnostics = build_spectral_diagnostics(
        [
            {"type": "family_z_cap", "severity": "budgeted", "message": "warn"},
            {"type": "max_spectral_norm", "severity": "budgeted"},
        ]
    )

    assert diagnostics[0]["severity"] == "warning"
    assert diagnostics[1]["severity"] == "error"
    assert diagnostics[1]["message"] == ""


def test_build_spectral_validation_metrics_and_message() -> None:
    metrics = build_spectral_validation_metrics(
        current_metrics={"m": 2.0},
        candidate_violations=[{"message": "warn"}],
        selected_violations=[{"message": "warn"}],
        fatal_violations=[],
        candidate_budgeted=1,
        caps_applied=1,
        caps_exceeded=False,
        family_summary={"ffn": {"max": 2.0}},
        family_caps={"ffn": {"kappa": 2.5}},
        sigma_quantile=0.95,
        deadband=0.05,
        max_caps=2,
        multiple_testing={"method": "bh"},
        multiple_testing_selection={"families_selected": ["ffn"]},
        estimator={"iters": 4},
        degeneracy={"enabled": True},
        family_quantiles={"ffn": {"q95": 2.0}},
        top_z_scores={"ffn": [{"module": "m", "z": 2.0, "family": "ffn"}]},
    )
    assert metrics["stability_score"] == 0.0
    assert metrics["top_z_scores"]["ffn"][0]["module"] == "m"
    assert (
        spectral_validation_message(
            passed=True,
            fatal_violations=[],
            caps_applied=1,
            max_caps=2,
        )
        == "Spectral validation passed with 1 violations (caps_applied=1, max_caps=2)"
    )


def test_build_spectral_validation_metrics_without_optional_family_views() -> None:
    metrics = build_spectral_validation_metrics(
        current_metrics={},
        candidate_violations=[],
        selected_violations=[],
        fatal_violations=[],
        candidate_budgeted=0,
        caps_applied=0,
        caps_exceeded=False,
        family_summary={},
        family_caps={},
        sigma_quantile=0.95,
        deadband=0.05,
        max_caps=2,
        multiple_testing={},
        multiple_testing_selection={},
        estimator={},
        degeneracy={},
        family_quantiles={},
        top_z_scores={},
    )
    assert metrics["modules_checked"] == 0
    assert "family_z_quantiles" not in metrics
    assert "top_z_scores" not in metrics


def test_build_spectral_finalize_metrics_and_message_buckets() -> None:
    metrics = build_spectral_finalize_metrics(
        final_metrics={"m": 3.0},
        selected_violations=[
            {"message": "warn", "severity": "budgeted", "type": "family_z_cap"}
        ],
        candidate_violations=[
            {"message": "warn", "severity": "budgeted", "type": "family_z_cap"}
        ],
        fatal_violations=[],
        candidate_budgeted=1,
        caps_applied=1,
        caps_exceeded=False,
        baseline_metrics={"m": 1.0},
        scope="ffn",
        correction_enabled=True,
        family_caps={"ffn": {"kappa": 2.5}},
        final_z_summary={"ffn": {"max": 2.0}},
        final_family_stats={"ffn": {"mean": 1.0}},
        sigma_quantile=0.95,
        deadband=0.05,
        max_caps=2,
        multiple_testing={"method": "bh"},
        multiple_testing_selection={"families_selected": ["ffn"]},
        estimator={"iters": 4},
        degeneracy={"enabled": True},
        family_quantiles={"ffn": {"q95": 2.0}},
        top_z_scores={"ffn": [{"module": "m", "z": 2.0, "family": "ffn"}]},
    )
    assert metrics["correction_applied"] is True
    warnings, errors = categorize_spectral_messages(
        [
            {"message": "warn", "severity": "budgeted", "type": "family_z_cap"},
            {"message": "fatal", "severity": "fatal", "type": "max_spectral_norm"},
        ]
    )
    assert warnings == ["warn"]
    assert errors == ["fatal"]
    assert (
        spectral_validation_message(
            passed=False,
            fatal_violations=[{"message": "fatal"}],
            caps_applied=0,
            max_caps=2,
        )
        == "Spectral validation failed: fatal spectral violation detected (caps_applied=0, max_caps=2)"
    )
