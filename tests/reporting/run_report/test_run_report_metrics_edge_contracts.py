from __future__ import annotations

from types import SimpleNamespace

import pytest

from invarlock.reporting import run_report_metrics_contract as metrics_contract


def test_report_count_and_fraction_parsers_fail_closed() -> None:
    assert metrics_contract._coerce_report_count(True) is None
    assert metrics_contract._coerce_report_count(object()) is None
    value, violation = metrics_contract._coerce_fraction("fraction", object())
    assert value is None
    assert violation is not None
    assert violation.code == "E001"
    assert metrics_contract._coerce_finite_float(True) is None


def test_pairing_validation_rejects_collapsed_and_mismatched_schedule() -> None:
    violations = metrics_contract.validate_pairing_report_metrics(
        {
            "window_match_fraction": "bad",
            "window_overlap_fraction": 0.5,
            "window_pairing_reason": "different windows",
            "paired_windows": 0,
        },
        baseline_requested=True,
        profile="release",
        preview_count_report=1,
        final_count_report=2,
        expected_preview=2,
        expected_final=2,
    )
    assert len(violations) == 5
    assert all(violation.code == "E001" for violation in violations)


def test_dataset_window_stats_reject_malformed_fractions() -> None:
    with pytest.raises(ValueError, match="not a finite numeric fraction"):
        metrics_contract.build_dataset_window_stats(
            match_fraction="bad", overlap_fraction=None, window_plan=None
        )
    with pytest.raises(ValueError, match="not a finite numeric fraction"):
        metrics_contract.build_dataset_window_stats(
            match_fraction=None, overlap_fraction="bad", window_plan=None
        )


def test_pseudo_accuracy_authorization_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    assert metrics_contract._pseudo_accuracy_allowed("dev", None) is True
    monkeypatch.setenv("INVARLOCK_ALLOW_PSEUDO_ACCURACY", "yes")
    assert metrics_contract._pseudo_accuracy_allowed("release", None) is True
    monkeypatch.delenv("INVARLOCK_ALLOW_PSEUDO_ACCURACY")
    config = SimpleNamespace(context={"eval": {"allow_pseudo_accuracy": True}})
    assert metrics_contract._pseudo_accuracy_allowed("release", config) is True
    assert metrics_contract._pseudo_accuracy_allowed("release", None) is False


def test_classification_record_resolution_prefers_measured_records() -> None:
    assert metrics_contract._classification_records(None) == []
    assert metrics_contract._classification_records(
        {"records": [{"correct": True}], "example_correct": [False]}
    ) == [{"correct": True}]
    assert metrics_contract._classification_records(
        {"records": ["noise"], "example_correct": [True, "bad", 0]}
    ) == [{"correct": True}, {"correct": False}]
    assert metrics_contract._classification_records({"input_ids": "bad"}) == []
    assert metrics_contract._classification_records({"input_ids": [[1, 2], "bad"]}) == [
        {"input_ids": [1, 2]}
    ]


def test_loss_context_and_accuracy_count_derivation_are_strict() -> None:
    assert metrics_contract._loss_type_from_context(None) is None
    assert (
        metrics_contract._loss_type_from_context(
            SimpleNamespace(
                context={"eval": {"loss": {"resolved_type": "CLASSIFICATION"}}}
            )
        )
        == "classification"
    )
    assert metrics_contract._classification_counts_from_primary_metric(None) is None
    assert (
        metrics_contract._classification_counts_from_primary_metric(
            {"kind": "ppl", "preview": 1, "final": 1}
        )
        is None
    )
    assert (
        metrics_contract._classification_counts_from_primary_metric(
            {
                "kind": "accuracy",
                "preview": 0.5,
                "final": None,
                "n_preview": 2,
                "n_final": 2,
            }
        )
        is None
    )
    assert (
        metrics_contract._classification_counts_from_primary_metric(
            {
                "kind": "accuracy",
                "preview": -0.1,
                "final": 0.5,
                "n_preview": 2,
                "n_final": 2,
            }
        )
        is None
    )
    assert (
        metrics_contract._classification_counts_from_primary_metric(
            {"kind": "accuracy", "preview": 0.5, "final": 0.5, "n_preview": "bad"}
        )
        is None
    )
    assert (
        metrics_contract._classification_counts_from_primary_metric(
            {
                "kind": "accuracy",
                "preview": 0.5,
                "final": 0.3,
                "n_preview": 2,
                "n_final": 3,
            }
        )
        is None
    )
    assert (
        metrics_contract._classification_counts_from_primary_metric(
            {
                "kind": "accuracy",
                "preview": 0.3,
                "final": 0.5,
                "n_preview": 3,
                "n_final": 2,
            }
        )
        is None
    )
    assert metrics_contract._classification_counts_from_primary_metric(
        {
            "kind": "accuracy",
            "preview": 0.5,
            "final": 1.0,
            "n_preview": 2,
            "n_final": 2,
        }
    ) == (1, 2, 2, 2)


def test_existing_classification_and_window_fallbacks() -> None:
    core = SimpleNamespace(
        metrics={"classification": {"final": {"total": 2}}},
        evaluation_windows={"final": {"records": []}},
    )
    assert metrics_contract._existing_classification_metrics({}, core) == {
        "final": {"total": 2}
    }
    assert metrics_contract._existing_classification_metrics({}, None) is None
    assert metrics_contract._evaluation_windows({}, core) == {"final": {"records": []}}
    assert metrics_contract._evaluation_windows(
        {"evaluation_windows": {"preview": {}}}, None
    ) == {"preview": {}}


def test_fallback_classification_counts_reject_bad_config_counts() -> None:
    cfg = SimpleNamespace(dataset=SimpleNamespace(preview_n=object(), final_n=object()))
    assert metrics_contract._fallback_classification_counts(
        report={},
        core_report=None,
        cfg=cfg,
        preview_count_report=None,
        final_count_report=None,
    ) == (0, 0, 0, 0, False)


def test_enrich_classification_rejects_release_pseudo_counts() -> None:
    report = {"metrics": {}, "evaluation_windows": {}}
    cfg = SimpleNamespace(dataset=SimpleNamespace(preview_n=2, final_n=2))
    with pytest.raises(ValueError, match="pseudo accuracy is only allowed"):
        metrics_contract._enrich_classification_metrics(
            report=report,
            core_report=None,
            run_config=None,
            cfg=cfg,
            profile="release",
            preview_count_report=2,
            final_count_report=2,
        )
