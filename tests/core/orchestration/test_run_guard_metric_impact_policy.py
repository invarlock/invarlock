from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from invarlock.eval.guard_metric_impact import (
    build_guard_metric_bare_report,
    extract_guard_metric_arm_facts,
)
from invarlock.reporting.report_metric_impact import (
    _append_guard_metric_impact_diagnostic,
    _coerce_guard_metric_impact_diagnostics,
    build_guard_metric_impact_summary,
    finalize_guard_metric_impact_payload,
    normalize_guard_metric_impact_result,
    prepare_guard_metric_impact_report,
)


def test_finalize_guard_metric_impact_payload_collects_validator_fields() -> None:
    result = SimpleNamespace(
        diagnostics=(
            {
                "kind": "validation_info",
                "severity": "info",
                "message": "ok",
                "details": {},
            },
            {
                "kind": "validation_warning",
                "severity": "warning",
                "message": "warn",
                "details": {},
            },
            {
                "kind": "validation_error",
                "severity": "error",
                "message": "err",
                "details": {},
            },
        ),
        checks={"ratio_ok": False},
        metrics={"degradation": 1.02, "display_value": 2.0},
        passed=False,
    )
    payload = finalize_guard_metric_impact_payload(
        {"degradation_limit": 0.01},
        result,
    )
    assert payload["diagnostics"] == [
        {
            "kind": "validation_info",
            "severity": "info",
            "message": "ok",
            "details": {},
        },
        {
            "kind": "validation_warning",
            "severity": "warning",
            "message": "warn",
            "details": {},
        },
        {
            "kind": "validation_error",
            "severity": "error",
            "message": "err",
            "details": {},
        },
    ]
    assert payload["checks"] == {"ratio_ok": False}
    assert payload["degradation"] == 1.02
    assert payload["display_value"] == 2.0
    assert payload["passed"] is False
    assert payload["evaluated"] is True


def test_finalize_guard_metric_impact_payload_uses_fallback_metric_attrs() -> None:
    result = SimpleNamespace(
        metrics={},
        degradation=1.005,
        display_value=0.5,
        passed=True,
    )
    payload = finalize_guard_metric_impact_payload(
        {},
        result,
    )
    assert payload["degradation"] is None
    assert payload["display_value"] is None
    assert payload["passed"] is False
    assert payload["evaluated"] is False


def test_build_guard_metric_impact_summary_formats_percent() -> None:
    summary = build_guard_metric_impact_summary(
        {"evaluated": True, "passed": False, "display_value": 1.23},
        default_limit=0.02,
    )
    assert summary.passed is False
    assert summary.display_value == 1.23
    assert summary.degradation is None
    assert summary.degradation_limit == 0.02


def test_build_guard_metric_impact_summary_falls_back_to_degradation_and_default() -> (
    None
):
    summary = build_guard_metric_impact_summary(
        {"evaluated": True, "passed": True, "degradation": 1.005},
        default_limit="bad",
    )
    assert summary.passed is True
    assert summary.display_value is None
    assert summary.degradation == 1.005
    assert summary.degradation_limit == 0.01


def test_build_guard_metric_impact_summary_marks_not_evaluated() -> None:
    summary = build_guard_metric_impact_summary(
        {"evaluated": False, "degradation": "bad"},
        default_limit=0.02,
    )
    assert summary.evaluated is False
    assert summary.display_value is None
    assert summary.degradation is None


def test_build_guard_metric_impact_summary_rejects_bool_numeric_fields() -> None:
    summary = build_guard_metric_impact_summary(
        {
            "evaluated": True,
            "passed": False,
            "display_value": True,
            "degradation": True,
            "degradation_limit": True,
        },
        default_limit=0.02,
    )
    assert summary.display_value is None
    assert summary.degradation is None
    assert summary.degradation_limit == 0.02


def test_normalize_guard_metric_impact_result_marks_missing_degradation_as_not_evaluated() -> (
    None
):
    out = normalize_guard_metric_impact_result(None)
    assert out["evaluated"] is False
    assert out["passed"] is False


def test_normalize_guard_metric_impact_result_handles_float_coercion_failure() -> None:
    class BadInt(int):
        def __float__(self) -> float:
            raise TypeError("boom")

    out = normalize_guard_metric_impact_result({"degradation": BadInt(1)})
    assert out["evaluated"] is False
    assert out["passed"] is False


def test_normalize_guard_metric_impact_result_propagates_unexpected_degradation_errors() -> (
    None
):
    class BadInt(int):
        def __float__(self) -> float:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        normalize_guard_metric_impact_result({"degradation": BadInt(1)})


def test_finalize_guard_metric_impact_payload_drops_bool_metric_values() -> None:
    result = SimpleNamespace(
        metrics={"degradation": True, "display_value": False},
        passed=False,
    )
    payload = finalize_guard_metric_impact_payload({}, result)
    assert payload["degradation"] is None
    assert payload["display_value"] is None
    assert payload["evaluated"] is False
    assert payload["passed"] is False


def test_prepare_guard_metric_impact_report_returns_skipped_payload() -> None:
    payload = prepare_guard_metric_impact_report(
        {"skipped": True, "reason": "config"},
        resolved_loss_type="ppl_causal",
        core_report={},
        report={},
        default_limit=0.01,
        extract_pm_snapshot_for_metric_impact_fn=lambda *_args, **_kwargs: {
            "final": 1.0
        },
        validate_guard_metric_impact_fn=lambda *_args, **_kwargs: object(),
    )
    assert payload == {
        "skipped": True,
        "reason": "config",
        "evaluated": False,
        "passed": False,
        "mode": "skipped",
    }


def test_prepare_guard_metric_impact_report_validates_and_finalizes() -> None:
    class Result:
        diagnostics = (
            {
                "kind": "validation_info",
                "severity": "info",
                "message": "ok",
                "details": {},
            },
        )
        checks = {
            "metric_kind_matches": True,
            "measurements_valid": True,
            "guard_metric_impact": True,
        }
        metrics = {
            "metric_kind": "ppl_causal",
            "direction": "lower",
            "bare_value": 10.0,
            "guarded_value": 10.1,
            "degradation_basis": "relative_increase",
            "degradation": 0.01,
            "display_value": 1.0,
            "display_unit": "percent",
        }
        passed = True

    final_ids = [10, 11]
    bare_source = {
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "final": 10.0},
        },
        "evaluation_windows": {
            "final": {
                "window_ids": final_ids,
                "logloss": [math.log(10.0), math.log(10.0)],
                "token_counts": [1, 1],
            }
        },
    }
    guarded_source = {
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "final": 10.1},
        },
        "evaluation_windows": {
            "final": {
                "window_ids": final_ids,
                "logloss": [math.log(10.1), math.log(10.1)],
                "token_counts": [1, 1],
            }
        },
    }
    bare_envelope = build_guard_metric_bare_report(bare_source, "ppl_causal")
    bare_facts = extract_guard_metric_arm_facts(bare_source, "ppl_causal")
    assert bare_envelope is not None
    assert bare_facts is not None

    payload = prepare_guard_metric_impact_report(
        {
            "bare_report": bare_envelope,
            "bare_facts": bare_facts,
            "degradation_limit": 0.02,
        },
        resolved_loss_type="causal",
        core_report=guarded_source,
        report={"metrics": {}},
        default_limit=0.01,
        extract_pm_snapshot_for_metric_impact_fn=lambda *_args, **_kwargs: {
            "kind": "ppl_causal",
            "final": 10.1,
        },
        validate_guard_metric_impact_fn=lambda bare, guarded, degradation_limit: (
            Result()
        ),
    )
    assert "guarded_report" not in payload
    assert payload["passed"] is True
    assert payload["evaluated"] is True
    assert payload["checks"]["arm_facts_replay"] is True
    assert payload["bare_report"] == bare_envelope
    assert isinstance(payload.get("schedule_digest"), str)
    assert payload["diagnostics"][0]["message"] == "ok"


def test_guard_metric_impact_diagnostic_helpers_filter_invalid_records() -> None:
    diagnostics: list[dict[str, object]] = []

    _append_guard_metric_impact_diagnostic(
        diagnostics,
        severity="warning",
        message=123,
    )

    coerced = _coerce_guard_metric_impact_diagnostics(
        [
            "skip-me",
            {"message": ""},
            {"message": "default-severity", "severity": ""},
            {
                "kind": "guard_metric_impact_custom",
                "message": "kept",
                "details": "not-a-dict",
            },
        ]
    )

    assert diagnostics == [
        {
            "kind": "guard_metric_impact_warning",
            "severity": "warning",
            "message": "123",
            "details": {},
        }
    ]
    assert coerced == [
        {
            "kind": "guard_metric_impact_info",
            "severity": "info",
            "message": "default-severity",
            "details": {},
        },
        {
            "kind": "guard_metric_impact_custom",
            "severity": "info",
            "message": "kept",
            "details": {},
        },
    ]


def test_finalize_guard_metric_impact_payload_handles_non_mapping_metrics() -> None:
    result = SimpleNamespace(
        diagnostics=(),
        checks=(),
        metrics="not-a-dict",
        degradation=1.01,
        display_value=1.0,
        passed=True,
    )

    payload = finalize_guard_metric_impact_payload({"warnings": ["old"]}, result)

    assert payload["checks"] == {}
    assert payload["degradation"] is None
    assert payload["display_value"] is None
    assert payload["diagnostics"] == []


@pytest.mark.parametrize(
    ("loss_type", "expected_kind"),
    [
        ("mlm", "ppl_mlm"),
        ("seq2seq", "ppl_seq2seq"),
    ],
)
def test_prepare_guard_metric_impact_report_selects_loss_specific_metric_kind(
    loss_type: str,
    expected_kind: str,
) -> None:
    calls: list[tuple[object, str]] = []

    class Result:
        diagnostics = ()
        checks = {}
        metrics = {"degradation": 0.0}
        passed = True

    def extract_pm(source, *, kind):
        calls.append((source, kind))
        if source == {"metrics": {}}:
            return {}
        return {"kind": kind, "final": 9.0}

    payload = prepare_guard_metric_impact_report(
        {"bare_report": {}},
        resolved_loss_type=loss_type,
        core_report={"metrics": {}},
        report={"metrics": {"primary_metric": {"final": 9.0}}},
        default_limit=0.01,
        extract_pm_snapshot_for_metric_impact_fn=extract_pm,
        validate_guard_metric_impact_fn=lambda *args, **kwargs: Result(),
    )

    assert calls == [
        ({"metrics": {}}, expected_kind),
        ({"metrics": {"primary_metric": {"final": 9.0}}}, expected_kind),
    ]
    assert "guarded_report" not in payload
    assert payload["evaluated"] is False


def test_prepare_guard_metric_impact_report_handles_snapshot_extraction_errors() -> (
    None
):
    class Result:
        diagnostics = ()
        checks = {}
        metrics = {}
        passed = True

    payload = prepare_guard_metric_impact_report(
        {"bare_report": {}},
        resolved_loss_type="causal",
        core_report={},
        report={},
        default_limit=0.01,
        extract_pm_snapshot_for_metric_impact_fn=lambda *_args, **_kwargs: (
            _ for _ in ()
        ).throw(AttributeError("boom")),
        validate_guard_metric_impact_fn=lambda *args, **kwargs: Result(),
    )

    assert "guarded_report" not in payload
    assert payload["evaluated"] is False


def test_build_guard_metric_impact_summary_coerces_invalid_limit_inputs() -> None:
    negative_default = build_guard_metric_impact_summary(
        {"degradation_limit": "bad"},
        default_limit=-1.0,
    )
    invalid_runtime_limit = build_guard_metric_impact_summary(
        {"degradation_limit": object()},
        default_limit=0.02,
    )

    assert negative_default.degradation_limit == 0.01
    assert invalid_runtime_limit.degradation_limit == 0.02
