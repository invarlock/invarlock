from __future__ import annotations

from types import SimpleNamespace

import pytest

from invarlock.core.run_guard_overhead_policy import (
    _append_guard_overhead_diagnostic,
    _coerce_guard_overhead_diagnostics,
    build_guard_overhead_summary,
    finalize_guard_overhead_payload,
    normalize_guard_overhead_result,
    prepare_guard_overhead_report,
)


def test_finalize_guard_overhead_payload_collects_validator_fields() -> None:
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
        metrics={"overhead_ratio": 1.02, "overhead_percent": 2.0},
        passed=False,
    )
    payload = finalize_guard_overhead_payload(
        {"overhead_threshold": 0.01},
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
    assert payload["overhead_ratio"] == 1.02
    assert payload["overhead_percent"] == 2.0
    assert payload["passed"] is False
    assert payload["evaluated"] is True


def test_finalize_guard_overhead_payload_uses_fallback_metric_attrs() -> None:
    result = SimpleNamespace(
        metrics={},
        overhead_ratio=1.005,
        overhead_percent=0.5,
        passed=True,
    )
    payload = finalize_guard_overhead_payload(
        {},
        result,
    )
    assert payload["overhead_ratio"] == 1.005
    assert payload["overhead_percent"] == 0.5
    assert payload["passed"] is True


def test_build_guard_overhead_summary_formats_percent() -> None:
    summary = build_guard_overhead_summary(
        {"evaluated": True, "passed": False, "overhead_percent": 1.23},
        default_threshold=0.02,
    )
    assert summary.passed is False
    assert summary.overhead_percent == 1.23
    assert summary.overhead_ratio is None
    assert summary.threshold_fraction == 0.02


def test_build_guard_overhead_summary_falls_back_to_ratio_and_default() -> None:
    summary = build_guard_overhead_summary(
        {"evaluated": True, "passed": True, "overhead_ratio": 1.005},
        default_threshold="bad",  # type: ignore[arg-type]
    )
    assert summary.passed is True
    assert summary.overhead_percent is None
    assert summary.overhead_ratio == 1.005
    assert summary.threshold_fraction == 0.01


def test_build_guard_overhead_summary_marks_not_evaluated() -> None:
    summary = build_guard_overhead_summary(
        {"evaluated": False, "overhead_ratio": "bad"},
        default_threshold=0.02,
    )
    assert summary.evaluated is False
    assert summary.overhead_percent is None
    assert summary.overhead_ratio is None


def test_build_guard_overhead_summary_rejects_bool_numeric_fields() -> None:
    summary = build_guard_overhead_summary(
        {
            "evaluated": True,
            "passed": False,
            "overhead_percent": True,
            "overhead_ratio": True,
            "overhead_threshold": True,
        },
        default_threshold=0.02,
    )
    assert summary.overhead_percent is None
    assert summary.overhead_ratio is None
    assert summary.threshold_fraction == 0.02


def test_normalize_guard_overhead_result_marks_missing_ratio_as_not_evaluated() -> None:
    out = normalize_guard_overhead_result(None)
    assert out["evaluated"] is False
    assert out["passed"] is True


def test_normalize_guard_overhead_result_handles_float_coercion_failure() -> None:
    class BadInt(int):
        def __float__(self) -> float:
            raise TypeError("boom")

    out = normalize_guard_overhead_result({"overhead_ratio": BadInt(1)})
    assert out["evaluated"] is False
    assert out["passed"] is True


def test_normalize_guard_overhead_result_propagates_unexpected_ratio_errors() -> None:
    class BadInt(int):
        def __float__(self) -> float:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        normalize_guard_overhead_result({"overhead_ratio": BadInt(1)})


def test_finalize_guard_overhead_payload_drops_bool_metric_values() -> None:
    result = SimpleNamespace(
        metrics={"overhead_ratio": True, "overhead_percent": False},
        passed=False,
    )
    payload = finalize_guard_overhead_payload({}, result)
    assert payload["overhead_ratio"] is None
    assert payload["overhead_percent"] is None
    assert payload["evaluated"] is False
    assert payload["passed"] is True


def test_prepare_guard_overhead_report_returns_skipped_payload() -> None:
    payload = prepare_guard_overhead_report(
        {"skipped": True, "reason": "config"},
        resolved_loss_type="ppl_causal",
        core_report={},
        report={},
        default_threshold=0.01,
        extract_pm_snapshot_for_overhead_fn=lambda *_args, **_kwargs: {"final": 1.0},
        validate_guard_overhead_fn=lambda *_args, **_kwargs: object(),
    )
    assert payload == {"skipped": True, "reason": "config"}


def test_prepare_guard_overhead_report_validates_and_finalizes() -> None:
    class Result:
        diagnostics = (
            {
                "kind": "validation_info",
                "severity": "info",
                "message": "ok",
                "details": {},
            },
        )
        checks = {"guard_overhead": True}
        metrics = {"overhead_ratio": 1.005, "overhead_percent": 0.5}
        passed = True

    payload = prepare_guard_overhead_report(
        {
            "bare_report": {"metrics": {"primary_metric": {"final": 10.0}}},
            "overhead_threshold": 0.02,
        },
        resolved_loss_type="causal",
        core_report={"metrics": {}},
        report={"metrics": {}},
        default_threshold=0.01,
        extract_pm_snapshot_for_overhead_fn=lambda *_args, **_kwargs: {"final": 10.1},
        validate_guard_overhead_fn=lambda bare, guarded, overhead_threshold: Result(),
    )
    assert payload["guarded_report"] == {"metrics": {"primary_metric": {"final": 10.1}}}
    assert payload["passed"] is True
    assert payload["evaluated"] is True
    assert payload["diagnostics"][0]["message"] == "ok"


def test_guard_overhead_diagnostic_helpers_filter_invalid_records() -> None:
    diagnostics: list[dict[str, object]] = []

    _append_guard_overhead_diagnostic(
        diagnostics,
        severity="warning",
        message=123,
    )

    coerced = _coerce_guard_overhead_diagnostics(
        [
            "skip-me",
            {"message": ""},
            {"message": "default-severity", "severity": ""},
            {
                "kind": "guard_overhead_custom",
                "message": "kept",
                "details": "not-a-dict",
            },
        ]
    )

    assert diagnostics == [
        {
            "kind": "guard_overhead_warning",
            "severity": "warning",
            "message": "123",
            "details": {},
        }
    ]
    assert coerced == [
        {
            "kind": "guard_overhead_info",
            "severity": "info",
            "message": "default-severity",
            "details": {},
        },
        {
            "kind": "guard_overhead_custom",
            "severity": "info",
            "message": "kept",
            "details": {},
        },
    ]


def test_finalize_guard_overhead_payload_handles_non_mapping_metrics() -> None:
    result = SimpleNamespace(
        diagnostics=(),
        checks=(),
        metrics="not-a-dict",
        overhead_ratio=1.01,
        overhead_percent=1.0,
        passed=True,
    )

    payload = finalize_guard_overhead_payload({"warnings": ["old"]}, result)

    assert payload["checks"] == {}
    assert payload["overhead_ratio"] == 1.01
    assert payload["overhead_percent"] == 1.0
    assert payload["diagnostics"] == []


@pytest.mark.parametrize(
    ("loss_type", "expected_kind"),
    [
        ("mlm", "ppl_mlm"),
        ("seq2seq", "ppl_seq2seq"),
    ],
)
def test_prepare_guard_overhead_report_selects_loss_specific_metric_kind(
    loss_type: str,
    expected_kind: str,
) -> None:
    calls: list[tuple[object, str]] = []

    class Result:
        diagnostics = ()
        checks = {}
        metrics = {"overhead_ratio": 1.0}
        passed = True

    def extract_pm(source, *, kind):  # type: ignore[no-untyped-def]
        calls.append((source, kind))
        if source == {"metrics": {}}:
            return {}
        return {"kind": kind, "final": 9.0}

    payload = prepare_guard_overhead_report(
        {"bare_report": {}},
        resolved_loss_type=loss_type,
        core_report={"metrics": {}},
        report={"metrics": {"primary_metric": {"final": 9.0}}},
        default_threshold=0.01,
        extract_pm_snapshot_for_overhead_fn=extract_pm,
        validate_guard_overhead_fn=lambda *args, **kwargs: Result(),
    )

    assert calls == [
        ({"metrics": {}}, expected_kind),
        ({"metrics": {"primary_metric": {"final": 9.0}}}, expected_kind),
    ]
    assert payload["guarded_report"] == {
        "metrics": {"primary_metric": {"kind": expected_kind, "final": 9.0}}
    }


def test_prepare_guard_overhead_report_handles_snapshot_extraction_errors() -> None:
    class Result:
        diagnostics = ()
        checks = {}
        metrics = {}
        passed = True

    payload = prepare_guard_overhead_report(
        {"bare_report": {}},
        resolved_loss_type="causal",
        core_report={},
        report={},
        default_threshold=0.01,
        extract_pm_snapshot_for_overhead_fn=lambda *_args, **_kwargs: (
            _ for _ in ()
        ).throw(AttributeError("boom")),
        validate_guard_overhead_fn=lambda *args, **kwargs: Result(),
    )

    assert payload["guarded_report"] is None


def test_build_guard_overhead_summary_coerces_invalid_threshold_inputs() -> None:
    negative_default = build_guard_overhead_summary(
        {"overhead_threshold": "bad"},
        default_threshold=-1.0,
    )
    invalid_runtime_threshold = build_guard_overhead_summary(
        {"overhead_threshold": object()},
        default_threshold=0.02,
    )

    assert negative_default.threshold_fraction == 0.01
    assert invalid_runtime_threshold.threshold_fraction == 0.02
