from __future__ import annotations

from types import SimpleNamespace

from invarlock.core.run_guard_overhead_policy import (
    build_guard_overhead_summary,
    finalize_guard_overhead_payload,
    normalize_guard_overhead_result,
)


def test_finalize_guard_overhead_payload_collects_validator_fields() -> None:
    result = SimpleNamespace(
        messages=("ok",),
        warnings=("warn",),
        errors=("err",),
        checks={"ratio_ok": False},
        metrics={"overhead_ratio": 1.02, "overhead_percent": 2.0},
        passed=False,
    )
    payload = finalize_guard_overhead_payload(
        {"overhead_threshold": 0.01},
        result,
    )
    assert payload["messages"] == ["ok"]
    assert payload["warnings"] == ["warn"]
    assert payload["errors"] == ["err"]
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
    assert summary.status == "FAIL"
    assert summary.overhead_display == "+1.23%"
    assert summary.threshold_fraction == 0.02
    assert summary.threshold_display == "≤ +2.0%"


def test_build_guard_overhead_summary_falls_back_to_ratio_and_default() -> None:
    summary = build_guard_overhead_summary(
        {"evaluated": True, "passed": True, "overhead_ratio": 1.005},
        default_threshold="bad",  # type: ignore[arg-type]
    )
    assert summary.status == "PASS"
    assert summary.overhead_display == "1.005x"
    assert summary.threshold_fraction == 0.01


def test_build_guard_overhead_summary_marks_not_evaluated() -> None:
    summary = build_guard_overhead_summary(
        {"evaluated": False, "overhead_ratio": "bad"},
        default_threshold=0.02,
    )
    assert summary.evaluated is False
    assert summary.overhead_display == "not evaluated"


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
