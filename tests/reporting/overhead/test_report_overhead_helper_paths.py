from __future__ import annotations

from types import SimpleNamespace

import pytest

import invarlock.reporting.validate as validate_mod
from invarlock.reporting import report_overhead as overhead


class _ExplodingGetDict(dict):
    def get(self, key, default=None):
        if key in {"mode", "skip_reason"}:
            raise RuntimeError(f"boom:{key}")
        return super().get(key, default)


class _DiagnosticObject:
    def __init__(self, *, details):
        self.kind = "attr_kind"
        self.severity = "warning"
        self.message = "from-object"
        self.details = details


class _LowerMetric:
    direction = "lower"


class _HigherMetric:
    direction = "higher"


def test_prepare_guard_overhead_imports_default_validator(monkeypatch) -> None:
    called: dict[str, bool] = {}

    def _validate_stub(_bare, _guarded, *, overhead_threshold):
        called["used"] = True
        assert overhead_threshold == 0.02
        return SimpleNamespace(
            metrics={
                "overhead_ratio": 1.01,
                "overhead_percent": 1.0,
                "bare_ppl": 100.0,
                "guarded_ppl": 101.0,
            },
            diagnostics=[
                {
                    "kind": "validation_info",
                    "severity": "info",
                    "message": "ok",
                    "details": {},
                }
            ],
            checks={"ratio_ok": True},
            passed=True,
        )

    monkeypatch.setattr(validate_mod, "validate_guard_overhead", _validate_stub)

    payload, passed = overhead.prepare_guard_overhead_section(
        {
            "overhead_threshold": 0.02,
            "bare_report": {"metrics": {}},
            "guarded_report": {"metrics": {}},
        }
    )
    assert called["used"] is True
    assert passed is True
    assert payload["evaluated"] is True
    assert payload["diagnostics"][0]["message"] == "ok"
    assert "messages" not in payload
    assert "warnings" not in payload
    assert "errors" not in payload


def test_prepare_guard_overhead_returns_empty_for_non_mapping_input() -> None:
    payload, passed = overhead.prepare_guard_overhead_section([])

    assert payload == {}
    assert passed is True


def test_normalize_guard_overhead_marks_unreadable_ratio_as_not_evaluated(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        overhead,
        "_coerce_non_bool_float",
        lambda _value: (_ for _ in ()).throw(TypeError("bad ratio")),
    )

    payload = overhead.normalize_guard_overhead_result({"overhead_ratio": 1.0})

    assert payload["evaluated"] is False
    assert payload["passed"] is True


def test_prepare_guard_overhead_handles_mode_skip_exceptions_and_preserves_errors() -> (
    None
):
    raw = _ExplodingGetDict(
        {
            "skipped": True,
            "diagnostics": [
                {
                    "kind": "validation_error",
                    "severity": "error",
                    "message": "already-present",
                    "details": {},
                }
            ],
        }
    )
    payload, passed = overhead.prepare_guard_overhead_section(raw)
    assert passed is True
    assert payload.get("skipped") is True
    assert payload.get("skip_reason") is None
    assert payload.get("mode") is None
    assert payload["diagnostics"][0]["severity"] == "error"
    assert payload["diagnostics"][0]["message"] == "already-present"
    assert payload.get("evaluated") is False


def test_prepare_guard_overhead_skipped_without_reason_or_mode_fallback() -> None:
    payload, passed = overhead.prepare_guard_overhead_section(
        {
            "mode": None,
            "guard_overhead_mode": None,
            "skipped": True,
            "skip_reason": 7,
        }
    )

    assert passed is True
    assert payload["skipped"] is True
    assert "mode" not in payload
    assert "skip_reason" not in payload
    assert payload["overhead_threshold"] == 0.01
    assert payload["threshold_percent"] == 1.0


def test_diagnostic_helpers_cover_append_and_coerce_variants() -> None:
    diagnostics: list[dict[str, object]] = []

    overhead._append_diagnostic(
        diagnostics,
        kind="added",
        severity="warning",
        message=123,
        details={"context": "unit"},
    )

    coerced = overhead._coerce_diagnostics(
        [
            {
                "kind": "mapping_kind",
                "severity": "error",
                "message": "from-mapping",
                "extra": 7,
            },
            _DiagnosticObject(details=["not-a-mapping"]),
            "plain diagnostic",
        ]
    )

    assert diagnostics == [
        {
            "kind": "added",
            "severity": "warning",
            "message": "123",
            "details": {"context": "unit"},
        }
    ]
    assert coerced == [
        {
            "kind": "mapping_kind",
            "severity": "error",
            "message": "from-mapping",
            "details": {"extra": 7},
        },
        {
            "kind": "attr_kind",
            "severity": "warning",
            "message": "from-object",
            "details": {},
        },
        {
            "kind": "guard_overhead_diagnostic",
            "severity": "info",
            "message": "plain diagnostic",
            "details": {},
        },
    ]


def test_prepare_guard_overhead_direct_ratio_mode_fallback_ignores_text_buckets() -> (
    None
):
    payload, passed = overhead.prepare_guard_overhead_section(
        {
            "overhead_threshold": 0.05,
            "guard_overhead_mode": " perplexity ",
            "skipped": True,
            "skip_reason": "  dataset too small  ",
            "bare_ppl": "100",
            "guarded_ppl": "110",
            "messages": ("info message",),
            "warnings": ["warn message"],
            "errors": ["error message"],
            "checks": {"ratio_ok": False},
        },
        validate_guard_overhead_fn=lambda *_args, **_kwargs: pytest.fail(
            "validator should not be called for direct-ratio inputs"
        ),
    )

    assert passed is False
    assert payload["mode"] == "perplexity"
    assert payload["skipped"] is True
    assert payload["skip_reason"] == "dataset too small"
    assert payload["bare_ppl"] == 100.0
    assert payload["guarded_ppl"] == 110.0
    assert payload["overhead_ratio"] == pytest.approx(1.1)
    assert payload["overhead_percent"] == pytest.approx(10.0)
    assert payload["checks"] == {"ratio_ok": False}
    assert payload["evaluated"] is True
    assert payload["diagnostics"] == []


def test_prepare_guard_overhead_zero_baseline_soft_passes_with_default_warning() -> (
    None
):
    payload, passed = overhead.prepare_guard_overhead_section(
        {
            "bare_ppl": 0,
            "guarded_ppl": 12,
            "messages": "not-a-sequence",
            "checks": ["not", "a", "dict"],
        }
    )

    assert passed is True
    assert payload["bare_ppl"] == 0.0
    assert payload["guarded_ppl"] == 12.0
    assert payload["checks"] == {}
    assert payload["evaluated"] is False
    assert payload["passed"] is True
    assert payload["diagnostics"] == [
        {
            "kind": "guard_overhead_unavailable",
            "severity": "warning",
            "message": "Guard overhead ratio unavailable",
            "details": {},
        }
    ]


def test_compute_quality_overhead_import_fallback_and_zero_baseline(
    monkeypatch,
) -> None:
    def _compute_stub(report, *, kind):
        assert kind == "ppl_causal"
        if report.get("role") == "bare":
            return {"final": 0.0}
        return {"final": 1.0}

    class _Metric:
        direction = "lower"

    monkeypatch.setattr(
        "invarlock.eval.primary_metric.compute_primary_metric_from_report",
        _compute_stub,
    )
    monkeypatch.setattr(
        "invarlock.eval.primary_metric.get_metric",
        lambda _kind: _Metric(),
    )

    out = overhead.compute_quality_overhead_from_guard(
        {
            "bare_report": {"role": "bare"},
            "guarded_report": {"role": "guarded"},
        },
        "ppl_causal",
    )
    assert out is None


def test_compute_quality_overhead_explicit_helpers_cover_ratio_delta_and_invalid_inputs() -> (
    None
):
    def _ratio_compute(report, *, kind):
        assert kind == "ppl_causal"
        return {"final": 3.0 if report["role"] == "guarded" else 2.0}

    def _delta_compute(report, *, kind):
        assert kind == "accuracy"
        return {"final": 0.91 if report["role"] == "guarded" else 0.84}

    assert (
        overhead.compute_quality_overhead_from_guard(
            "bad-guard",
            compute_primary_metric_from_report_fn=_ratio_compute,
            get_metric_fn=lambda _kind: _LowerMetric(),
        )
        is None
    )
    assert (
        overhead.compute_quality_overhead_from_guard(
            {"bare_report": {"role": "bare"}},
            compute_primary_metric_from_report_fn=_ratio_compute,
            get_metric_fn=lambda _kind: _LowerMetric(),
        )
        is None
    )
    assert (
        overhead.compute_quality_overhead_from_guard(
            {
                "bare_report": {"role": "bare"},
                "guarded_report": {"role": "guarded"},
            },
            " ",
            compute_primary_metric_from_report_fn=lambda _report, *, kind: {
                "final": "not-a-number"
            },
            get_metric_fn=lambda _kind: _LowerMetric(),
        )
        is None
    )

    ratio = overhead.compute_quality_overhead_from_guard(
        {
            "bare_report": {"role": "bare"},
            "guarded_report": {"role": "guarded"},
        },
        " ",
        compute_primary_metric_from_report_fn=_ratio_compute,
        get_metric_fn=lambda _kind: _LowerMetric(),
    )
    delta = overhead.compute_quality_overhead_from_guard(
        {
            "bare_report": {"role": "bare"},
            "guarded_report": {"role": "guarded"},
        },
        " accuracy ",
        compute_primary_metric_from_report_fn=_delta_compute,
        get_metric_fn=lambda _kind: _HigherMetric(),
    )

    assert ratio == {"basis": "ratio", "value": 1.5, "kind": "ppl_causal"}
    assert delta == {
        "basis": "delta_pp",
        "value": pytest.approx(7.0),
        "kind": "accuracy",
    }
