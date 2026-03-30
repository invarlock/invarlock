from __future__ import annotations

from types import SimpleNamespace

from invarlock.reporting import report_overhead as overhead


class _ExplodingGetDict(dict):
    def get(self, key, default=None):  # type: ignore[override]
        if key in {"mode", "skip_reason"}:
            raise RuntimeError(f"boom:{key}")
        return super().get(key, default)


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

    monkeypatch.setattr(
        "invarlock.reporting.validate.validate_guard_overhead", _validate_stub
    )

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
