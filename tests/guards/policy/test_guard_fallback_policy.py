from __future__ import annotations

from types import SimpleNamespace

import torch.nn as nn

from invarlock.guards import rmt_detection, spectral_measurement, variance_runtime


def test_spectral_numeric_fallback_records_structured_error() -> None:
    diagnostics: list[dict[str, object]] = []

    sigma = spectral_measurement.compute_sigma_max(
        object(),
        diagnostics=diagnostics,
        module_name="bad.weight",
    )

    assert sigma == 1.0
    assert diagnostics == [
        {
            "kind": "spectral_sigma_fallback_non_tensor",
            "severity": "error",
            "message": "Spectral sigma measurement received a non-tensor weight.",
            "fallback_value": 1.0,
            "module": "bad.weight",
            "observed_type": "object",
        }
    ]


def test_rmt_correction_failure_is_reported_as_error_event(
    monkeypatch,
) -> None:
    module = nn.Linear(2, 2)

    monkeypatch.setattr(
        rmt_detection.rmt_analysis,
        "layer_svd_stats",
        lambda *args, **kwargs: {
            "sigma_min": 0.1,
            "sigma_max": 9.0,
            "worst_ratio": 9.0,
        },
    )

    def _raise(*args, **kwargs) -> None:
        raise RuntimeError("svd failed")

    monkeypatch.setattr(rmt_detection, "_apply_rmt_correction", _raise)

    result = rmt_detection.step5_detect_and_correct_modules(
        [("layer.0", module)],
        baseline_sigmas={"layer.0": 1.0},
        baseline_mp_stats={
            "layer.0": {
                "sigma_base": 1.0,
                "mp_bulk_edge_base": 1.0,
            }
        },
        deadband=0.0,
        margin=1.5,
        correct=True,
    )

    assert result["has_outliers"] is True
    assert result["events"] == [
        {
            "operation": "rmt_correct_failed",
            "module_name": "layer.0",
            "error": "svd failed",
        }
    ]


def test_variance_unprepared_finalize_fails_closed_with_error_diagnostics() -> None:
    events: list[dict[str, object]] = []
    guard = SimpleNamespace(
        _prepared=False,
        _log_event=lambda kind, **kwargs: events.append({"kind": kind, **kwargs}),
    )

    result = variance_runtime.finalize_guard(guard, nn.Linear(2, 2))

    assert result["passed"] is False
    assert result["errors"] == ["Preparation failed or no target modules found"]
    assert any(item["severity"] == "error" for item in result["diagnostics"])
    assert events[0]["kind"] == "finalize_failed"
    assert events[0]["level"] == "ERROR"
