from __future__ import annotations

import torch
import torch.nn as nn

import invarlock.guards.spectral_measurement as spectral_measurement
from invarlock.guards.spectral import SpectralGuard
from invarlock.guards.spectral_measurement import compute_sigma_max, scan_model_gains


def test_compute_sigma_max_diagnoses_bad_tensor_types() -> None:
    diagnostics: list[dict[str, object]] = []

    sigma = compute_sigma_max("not-a-tensor", diagnostics=diagnostics)

    assert sigma == 1.0
    assert diagnostics[0]["kind"] == "spectral_sigma_fallback_non_tensor"
    assert diagnostics[0]["severity"] == "error"


def test_compute_sigma_max_diagnoses_quantized_weight() -> None:
    diagnostics: list[dict[str, object]] = []
    weight = torch.ones((2, 2), dtype=torch.int8)

    sigma = compute_sigma_max(weight, diagnostics=diagnostics, module_name="q")

    assert sigma == 1.0
    assert diagnostics == [
        {
            "kind": "spectral_sigma_fallback_quantized_weight",
            "severity": "warning",
            "message": "Spectral sigma measurement skipped a quantized weight.",
            "fallback_value": 1.0,
            "module": "q",
            "dtype": "torch.int8",
        }
    ]


def test_compute_sigma_max_diagnoses_non_finite_weights() -> None:
    diagnostics: list[dict[str, object]] = []
    weight = torch.tensor([[1.0, float("nan")], [2.0, 3.0]])

    sigma = compute_sigma_max(weight, diagnostics=diagnostics, module_name="bad")

    assert sigma == 0.0
    assert diagnostics[0]["kind"] == "spectral_sigma_fallback_non_finite_weight"
    assert diagnostics[0]["severity"] == "error"
    assert diagnostics[0]["module"] == "bad"


def test_compute_sigma_max_diagnoses_estimator_exception() -> None:
    diagnostics: list[dict[str, object]] = []

    def _raise(*_args: object, **_kwargs: object) -> float:
        raise RuntimeError("boom")

    sigma = compute_sigma_max(
        torch.eye(2),
        power_iter_sigma_max_fn=_raise,
        diagnostics=diagnostics,
        module_name="linear",
    )

    assert sigma == 1.0
    assert diagnostics[0]["kind"] == "spectral_sigma_fallback_estimator_error"
    assert diagnostics[0]["severity"] == "error"
    assert diagnostics[0]["error"] == "boom"


def test_compute_sigma_max_continues_when_finite_check_fails(monkeypatch) -> None:
    diagnostics: list[dict[str, object]] = []

    def _raise_isfinite(_weight: object) -> object:
        raise RuntimeError("finite unavailable")

    monkeypatch.setattr(spectral_measurement.torch, "isfinite", _raise_isfinite)

    sigma = compute_sigma_max(
        torch.eye(2),
        diagnostics=diagnostics,
        power_iter_sigma_max_fn=lambda *_args, **_kwargs: 2.0,
    )

    assert sigma == 2.0
    assert diagnostics == []


def test_compute_sigma_max_diagnoses_non_finite_estimate() -> None:
    diagnostics: list[dict[str, object]] = []

    sigma = compute_sigma_max(
        torch.eye(2),
        diagnostics=diagnostics,
        module_name="linear",
        power_iter_sigma_max_fn=lambda *_args, **_kwargs: float("inf"),
    )

    assert sigma == 1.0
    assert diagnostics[0]["kind"] == "spectral_sigma_fallback_non_finite_estimate"
    assert diagnostics[0]["observed_value"] == "inf"


def test_scan_model_gains_records_custom_estimator_fallback_diagnostic() -> None:
    class _Model:
        def named_modules(self):
            yield "linear", nn.Linear(2, 2)

    def _raise(_weight: object) -> float:
        raise RuntimeError("custom failed")

    result = scan_model_gains(
        _Model(),
        should_process_module_fn=lambda *_args: True,
        compute_sigma_max_fn=_raise,
    )

    assert result["scanned_modules"] == 1
    assert result["spectral_norms"] == [1.0]
    assert result["diagnostics"][0]["kind"] == (
        "spectral_sigma_fallback_custom_estimator_error"
    )
    assert result["diagnostics"][0]["error"] == "custom failed"


def test_record_guard_measurement_diagnostics_preserves_info_level() -> None:
    class _Guard:
        def __init__(self) -> None:
            self._measurement_diagnostics: list[dict[str, object]] = []
            self.events: list[dict[str, object]] = []

        def _log_event(self, kind: str, **details: object) -> None:
            self.events.append({"kind": kind, **details})

    guard = _Guard()

    spectral_measurement._record_guard_measurement_diagnostics(
        guard,
        [
            {
                "kind": "spectral_measurement_note",
                "severity": "info",
                "message": "noted",
                "fallback_value": 1.0,
            }
        ],
        phase="validate",
    )

    assert guard._measurement_diagnostics[0]["phase"] == "validate"
    assert guard.events[0]["level"] == "INFO"


def test_spectral_guard_carries_measurement_fallback_diagnostics() -> None:
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight[0, 0] = float("nan")
    guard = SpectralGuard()

    sigmas = guard._capture_sigmas(model, phase="validate")

    assert sigmas == {}
    assert any(
        item["kind"] == "spectral_sigma_fallback_non_finite_weight"
        for item in guard.diagnostic_records
    )
    assert any(
        item["kind"] == "spectral_sigma_fallback_non_finite_weight"
        for item in guard._measurement_diagnostics
    )
    assert guard.measurement_inventory["validate"]["excluded_modules"] == [
        {
            "module": "",
            "stage": "measurement",
            "reason": "non_finite_weight",
        }
    ]
