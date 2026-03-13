from __future__ import annotations

from types import SimpleNamespace

import torch

import invarlock.guards.spectral_measurement as spectral_measurement
from invarlock.guards.spectral_measurement import (
    auto_sigma_target,
    capture_baseline_sigmas,
    capture_sigmas,
    compute_sigma_max,
    scan_model_gains,
)


class _Module:
    def __init__(self, weight):
        self.weight = weight


class _Model:
    def __init__(self, modules):
        self._modules = modules

    def named_modules(self):
        yield from self._modules.items()


def test_auto_sigma_target_ignores_non_positive_sigmas() -> None:
    model = _Model({"zero": _Module(torch.eye(2))})

    target = auto_sigma_target(
        model,
        percentile=0.9,
        compute_sigma_max_fn=lambda _weight: 0.0,
    )

    assert target == 0.9


def test_capture_baseline_sigmas_skips_out_of_scope_modules() -> None:
    model = _Model({"kept": _Module(torch.eye(2)), "skipped": _Module(torch.eye(2))})

    sigmas = capture_baseline_sigmas(
        model,
        scope="all",
        should_process_module_fn=lambda name, *_args: name == "kept",
        compute_sigma_max_fn=lambda _weight: 1.5,
    )

    assert sigmas == {"kept": 1.5}


def test_scan_model_gains_reports_when_no_modules_are_scanned() -> None:
    model = _Model({"skipped": _Module(torch.eye(2))})

    result = scan_model_gains(
        model,
        scope="all",
        should_process_module_fn=lambda *_args: False,
        compute_sigma_max_fn=lambda _weight: 2.0,
    )

    assert result["total_layers"] == 1
    assert result["scanned_modules"] == 0
    assert result["message"].startswith("Scanned 0 modules")


def test_scan_model_gains_ignores_weight_stat_failures() -> None:
    class _BadStatsWeight:
        ndim = 2

        def mean(self):
            raise RuntimeError("bad mean")

        def std(self):
            raise RuntimeError("bad std")

        def min(self):
            raise RuntimeError("bad min")

        def max(self):
            raise RuntimeError("bad max")

    model = _Model({"fragile": _Module(_BadStatsWeight())})

    result = scan_model_gains(
        model,
        scope="all",
        should_process_module_fn=lambda *_args: True,
        compute_sigma_max_fn=lambda _weight: 2.5,
    )

    assert result["scanned_modules"] == 1
    assert result["spectral_norms"] == [2.5]
    assert result["weight_statistics"] == {}


def test_scan_model_gains_returns_error_payload_when_model_iteration_fails() -> None:
    class _BrokenModel:
        def named_modules(self):
            raise RuntimeError("boom")

    result = scan_model_gains(_BrokenModel(), scope="all")

    assert result["scanned_modules"] == 0
    assert result["error"] == "boom"
    assert "failed" in result["message"].lower()


def test_compute_sigma_max_uses_explicit_power_iter_callable() -> None:
    seen: list[tuple[int, str]] = []

    sigma = compute_sigma_max(
        torch.eye(2),
        iters="7",
        init="e0",
        power_iter_sigma_max_fn=lambda weight, *, iters, init: (
            seen.append((iters, init)),
            float(weight.sum().item()),
        )[1],
    )

    assert sigma == 2.0
    assert seen == [(7, "e0")]


def test_capture_baseline_sigmas_skips_non_matrix_weights_even_when_in_scope() -> None:
    model = _Model(
        {
            "vector": _Module(torch.ones(3)),
            "missing_weight": SimpleNamespace(),
        }
    )

    sigmas = capture_baseline_sigmas(
        model,
        scope="all",
        should_process_module_fn=lambda *_args: True,
        compute_sigma_max_fn=lambda _weight: 9.9,
    )

    assert sigmas == {}


def test_scan_model_gains_skips_non_matrix_weights_even_when_in_scope() -> None:
    model = _Model(
        {
            "vector": _Module(torch.ones(3)),
            "missing_weight": SimpleNamespace(),
        }
    )

    result = scan_model_gains(
        model,
        scope="all",
        should_process_module_fn=lambda *_args: True,
        compute_sigma_max_fn=lambda _weight: 7.7,
    )

    assert result["total_layers"] == 2
    assert result["scanned_modules"] == 0
    assert result["spectral_norms"] == []
    assert result["weight_statistics"] == {}
    assert result["message"].startswith("Scanned 0 modules")


def test_capture_sigmas_falls_back_for_invalid_estimator_and_power_iter_failure(
    monkeypatch,
) -> None:
    guard = SimpleNamespace(
        estimator={"iters": "bad", "init": "bad"},
        _should_check_module=lambda *_args: True,
    )

    class _ModelWithWeight:
        def named_modules(self):
            yield "fragile", _Module(torch.eye(2))

    monkeypatch.setattr(
        spectral_measurement,
        "power_iter_sigma_max",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("kaput")),
    )

    sigmas = capture_sigmas(guard, _ModelWithWeight(), phase="after_edit")

    assert sigmas == {"fragile": 1.0}
