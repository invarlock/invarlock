from __future__ import annotations

import numpy as np
import torch

import invarlock.guards.spectral as spectral


def test_capture_sigmas_clamps_iters_defaults_init_and_skips_non_tensor(
    monkeypatch,
) -> None:
    guard = spectral.SpectralGuard()
    guard.estimator = {"iters": -1, "init": "bad"}

    calls: dict[str, object] = {}

    def fake_power_iter_sigma_max(weight: torch.Tensor, *, iters: int, init: str) -> float:
        calls["iters"] = iters
        calls["init"] = init
        return 2.0

    monkeypatch.setattr(spectral, "power_iter_sigma_max", fake_power_iter_sigma_max)

    class Mod:
        def __init__(self, weight):
            self.weight = weight

    class Model:
        def __init__(self):
            self._mods = {
                "int8": Mod(torch.zeros((2, 2), dtype=torch.int8)),
                "np": Mod(np.zeros((2, 2), dtype=float)),
                "fp": Mod(torch.eye(2)),
            }

        def named_modules(self):
            yield from self._mods.items()

    sigmas = guard._capture_sigmas(Model(), phase="after_edit")
    assert sigmas["int8"] == 1.0
    assert "np" not in sigmas
    assert sigmas["fp"] == 2.0
    assert calls["iters"] == 1
    assert calls["init"] == "ones"


def test_prepare_degeneracy_skips_non_tensor_and_invalid_sigma(monkeypatch) -> None:
    monkeypatch.setattr(
        spectral.SpectralGuard,
        "_capture_sigmas",
        lambda *_a, **_k: {"t": float("nan")},
        raising=False,
    )
    monkeypatch.setattr(
        spectral, "classify_model_families", lambda *_a, **_k: {"t": "ffn", "n": "ffn"}
    )
    monkeypatch.setattr(
        spectral, "compute_family_stats", lambda *_a, **_k: {"ffn": {"mean": 1.0, "std": 0.0}}
    )
    monkeypatch.setattr(spectral, "scan_model_gains", lambda *_a, **_k: {})
    monkeypatch.setattr(spectral, "auto_sigma_target", lambda *_a, **_k: 1.0)

    guard = spectral.SpectralGuard(degeneracy={"enabled": True})

    class Mod:
        def __init__(self, weight):
            self.weight = weight

    class Model:
        def named_modules(self):
            yield "n", Mod(np.zeros((2, 2), dtype=float))
            yield "t", Mod(torch.eye(2))

    out = guard.prepare(Model(), object(), None, {})
    assert out["ready"] is True
    assert guard.baseline_degeneracy == {}


def test_after_edit_applies_spectral_control_when_enabled(monkeypatch) -> None:
    guard = spectral.SpectralGuard(correction_enabled=True)
    guard.prepared = True
    guard.baseline_sigmas = {}
    guard.target_sigma = 1.0

    monkeypatch.setattr(
        spectral.SpectralGuard,
        "_capture_sigmas",
        lambda *_a, **_k: {"m": 1.0},
        raising=False,
    )
    monkeypatch.setattr(
        spectral.SpectralGuard,
        "_detect_spectral_violations",
        lambda *_a, **_k: [{"type": "mock_violation"}],
        raising=False,
    )
    monkeypatch.setattr(spectral, "apply_spectral_control", lambda *_a, **_k: {"applied": True})

    class Model:
        def named_modules(self):
            return iter([])

    guard.after_edit(Model())
    assert any(e.get("operation") == "spectral_control_applied" for e in guard.events)
