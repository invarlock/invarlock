from __future__ import annotations

import pytest
import torch

from invarlock.guards.spectral_control import (
    _is_matrix_weight,
    apply_relative_spectral_cap,
    apply_spectral_control,
    apply_weight_rescale,
)


class _Module:
    def __init__(self, weight, bias=None):
        self.weight = weight
        self.bias = bias


class _Model:
    def __init__(self, modules):
        self._modules = modules

    def named_modules(self):
        yield from self._modules.items()


class _BrokenWeight:
    ndim = 2
    dtype = torch.float32

    def mul_(self, _scale):
        raise RuntimeError("mul failed")


def test_apply_weight_rescale_tracks_skips_failures_and_bias_scaling() -> None:
    bias = torch.tensor([2.0])
    good = _Module(torch.tensor([[2.0, 0.0], [0.0, 1.0]]), bias=bias)
    broken = _Module(_BrokenWeight())
    quantized = _Module(torch.ones((2, 2), dtype=torch.int8))

    result = apply_weight_rescale(
        _Model({"good": good, "broken": broken, "quantized": quantized}),
        scale_factor=0.5,
        scope="all",
        should_process_module_fn=lambda name, *_args: name != "skip",
    )

    assert result["applied"] is True
    assert result["rescaled_modules"] == ["good"]
    assert result["failed_modules"][0][0] == "broken"
    assert torch.allclose(good.weight, torch.tensor([[1.0, 0.0], [0.0, 0.5]]))
    assert torch.allclose(bias, torch.tensor([1.0]))


def test_apply_weight_rescale_returns_error_when_model_iteration_fails() -> None:
    class BrokenModel:
        def named_modules(self):
            raise RuntimeError("boom")

    result = apply_weight_rescale(BrokenModel(), scale_factor=0.5)
    assert result["applied"] is False
    assert "boom" in result["error"]


def test_apply_weight_rescale_skips_modules_without_matrix_weight_and_without_bias() -> (
    None
):
    class NoWeightModule:
        pass

    no_bias = _Module(torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32))
    vector = _Module(torch.tensor([1.0, 2.0], dtype=torch.float32))

    result = apply_weight_rescale(
        _Model({"no_weight": NoWeightModule(), "vector": vector, "no_bias": no_bias}),
        scale_factor=0.5,
        scope="all",
        should_process_module_fn=lambda *_args: True,
    )

    assert result["applied"] is True
    assert result["rescaled_modules"] == ["no_bias"]
    assert torch.allclose(no_bias.weight, torch.tensor([[0.5, 0.0], [0.0, 1.0]]))


def test_apply_relative_spectral_cap_uses_baselines_and_collects_failures() -> None:
    good = _Module(torch.tensor([[3.0, 0.0], [0.0, 1.0]], dtype=torch.float32))
    broken = _Module(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32))
    quantized = _Module(torch.ones((2, 2), dtype=torch.int8))

    def fake_sigma(weight):
        if weight is broken.weight:
            raise RuntimeError("sigma failure")
        return 3.0 if weight is good.weight else 1.0

    result = apply_relative_spectral_cap(
        _Model({"good": good, "broken": broken, "quantized": quantized}),
        cap_ratio=1.0,
        scope="all",
        baseline_sigmas=None,
        capture_baseline_sigmas_fn=lambda *_args, **_kwargs: {
            "good": 1.5,
            "broken": 1.0,
        },
        compute_sigma_max_fn=fake_sigma,
    )

    assert result["applied"] is True
    assert result["capped_modules"][0]["module"] == "good"
    assert result["failed_modules"][0][0] == "broken"


def test_apply_relative_spectral_cap_skips_modules_without_matrix_weight() -> None:
    class NoWeightModule:
        pass

    vector = _Module(torch.tensor([1.0, 2.0], dtype=torch.float32))
    matrix = _Module(torch.tensor([[4.0, 0.0], [0.0, 1.0]], dtype=torch.float32))

    result = apply_relative_spectral_cap(
        _Model({"no_weight": NoWeightModule(), "vector": vector, "matrix": matrix}),
        cap_ratio=1.0,
        scope="all",
        baseline_sigmas={"matrix": 1.0},
        compute_sigma_max_fn=lambda weight: 4.0 if weight is matrix.weight else 1.0,
    )

    assert result["applied"] is True
    assert [entry["module"] for entry in result["capped_modules"]] == ["matrix"]


def test_apply_relative_spectral_cap_uses_explicit_predicate_and_skips_non_matrix() -> (
    None
):
    class NoWeightModule:
        pass

    vector = _Module(torch.tensor([1.0, 2.0], dtype=torch.float32))
    matrix = _Module(torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float32))

    result = apply_relative_spectral_cap(
        _Model({"no_weight": NoWeightModule(), "vector": vector, "matrix": matrix}),
        cap_ratio=1.0,
        scope="all",
        baseline_sigmas={"matrix": 1.0},
        should_process_module_fn=lambda *_args: True,
        capture_baseline_sigmas_fn=lambda *_args, **_kwargs: {"matrix": 1.0},
        compute_sigma_max_fn=lambda weight: 2.0 if weight is matrix.weight else 1.0,
    )

    assert result["applied"] is True
    assert [entry["module"] for entry in result["capped_modules"]] == ["matrix"]


def test_apply_relative_spectral_cap_returns_outer_error_when_baseline_capture_fails() -> (
    None
):
    result = apply_relative_spectral_cap(
        _Model({}),
        capture_baseline_sigmas_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("capture failed")
        ),
    )
    assert result["applied"] is False
    assert "capture failed" in result["error"]


def test_apply_spectral_control_handles_cap_rescale_and_exception() -> None:
    result = apply_spectral_control(
        _Model({}),
        {"scope": "all", "rescale_factor": 0.5},
        apply_relative_spectral_cap_fn=lambda *_args, **_kwargs: {
            "applied": True,
            "capped_modules": [{"module": "m"}],
            "failed_modules": [],
        },
        apply_weight_rescale_fn=lambda *_args, **_kwargs: {
            "applied": True,
            "rescaled_modules": ["m"],
        },
    )

    assert result["applied"] is True
    assert result["capping_applied"] is True
    assert result["rescaling_applied"] is True
    assert result["modules_processed"] == 1

    failed = apply_spectral_control(
        _Model({}),
        {"scope": "all"},
        apply_relative_spectral_cap_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("kaput")
        ),
    )
    assert failed["applied"] is False
    assert "kaput" in failed["error"]

    no_rescale = apply_spectral_control(
        _Model({}),
        {"scope": "all", "rescale_factor": 0.5},
        apply_relative_spectral_cap_fn=lambda *_args, **_kwargs: {
            "applied": False,
            "capped_modules": [],
            "failed_modules": [],
        },
        apply_weight_rescale_fn=lambda *_args, **_kwargs: {
            "applied": False,
            "rescaled_modules": [],
        },
    )
    assert no_rescale["applied"] is False
    assert no_rescale["rescaling_applied"] is False


def test_apply_spectral_control_reraises_unexpected_errors() -> None:
    class _ExplodingWeight:
        ndim = 2
        dtype = torch.float32

        def mul_(self, _scale):
            raise AssertionError("explode")

    with pytest.raises(AssertionError, match="explode"):
        apply_weight_rescale(
            _Model({"boom": _Module(_ExplodingWeight())}),
            scale_factor=0.5,
        )

    with pytest.raises(AssertionError, match="explode"):
        apply_spectral_control(
            _Model({}),
            {"scope": "all"},
            apply_relative_spectral_cap_fn=lambda *_args, **_kwargs: (
                _ for _ in ()
            ).throw(AssertionError("explode")),
        )


def test_is_matrix_weight_rejects_malformed_ndim_metadata() -> None:
    class _BadNdim:
        @property
        def ndim(self):
            return self

        def __int__(self):
            raise TypeError("bad ndim")

    assert _is_matrix_weight(_BadNdim()) is False
