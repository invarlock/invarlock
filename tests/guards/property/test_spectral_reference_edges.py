from __future__ import annotations

from typing import Any, cast

from tests.guards.property.strategies import _finite01, spectral_decide


def test_spectral_decide_invalid_alpha_falls_back_to_default() -> None:
    inputs = (
        {"layer.0": 1.3},
        {"layer.0": 1.0},
        {"layer.0": "mlp"},
        0.05,
        {"mlp": 1.0},
    )

    expected = spectral_decide(*inputs, {"method": "bonferroni", "alpha": 0.05})
    actual = spectral_decide(*inputs, {"method": "bonferroni", "alpha": object()})

    assert actual == expected


def test_finite01_rejects_non_numeric_objects() -> None:
    assert _finite01(cast(Any, object())) is False
