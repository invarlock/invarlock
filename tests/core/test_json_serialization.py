from __future__ import annotations

import json

import pytest

from invarlock.json_serialization import (
    FiniteJsonError,
    dumps_finite_json,
    normalize_optional_nonfinite_json,
    require_finite_json,
)


def test_dumps_finite_json_produces_strict_standard_json() -> None:
    payload = {"metric": 1.0, "optional_ratio": None, "values": [2, 3.5]}

    encoded = dumps_finite_json(payload, sort_keys=True)

    assert (
        json.loads(encoded, parse_constant=lambda value: pytest.fail(value)) == payload
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_require_finite_json_reports_nested_nonfinite_path(value: float) -> None:
    with pytest.raises(
        FiniteJsonError,
        match=r"non-finite JSON number at \$\.metrics\.samples\[1\]\.ratio",
    ):
        require_finite_json({"metrics": {"samples": [1.0, {"ratio": value}]}})


def test_dumps_finite_json_cannot_be_overridden_to_allow_nan() -> None:
    with pytest.raises(FiniteJsonError):
        dumps_finite_json({"required_metric": float("nan")}, allow_nan=True)


def test_optional_nonfinite_normalization_is_explicit_and_non_mutating() -> None:
    payload = {"required": 1.0, "optional": [float("nan"), float("inf")]}

    normalized = normalize_optional_nonfinite_json(payload)

    assert normalized == {"required": 1.0, "optional": [None, None]}
    assert payload["optional"][0] != payload["optional"][0]
