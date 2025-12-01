import math

from invarlock.reporting.utils import (
    _coerce_int,
    _coerce_interval,
    _get_mapping,
    _sanitize_seed_bundle,
)


def test_coerce_int_variants_and_seed_sanitizer_corrupt_values():
    # Non-finite float returns None
    assert _coerce_int(float("inf")) is None
    assert _coerce_int(float("nan")) is None

    # Near-integer but not exactly an integer should return None
    assert _coerce_int(5.000000001) is None

    # Booleans are accepted as ints
    assert _coerce_int(True) == 1
    assert _coerce_int(False) == 0

    # Seed bundle: non-coercible string keeps fallback for that key
    out = _sanitize_seed_bundle(
        {"python": "abc", "numpy": None, "torch": 7.0}, fallback=3
    )
    # python stayed at fallback, numpy kept explicit None, torch coerced to int
    assert out["python"] == 3 and out["numpy"] is None and out["torch"] == 7


def test_coerce_interval_non_numeric_and_get_mapping_non_dict():
    lo, hi = _coerce_interval(["a", "b"])
    assert math.isnan(lo) and math.isnan(hi)

    # _get_mapping should return empty dict for non-dict sections
    src = {"meta": None, "context": []}
    assert _get_mapping(src, "meta") == {}
    assert _get_mapping(src, "context") == {}
