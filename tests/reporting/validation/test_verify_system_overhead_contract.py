from __future__ import annotations

import math

from invarlock.reporting.verify_system_overhead import validate_system_overhead


def test_system_overhead_accepts_absence_and_exact_measurements() -> None:
    assert validate_system_overhead({}) == []
    assert (
        validate_system_overhead(
            {
                "system_overhead": {
                    "latency": {
                        "baseline": 2.0,
                        "edited": 3.0,
                        "delta": 1.0,
                        "ratio": 1.5,
                    },
                    "standalone": {"edited": 4.0},
                }
            }
        )
        == []
    )


def test_system_overhead_rejects_malformed_container_and_entries() -> None:
    assert validate_system_overhead({"system_overhead": []}) == [
        "system_overhead must be an object."
    ]
    assert validate_system_overhead({"system_overhead": {"latency": 1}}) == [
        "system_overhead.latency must be a structured entry."
    ]


def test_system_overhead_rejects_nonfinite_and_unbound_derivations() -> None:
    errors = validate_system_overhead(
        {
            "system_overhead": {
                "bool": {"edited": True},
                "nan": {"edited": math.nan},
                "unbound": {"edited": 1.0, "delta": 0.0, "ratio": 1.0},
            }
        }
    )
    assert errors == [
        "system_overhead.bool.edited must be finite.",
        "system_overhead.nan.edited must be finite.",
        "system_overhead.unbound cannot declare delta or ratio without baseline.",
    ]


def test_system_overhead_rejects_forged_delta_ratio_and_zero_ratio() -> None:
    errors = validate_system_overhead(
        {
            "system_overhead": {
                "forged": {
                    "baseline": 2.0,
                    "edited": 3.0,
                    "delta": "bad",
                    "ratio": 99.0,
                },
                "zero": {
                    "baseline": 0.0,
                    "edited": 0.0,
                    "delta": 0.0,
                    "ratio": 0.0,
                },
            }
        }
    )
    assert errors == [
        "system_overhead.forged.delta does not match edited-baseline.",
        "system_overhead.forged.ratio does not match edited/baseline.",
        "system_overhead.zero.ratio is undefined for a zero baseline.",
    ]
