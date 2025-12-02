from __future__ import annotations

from invarlock.reporting import certificate as C


def test_prepare_guard_overhead_section_direct_ratio_and_lists() -> None:
    raw = {
        "overhead_threshold": "0.02",
        "bare_final": 10.0,
        "guarded_final": 10.5,
        "messages": ["ok", 123],
        "warnings": ["warn"],
        "errors": [],
    }
    sanitized, passed = C._prepare_guard_overhead_section(raw)
    assert isinstance(sanitized, dict)
    assert 0.01 < sanitized["overhead_threshold"] < 0.03
    assert "overhead_ratio" in sanitized and "overhead_percent" in sanitized
    assert isinstance(sanitized.get("messages"), list)
    assert passed in {True, False}
