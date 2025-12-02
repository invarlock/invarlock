from __future__ import annotations

from invarlock.reporting.certificate import (
    _format_epsilon_map,
    _format_family_caps,
    _normalize_override_entry,
)


def test_normalize_override_entry_set_and_none():
    assert _normalize_override_entry({"a", "b"})
    assert _normalize_override_entry(None) == []


def test_formatters_numeric_and_filtering():
    caps = {"ffn": 2.5, "attn": {"kappa": 3.1}, "bad": "x"}
    formatted = _format_family_caps(caps)
    assert formatted.get("ffn", {}).get("kappa") == 2.5
    assert formatted.get("attn", {}).get("kappa") == 3.1

    eps = {"ffn": 0.1, "attn": "bad"}
    eps_fmt = _format_epsilon_map(eps)
    assert "ffn" in eps_fmt and "attn" not in eps_fmt
