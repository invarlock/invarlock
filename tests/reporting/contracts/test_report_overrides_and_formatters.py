import pytest

from invarlock.reporting.policy_utils import (
    _extract_policy_overrides,
    _format_epsilon_map,
    _format_family_caps,
    _resolve_policy_tier,
)
from invarlock.reporting.report_metric_impact import prepare_guard_metric_impact_section
from tests.reporting._support_guard_metric_impact import ppl_guard_context


def test_extract_policy_overrides_dedup_and_sources():
    report = {
        "meta": {
            "policy_overrides": ["a.yaml", "b.yaml"],
            "overrides": "c.yaml",
            "auto": {"overrides": ("d.yaml", "a.yaml")},
        },
        "config": {"overrides": ["e.yaml", None]},
    }
    out = _extract_policy_overrides(report)
    # Dedup preserves order of first occurrences
    assert out == ["a.yaml", "b.yaml", "c.yaml", "d.yaml", "e.yaml"]


def test_format_family_caps_and_epsilon_map_variants():
    caps = {"ffn": {"kappa": 2.5}, "attn": 3.1, "bad": "x"}
    out = _format_family_caps(caps)
    assert out["ffn"]["kappa"] == 2.5 and out["attn"]["kappa"] == 3.1

    eps = {"ffn": 0.1, "attn": 0.12, "bad": "x"}
    out_eps = _format_epsilon_map(eps)
    assert out_eps == {"ffn": 0.1, "attn": 0.12}


def test_resolve_policy_tier_from_canonical_meta_and_guard_metric_degradation_compute():
    report = {"meta": {"auto": {"tier": "AGGRESSIVE"}}}
    assert _resolve_policy_tier(report) == "aggressive"

    # Guard metric impact: compute from bare/guarded
    raw = ppl_guard_context(10.0, 10.5, degradation_limit=0.01)
    payload, passed = prepare_guard_metric_impact_section(raw)
    assert (
        payload["evaluated"] is True
        and payload["degradation"] == pytest.approx(0.05)
        and passed is False
    )
