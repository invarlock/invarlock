from invarlock.reporting.certificate import _prepare_guard_overhead_section
from invarlock.reporting.policy_utils import (
    _extract_policy_overrides,
    _format_epsilon_map,
    _format_family_caps,
    _resolve_policy_tier,
)


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


def test_resolve_policy_tier_from_context_auto_and_guard_overhead_ratio_compute():
    report = {"context": {"auto": {"tier": "AGGRESSIVE"}}, "meta": {}}
    assert _resolve_policy_tier(report) == "aggressive"

    # Guard overhead: compute from bare/guarded
    raw = {"bare_ppl": 10.0, "guarded_ppl": 10.5, "overhead_threshold": 0.01}
    payload, passed = _prepare_guard_overhead_section(raw)
    assert (
        payload["evaluated"] is True
        and payload["overhead_ratio"] == 1.05
        and passed is False
    )
