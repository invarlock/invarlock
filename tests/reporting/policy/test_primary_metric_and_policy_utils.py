from __future__ import annotations

from invarlock.reporting.policy_utils import (
    _compute_thresholds_payload,
    _compute_variance_policy_digest,
    _format_epsilon_map,
    _format_family_caps,
    _resolve_policy_tier,
)
from invarlock.reporting.primary_metric_utils import attach_primary_metric


def _mk_report_with_pm() -> dict:
    return {
        "meta": {"model_id": "m"},
        "metrics": {
            "primary_metric": {"kind": "ppl_causal", "final": 10.0, "preview": 9.9}
        },
        "evaluation_windows": {
            "preview": {"logloss": [1.0, 1.2], "token_counts": [10, 20]},
            "final": {"logloss": [1.1, 1.3], "token_counts": [10, 20]},
        },
    }


def test_attach_primary_metric_present_and_fallback():
    cert = {"schema_version": "v1", "run_id": "rid"}
    rep = _mk_report_with_pm()
    base_ref = {"primary_metric": {"final": 10.0}}
    attach_primary_metric(
        cert,
        rep,
        baseline_raw=None,
        baseline_ref=base_ref,
        ppl_analysis={"unstable": True},
    )
    pm = cert["primary_metric"]
    assert pm.get("unstable") is True
    assert pm.get("display_ci") and isinstance(pm.get("display_ci"), list)
    # Fallback path (no pm in report): compute from report helpers
    cert2 = {"schema_version": "v1", "run_id": "rid"}
    rep2 = {"metrics": {}, "evaluation_windows": {}}
    attach_primary_metric(
        cert2, rep2, baseline_raw={}, baseline_ref={}, ppl_analysis=None
    )
    assert "primary_metric" in cert2


def test_policy_utils_helpers():
    d1 = _compute_variance_policy_digest({})
    d2 = _compute_variance_policy_digest({"deadband": 0.1, "min_abs_adjust": 0.0})
    assert d1 == "" and isinstance(d2, str) and len(d2) == 16
    payload = _compute_thresholds_payload(
        "balanced", {"variance": {"min_effect_lognll": 0.2}}
    )
    assert payload["tier"] == "balanced" and "pm_ratio" in payload
    assert (
        _resolve_policy_tier({"meta": {"auto": {"tier": "conservative"}}})
        == "conservative"
    )
    assert _format_family_caps({"attn": 1.2, "mlp": {"kappa": 0.8}}) == {
        "attn": {"kappa": 1.2},
        "mlp": {"kappa": 0.8},
    }
    assert _format_epsilon_map({"attn": 0.1}) == {"attn": 0.1}
