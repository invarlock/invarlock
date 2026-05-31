from __future__ import annotations

from invarlock.reporting.policy_utils import _build_resolved_policies


def test_resolved_policies_balanced_vs_aggressive():
    # Balanced tier: correction disabled, max_spectral_norm None
    spectral_bal = {"family_caps": {"ffn": {"kappa": 2.5}}, "policy": {}}
    rmt_bal = {"epsilon_by_family": {"ffn": 0.1}}
    variance_bal = {"predictive_gate": {}}
    res_bal = _build_resolved_policies("balanced", spectral_bal, rmt_bal, variance_bal)
    spb = res_bal.get("spectral", {})
    assert spb.get("correction_enabled") is False
    assert spb.get("max_spectral_norm") is None

    # Aggressive: policy turns on correction and sets max_spectral_norm; epsilon map preserved
    spectral_agg = {
        "family_caps": {"ffn": 3.0},
        "policy": {"correction_enabled": True, "max_spectral_norm": 1.23},
    }
    rmt_agg = {"epsilon_by_family": {"attn": 0.15, "ffn": 0.12}}
    variance_agg = {"predictive_gate": {"sided": "one_sided"}}
    res_agg = _build_resolved_policies(
        "aggressive", spectral_agg, rmt_agg, variance_agg
    )
    spa = res_agg.get("spectral", {})
    rma = res_agg.get("rmt", {})
    vara = res_agg.get("variance", {})
    assert spa.get("correction_enabled") is True
    assert isinstance(spa.get("max_spectral_norm"), int | float)
    assert rma.get("epsilon_by_family", {}).get("attn") == 0.15
    assert vara.get("predictive_one_sided") in {True, False}
