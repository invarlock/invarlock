from __future__ import annotations

from hypothesis import given

from invarlock.guards.variance_policy import predictive_gate_outcome
from tests.guards.property.strategies import variance_decide, variance_inputs


@given(variance_inputs())
def test_variance_production_decision_matches_reference(data):
    mu, ci, direction, me, one_sided = data
    reference = variance_decide(mu, ci, direction, me, one_sided)
    production_mu = mu
    production_ci = ci
    if direction == "higher":
        lo, hi = ci
        production_mu = -mu
        production_ci = (-hi, -lo)

    production_pass, production_reason = predictive_gate_outcome(
        production_mu,
        production_ci,
        me,
        one_sided,
    )

    assert reference["evaluated"] is True
    assert bool(reference["pass"]) is production_pass
    assert reference["reason"] == production_reason


@given(variance_inputs())
def test_variance_enablement(data):
    mu, ci, direction, me, one_sided = data
    lo, hi = ci
    # Normalize to the same frame as the reference: "lower is better"
    mu_n, lo_n, hi_n = mu, lo, hi
    if str(direction).lower() == "higher":
        mu_n = -mu
        lo_n, hi_n = -hi, -lo
    r = variance_decide(mu, ci, direction, me, one_sided)
    if one_sided:
        assert r["evaluated"] is True
        if hi_n >= 0.0 or mu_n >= 0.0 or hi_n > -me or mu_n > -me:
            assert r["evaluated"] is True and r["pass"] is False
    else:
        assert r["evaluated"] is True
        if lo_n <= 0.0 <= hi_n or lo_n > 0.0 or hi_n > -me or mu_n >= 0.0 or mu_n > -me:
            assert r["pass"] is False
