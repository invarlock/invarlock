from __future__ import annotations

from hypothesis import given

from tests.guards.property.strategies import variance_decide, variance_inputs


@given(variance_inputs())
def test_variance_idempotent(data):
    mu, ci, direction, me, one_sided = data
    r1 = variance_decide(mu, ci, direction, me, one_sided)
    r2 = variance_decide(mu, ci, direction, me, one_sided)
    assert r1 == r2


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
