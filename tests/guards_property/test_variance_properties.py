from __future__ import annotations

from hypothesis import given

from invarlock.guards_ref.variance_ref import variance_decide
from tests.guards_property.strategies import variance_inputs


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
        # One-sided parity: evaluate regardless of 0 in CI; pass when mu indicates improvement and min_effect met
        if mu_n >= 0.0 or (me > 0 and (-mu_n) < me) or lo_n >= 0.0:
            assert r["evaluated"] is True and r["pass"] is False
    else:
        # Two-sided: If CI contains 0 or |mu| < min_effect => not evaluated
        if lo_n <= 0.0 <= hi_n or abs(mu_n) < me:
            assert r["evaluated"] is False and r["pass"] is True
