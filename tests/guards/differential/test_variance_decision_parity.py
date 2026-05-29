from __future__ import annotations

from invarlock.guards.variance_policy import predictive_gate_outcome
from invarlock.guards_ref.variance_ref import variance_decide


def test_variance_decision_parity_cases():
    cases = [
        (-0.003, (-0.01, -0.001), "lower", 0.0009, False),
        (-0.0001, (-0.001, 0.001), "lower", 0.0005, True),
        (0.002, (-0.001, 0.004), "lower", 0.0005, False),
        (-0.002, (-0.003, -0.0015), "higher", 0.001, False),
        (-0.003, (-0.004, -0.002), "lower", 0.001, True),
        (0.003, (0.002, 0.004), "lower", 0.001, False),
    ]
    for mu, ci, direction, me, one_sided in cases:
        ref = variance_decide(mu, ci, direction, me, one_sided)
        mu_prod, ci_prod = mu, ci
        # production expects lower-is-better ΔlogNLL semantics
        if direction == "higher":
            mu_prod = -mu_prod
            lo, hi = ci
            ci_prod = (-hi, -lo)
        prod_pass, prod_reason = predictive_gate_outcome(
            mu_prod, ci_prod, me, one_sided
        )
        assert ref["evaluated"] is True
        assert bool(ref["pass"]) == bool(prod_pass)
        assert str(ref["reason"]) == prod_reason
