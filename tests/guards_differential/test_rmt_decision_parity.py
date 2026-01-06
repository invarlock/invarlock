from __future__ import annotations

from invarlock.guards.rmt import RMTGuard
from invarlock.guards_ref.rmt_ref import rmt_decide


def test_rmt_decision_parity_simple():
    bare = {"ffn": 10.0, "attn": 5.0}
    guarded = {"ffn": 12.0, "attn": 6.0}
    eps = {"ffn": 0.10, "attn": 0.10}

    ref = rmt_decide(bare, guarded, eps)

    g = RMTGuard(epsilon=eps)
    g.baseline_edge_risk_by_family = dict(bare)
    g.edge_risk_by_family = dict(guarded)
    violations = g._compute_epsilon_violations()
    # Build production parity decision
    prod_pass = len(violations) == 0
    # Compare pass/fail
    assert bool(ref["pass"]) == prod_pass
