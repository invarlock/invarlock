from __future__ import annotations

import pytest

from invarlock.guards.rmt import RMTGuard
from invarlock.guards_ref.rmt_ref import rmt_decide


@pytest.mark.parametrize(
    ("baseline", "current", "epsilon"),
    [
        (
            {"ffn": 10.0, "attn": 5.0, "embed": 0.0},
            {"ffn": 12.0, "attn": 6.0, "embed": 9.0},
            {"ffn": 0.10, "attn": 0.10, "embed": 0.01},
        ),
        (
            {"ffn": 10.0, "attn": 5.0, "other": 2.0},
            {"ffn": 10.5, "attn": 5.2, "other": 2.0},
            {"ffn": 0.10, "attn": 0.10, "other": 0.0},
        ),
    ],
)
def test_rmt_decision_parity_production_vs_reference(
    baseline: dict[str, float],
    current: dict[str, float],
    epsilon: dict[str, float],
) -> None:
    ref = rmt_decide(baseline, current, epsilon)

    guard = RMTGuard(epsilon_by_family=epsilon)
    guard.baseline_edge_risk_by_family = dict(baseline)
    guard.edge_risk_by_family = dict(current)
    violations = guard._compute_epsilon_violations()

    prod_pass = not violations
    assert bool(ref["pass"]) == prod_pass
    ref_delta = ref["delta_by_family"]
    ref_allowed = ref["allowed_by_family"]
    expected_failing = {
        family
        for family in ref_allowed
        if current.get(family, 0.0) > ref_allowed[family]
    }
    assert {str(item["family"]) for item in violations} == expected_failing
    for item in violations:
        family = str(item["family"])
        assert item["allowed"] == pytest.approx(ref_allowed[family])
        assert item["delta"] == pytest.approx(ref_delta[family])
