from __future__ import annotations

from hypothesis import given

from invarlock.guards.rmt import RMTGuard
from tests.guards.property.strategies import rmt_decide, rmt_inputs


def _production_violations(
    baseline: dict[str, float],
    current: dict[str, float],
    epsilon: dict[str, float],
) -> list[dict[str, object]]:
    guard = RMTGuard(epsilon_by_family=epsilon)
    guard.baseline_edge_risk_by_family = dict(baseline)
    guard.edge_risk_by_family = dict(current)
    return guard._compute_epsilon_violations()


@given(rmt_inputs())
def test_rmt_production_matches_reference_and_is_monotone_in_epsilon(data):
    bare, guarded, eps = data
    res0 = rmt_decide(bare, guarded, eps)
    violations0 = _production_violations(bare, guarded, eps)
    assert bool(res0["pass"]) is (not violations0)

    eps2 = {k: v * 2.0 for k, v in eps.items()}
    res1 = rmt_decide(bare, guarded, eps2)
    violations1 = _production_violations(bare, guarded, eps2)
    assert bool(res1["pass"]) is (not violations1)
    assert (not res0["pass"]) or res1["pass"]
    assert {str(item["family"]) for item in violations1}.issubset(
        {str(item["family"]) for item in violations0}
    )
