from __future__ import annotations

from hypothesis import given

from tests.guards.property.strategies import rmt_decide, rmt_inputs


@given(rmt_inputs())
def test_rmt_monotone_epsilon(data):
    bare, guarded, eps = data
    res0 = rmt_decide(bare, guarded, eps)
    eps2 = {k: v * 2.0 for k, v in eps.items()}
    res1 = rmt_decide(bare, guarded, eps2)
    assert (not res0["pass"]) or res1["pass"]
