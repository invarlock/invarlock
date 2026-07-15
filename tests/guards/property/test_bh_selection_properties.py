from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from invarlock.guards.spectral import bh_reject_families
from tests.guards.property.strategies import bh_select


@given(
    st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=50),
    st.floats(min_value=1e-6, max_value=1.0),
)
def test_bh_selection_monotone(pvals, alpha):
    family_pvalues = {f"family-{index}": value for index, value in enumerate(pvals)}
    reference = bh_select(pvals, alpha)
    production = bh_reject_families(
        family_pvalues,
        alpha=alpha,
        m=len(family_pvalues),
    )
    expected = {
        f"family-{index}" for index, rejected in enumerate(reference) if rejected
    }
    assert production == expected

    # If we decrease any p_i, the production rejection count cannot decrease.
    if not pvals:
        return
    p2 = list(pvals)
    p2[0] = max(0.0, p2[0] * 0.5)
    changed = bh_reject_families(
        {f"family-{index}": value for index, value in enumerate(p2)},
        alpha=alpha,
        m=len(p2),
    )
    assert len(changed) >= len(production)
