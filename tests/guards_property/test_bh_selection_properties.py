from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from invarlock.guards_ref.spectral_ref import bh_select


@given(
    st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=50),
    st.floats(min_value=1e-6, max_value=1.0),
)
def test_bh_selection_monotone(pvals, alpha):
    # If we decrease any p_i, number of rejections cannot decrease
    base = bh_select(pvals, alpha)
    if not pvals:
        return
    p2 = list(pvals)
    p2[0] = max(0.0, p2[0] * 0.5)
    changed = bh_select(p2, alpha)
    assert sum(changed) >= sum(base)
