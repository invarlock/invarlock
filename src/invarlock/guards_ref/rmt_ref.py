from __future__ import annotations

import math
from collections.abc import Mapping


def rmt_decide(
    bare_by_family: Mapping[str, int],
    guarded_by_family: Mapping[str, int],
    epsilon_by_family: Mapping[str, float],
) -> dict[str, object]:
    """
    Reference epsilon-rule decision for RMT.

    Allowed excess A_f = ceil(epsilon_f * max(1, b_f)).
    PASS iff for all families Δ_f <= A_f and sum Δ_f <= sum A_f.
    """
    families = set(bare_by_family) | set(guarded_by_family) | set(epsilon_by_family)
    delta_by_family: dict[str, int] = {}
    allowed_by_family: dict[str, int] = {}
    sum_delta = 0
    sum_allowed = 0
    for f in families:
        b = int(bare_by_family.get(f, 0) or 0)
        g = int(guarded_by_family.get(f, 0) or 0)
        eps = float(epsilon_by_family.get(f, 0.0) or 0.0)
        d = g - b
        a = int(math.ceil(eps * max(1, b)))
        delta_by_family[f] = d
        allowed_by_family[f] = a
        sum_delta += d
        sum_allowed += a
    ok = all(delta_by_family[f] <= allowed_by_family[f] for f in families) and (
        sum_delta <= sum_allowed
    )
    return {
        "pass": ok,
        "delta_by_family": delta_by_family,
        "allowed_by_family": allowed_by_family,
    }
