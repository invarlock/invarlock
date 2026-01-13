from __future__ import annotations

from collections.abc import Mapping


def rmt_decide(
    baseline_by_family: Mapping[str, float],
    current_by_family: Mapping[str, float],
    epsilon_by_family: Mapping[str, float],
) -> dict[str, object]:
    """
    Reference epsilon-rule decision for RMT activation edge-risk drift.

    For each family with baseline edge-risk > 0:
        PASS iff current_edge <= (1 + epsilon) * baseline_edge
    """
    families = set(baseline_by_family) | set(current_by_family) | set(epsilon_by_family)
    delta_by_family: dict[str, float] = {}
    allowed_by_family: dict[str, float] = {}
    for family in families:
        baseline = float(baseline_by_family.get(family, 0.0) or 0.0)
        current = float(current_by_family.get(family, 0.0) or 0.0)
        if baseline <= 0.0:
            continue
        epsilon_val = float(epsilon_by_family.get(family, 0.0) or 0.0)
        allowed = (1.0 + epsilon_val) * baseline
        allowed_by_family[family] = allowed
        delta_by_family[family] = (
            (current / baseline) - 1.0 if baseline > 0 else float("inf")
        )

    ok = all(
        float(current_by_family.get(family, 0.0) or 0.0) <= allowed_by_family[family]
        for family in allowed_by_family
    )
    return {
        "pass": ok,
        "delta_by_family": delta_by_family,
        "allowed_by_family": allowed_by_family,
    }
