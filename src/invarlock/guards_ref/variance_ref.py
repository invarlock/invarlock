from __future__ import annotations


def variance_decide(
    mean_delta: float,
    ci: tuple[float, float] | list[float],
    direction: str,  # "lower" or "higher" is better
    min_effect: float,
    predictive_one_sided: bool,
) -> dict[str, object]:
    """
    Reference predictive gate decision.

    For direction=="lower", negative deltas are improvements (Δ<0 better).
    For direction=="higher", flip sign so that improvements are treated consistently.
    """
    if not (isinstance(ci, tuple | list) and len(ci) == 2):
        return {"evaluated": False, "pass": True, "reason": "ci_unavailable"}
    lo, hi = float(ci[0]), float(ci[1])
    mu = float(mean_delta)
    me = float(min_effect or 0.0)

    dir_norm = (direction or "lower").strip().lower()
    # Normalize to "lower is better" frame
    if dir_norm == "higher":
        mu = -mu
        lo, hi = -hi, -lo

    # One-sided vs two-sided enablement semantics
    if predictive_one_sided:
        # Production parity: evaluate with one-sided criteria (no strict 0-exclusion required)
        evaluated = True
        if mu >= 0.0:
            return {
                "evaluated": evaluated,
                "pass": False,
                "reason": "mean_not_negative",
            }
        if me > 0.0 and (-mu) < me:
            return {
                "evaluated": evaluated,
                "pass": False,
                "reason": "gain_below_threshold",
            }
        if lo >= 0.0:
            return {"evaluated": evaluated, "pass": False, "reason": "ci_contains_zero"}
        return {"evaluated": evaluated, "pass": True, "reason": "ci_gain_met"}

    # Two-sided enablement requires strict exclusion of 0 and sufficient effect
    evaluated = (lo <= hi) and (abs(mu) >= me) and not (lo <= 0.0 <= hi)
    if not evaluated:
        return {"evaluated": False, "pass": True, "reason": "not_evaluated"}

    # Two-sided: require CI strictly below zero with gain >= min_effect
    if hi >= 0.0:
        return {"evaluated": True, "pass": False, "reason": "ci_contains_zero"}
    gain_lower = -hi
    if gain_lower < me:
        return {"evaluated": True, "pass": False, "reason": "gain_below_threshold"}
    return {"evaluated": True, "pass": True, "reason": "ci_gain_met"}
