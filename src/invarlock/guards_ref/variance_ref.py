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
        evaluated = True
        if hi >= 0.0:
            return {
                "evaluated": evaluated,
                "pass": False,
                "reason": "ci_contains_zero",
            }
        if mu >= 0.0:
            return {
                "evaluated": evaluated,
                "pass": False,
                "reason": "mean_not_negative",
            }
        if hi > -me:
            return {
                "evaluated": evaluated,
                "pass": False,
                "reason": "gain_below_threshold",
            }
        if mu > -me:
            return {
                "evaluated": evaluated,
                "pass": False,
                "reason": "gain_below_threshold",
            }
        return {"evaluated": evaluated, "pass": True, "reason": "ci_gain_met"}

    evaluated = True
    if lo <= 0.0 <= hi:
        return {"evaluated": evaluated, "pass": False, "reason": "ci_contains_zero"}
    if lo > 0.0:
        if lo >= me and mu >= me:
            return {
                "evaluated": evaluated,
                "pass": False,
                "reason": "regression_detected",
            }
        return {"evaluated": evaluated, "pass": False, "reason": "mean_not_negative"}
    if hi > -me:
        return {
            "evaluated": evaluated,
            "pass": False,
            "reason": "gain_below_threshold",
        }
    if mu >= 0.0:
        return {"evaluated": evaluated, "pass": False, "reason": "mean_not_negative"}
    if mu > -me:
        return {
            "evaluated": evaluated,
            "pass": False,
            "reason": "gain_below_threshold",
        }
    return {"evaluated": evaluated, "pass": True, "reason": "ci_gain_met"}
