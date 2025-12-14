from __future__ import annotations

import math
from typing import Any


def _extract_guard(report: dict[str, Any], name: str) -> dict[str, Any] | None:
    guards = report.get("guards")
    if isinstance(guards, list):
        for item in guards:
            if isinstance(item, dict) and item.get("name") == name:
                return item
    return None


def _coerce_delta_ci(value: Any) -> tuple[float, float] | None:
    if not (isinstance(value, tuple | list) and len(value) == 2):
        return None
    try:
        lo = float(value[0])
        hi = float(value[1])
    except Exception:
        return None
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    return (lo, hi)


def _gain_lower_bound(
    *, mean_delta: float | None, delta_ci: tuple[float, float] | None, one_sided: bool
) -> float:
    if delta_ci is None:
        return 0.0
    lo, hi = delta_ci
    if hi >= 0.0:
        return 0.0
    if one_sided and (mean_delta is None or not (mean_delta < 0.0)):
        return 0.0
    # Gain CI lower bound is -upper (worst-case gain).
    return max(0.0, -hi)


def _recommend_threshold_for_target_rate(
    gains: list[float],
    *,
    target_rate: float,
    safety_margin: float,
) -> tuple[float, float]:
    n = len(gains)
    if n <= 0:
        return 0.0, 0.0
    target = float(target_rate)
    if not (0.0 <= target <= 1.0):
        target = 0.05
    desired_passes = int(math.floor(target * n))
    gains_desc = sorted((max(0.0, float(g)) for g in gains), reverse=True)

    def pass_count(thr: float) -> int:
        return sum(1 for g in gains_desc if g >= thr)

    if desired_passes <= 0:
        thr = (
            (gains_desc[0] * (1.0 + max(0.0, safety_margin)))
            if gains_desc[0] > 0
            else 0.0
        )
        return float(round(thr, 3)), 0.0

    unique_vals = sorted(set(gains_desc), reverse=True)
    chosen = None
    chosen_rate = 0.0
    for val in unique_vals:
        cnt = pass_count(val)
        rate = cnt / float(n)
        if cnt <= desired_passes:
            chosen = float(val)
            chosen_rate = float(rate)
            break

    if chosen is None:
        # Ties at max prevent meeting desired_passes; force zero enable rate.
        thr = gains_desc[0] * (1.0 + max(0.0, safety_margin))
        return float(round(thr, 3)), 0.0

    thr = chosen * (1.0 + max(0.0, safety_margin))
    return float(round(thr, 3)), float(chosen_rate)


def summarize_ve_sweep_reports(
    reports: list[dict[str, Any]],
    *,
    tier: str,
    target_enable_rate: float = 0.05,
    safety_margin: float = 0.0,
    predictive_one_sided: bool = True,
) -> dict[str, Any]:
    """Summarize VE predictive-gate sweeps and recommend min_effect_lognll."""

    tier_norm = (tier or "").strip().lower() or "balanced"
    one_sided = bool(predictive_one_sided)
    margin = float(safety_margin or 0.0)
    if not (0.0 <= margin <= 1.0):
        margin = 0.0

    gains: list[float] = []
    widths: list[float] = []
    evaluated = 0

    for report in reports:
        if not isinstance(report, dict):
            continue
        g = _extract_guard(report, "variance") or {}
        metrics = g.get("metrics", {}) if isinstance(g.get("metrics"), dict) else {}
        pg = metrics.get("predictive_gate")
        if not isinstance(pg, dict):
            continue
        evaluated += 1 if pg.get("evaluated") is True else 0

        delta_ci = _coerce_delta_ci(pg.get("delta_ci"))
        mean_delta = pg.get("mean_delta")
        try:
            mean_delta_f = float(mean_delta) if mean_delta is not None else None
        except Exception:
            mean_delta_f = None

        gains.append(
            _gain_lower_bound(
                mean_delta=mean_delta_f, delta_ci=delta_ci, one_sided=one_sided
            )
        )
        if delta_ci is not None:
            widths.append(float(abs(delta_ci[1] - delta_ci[0])))

    recommended, expected_rate = _recommend_threshold_for_target_rate(
        gains, target_rate=target_enable_rate, safety_margin=margin
    )

    mean_width = float(sum(widths) / max(len(widths), 1)) if widths else None

    return {
        "tier": tier_norm,
        "n_runs": int(len(gains)),
        "observed": {
            "evaluated_runs": int(evaluated),
            "mean_ci_width": mean_width,
        },
        "recommendations": {
            "min_effect_lognll": float(recommended),
            "expected_enable_rate": float(expected_rate),
        },
    }


__all__ = ["summarize_ve_sweep_reports"]
