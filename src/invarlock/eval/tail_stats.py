from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "compute_tail_summary",
    "evaluate_metric_tail",
]


def _as_finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _linear_quantile(sorted_values: Sequence[float], q: float) -> float:
    """Deterministic linear-interpolated quantile on sorted values (q in [0, 1])."""
    n = len(sorted_values)
    if n == 0:
        return float("nan")
    if n == 1:
        return float(sorted_values[0])
    if q <= 0.0:
        return float(sorted_values[0])
    if q >= 1.0:
        return float(sorted_values[-1])
    pos = float(q) * float(n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - float(lo)
    a = float(sorted_values[lo])
    b = float(sorted_values[hi])
    return a + frac * (b - a)


def compute_tail_summary(
    deltas: Sequence[float] | Sequence[Any],
    *,
    quantiles: Sequence[float] = (0.5, 0.9, 0.95, 0.99),
    epsilon: float = 1e-4,
    weights: Sequence[float] | Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Compute deterministic tail summaries for Δlog-loss samples.

    - Quantiles are computed unweighted using linear interpolation on sorted values.
    - tail_mass is Pr[delta > epsilon] (unweighted).
    - tail_mass_weighted is included when weights are provided and finite.
    """
    eps = _as_finite_float(epsilon)
    if eps is None or eps < 0.0:
        eps = 0.0

    values: list[float] = []
    paired_weights: list[float] | None = [] if weights is not None else None

    if weights is None:
        for d in deltas:
            dv = _as_finite_float(d)
            if dv is None:
                continue
            values.append(float(dv))
    else:
        for d, w in zip(deltas, weights, strict=False):
            dv = _as_finite_float(d)
            if dv is None:
                continue
            wv = _as_finite_float(w)
            if wv is None or wv < 0.0:
                wv = 0.0
            values.append(float(dv))
            if paired_weights is not None:
                paired_weights.append(float(wv))

    n = int(len(values))
    values_sorted = sorted(values)

    summary: dict[str, Any] = {
        "n": n,
        "epsilon": float(eps),
    }
    if n == 0:
        summary.update({"max": float("nan"), "tail_mass": 0.0})
        for q in quantiles:
            try:
                qf = float(q)
            except Exception:
                continue
            label = f"q{int(round(100.0 * max(0.0, min(1.0, qf))))}"
            summary[label] = float("nan")
        return summary

    summary["max"] = float(values_sorted[-1])
    tail_ct = sum(1 for v in values if v > eps)
    summary["tail_mass"] = float(tail_ct / n)

    if paired_weights is not None:
        total_w = 0.0
        tail_w = 0.0
        for v, w in zip(values, paired_weights, strict=False):
            total_w += float(w)
            if v > eps:
                tail_w += float(w)
        if total_w > 0.0:
            summary["tail_mass_weighted"] = float(tail_w / total_w)
            summary["tail_mass_weighted_by"] = "weights"

    for q in quantiles:
        try:
            qf = float(q)
        except Exception:
            continue
        qf = max(0.0, min(1.0, qf))
        label = f"q{int(round(100.0 * qf))}"
        summary[label] = float(_linear_quantile(values_sorted, qf))

    return summary


def evaluate_metric_tail(
    *,
    deltas: Sequence[float] | Sequence[Any],
    policy: Mapping[str, Any] | None = None,
    weights: Sequence[float] | Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a tail policy against Δlog-loss samples.

    Policy keys:
      - mode: "off" | "warn" | "fail" (default: "warn")
      - min_windows: int (default: 1)
      - quantile: float in [0, 1] (default: 0.95)
      - quantile_max: float threshold in Δlog-loss (optional)
      - epsilon: float deadband for tail_mass (default: 1e-4)
      - mass_max: float in [0, 1] (optional)
    """
    pol = dict(policy or {})
    mode = str(pol.get("mode", "warn") or "warn").strip().lower()
    if mode not in {"off", "warn", "fail"}:
        mode = "warn"

    min_windows = pol.get("min_windows", 1)
    try:
        min_windows_i = int(min_windows)
    except Exception:
        min_windows_i = 1
    min_windows_i = max(1, min_windows_i)

    q = _as_finite_float(pol.get("quantile", 0.95))
    if q is None:
        q = 0.95
    q = max(0.0, min(1.0, float(q)))

    eps = _as_finite_float(pol.get("epsilon", 1e-4))
    if eps is None or eps < 0.0:
        eps = 0.0

    qmax = _as_finite_float(pol.get("quantile_max"))
    mmax = _as_finite_float(pol.get("mass_max"))
    if mmax is not None:
        mmax = max(0.0, min(1.0, float(mmax)))

    quantiles = sorted({0.5, 0.9, 0.95, 0.99, float(q)})
    stats = compute_tail_summary(
        deltas, quantiles=tuple(quantiles), epsilon=float(eps), weights=weights
    )
    n = int(stats.get("n", 0) or 0)

    thresholds_present = (qmax is not None) or (mmax is not None)
    evaluated = bool(mode != "off" and thresholds_present and n >= min_windows_i)

    violations: list[dict[str, Any]] = []
    passed = True
    if evaluated:
        passed = True
        q_label = f"q{int(round(100.0 * q))}"
        q_obs = stats.get(q_label)
        if not (isinstance(q_obs, int | float) and math.isfinite(float(q_obs))):
            q_obs = float("nan")
        if qmax is not None and math.isfinite(q_obs) and q_obs > float(qmax):
            passed = False
            violations.append(
                {
                    "type": "quantile_max_exceeded",
                    "quantile": float(q),
                    "observed": float(q_obs),
                    "threshold": float(qmax),
                }
            )

        tail_mass = stats.get("tail_mass")
        if (
            mmax is not None
            and isinstance(tail_mass, int | float)
            and math.isfinite(float(tail_mass))
            and float(tail_mass) > float(mmax)
        ):
            passed = False
            violations.append(
                {
                    "type": "tail_mass_exceeded",
                    "epsilon": float(eps),
                    "observed": float(tail_mass),
                    "threshold": float(mmax),
                }
            )

    warned = bool(evaluated and (not passed) and mode == "warn")

    return {
        "mode": mode,
        "evaluated": evaluated,
        "passed": bool(passed),
        "warned": warned,
        "violations": violations,
        "policy": {
            "mode": mode,
            "min_windows": int(min_windows_i),
            "quantile": float(q),
            "quantile_max": float(qmax) if qmax is not None else None,
            "epsilon": float(eps),
            "mass_max": float(mmax) if mmax is not None else None,
        },
        "stats": stats,
    }
