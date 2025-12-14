from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any


def _finite01(value: Any) -> bool:
    try:
        f = float(value)
        return math.isfinite(f) and 0.0 <= f <= 1.0
    except Exception:
        return False


def _bh_reject_families(
    family_pvals: dict[str, float],
    *,
    alpha: float,
    m: int,
) -> set[str]:
    if not family_pvals:
        return set()
    try:
        alpha_f = float(alpha)
    except Exception:
        return set()
    if not (0.0 < alpha_f <= 1.0):
        return set()

    names = list(family_pvals.keys())
    pvals = [family_pvals[name] for name in names]
    n = len(pvals)
    m_eff = max(int(m) if isinstance(m, int) else 0, n, 1)

    order = sorted(
        range(n),
        key=lambda idx: (float("inf") if not _finite01(pvals[idx]) else pvals[idx]),
    )
    max_k = 0
    for rank, idx in enumerate(order, start=1):
        p = pvals[idx]
        if not _finite01(p):
            continue
        if p <= (alpha_f * rank) / m_eff:
            max_k = rank
    if max_k <= 0:
        return set()
    cutoff = (alpha_f * max_k) / m_eff
    selected: set[str] = set()
    for idx in order:
        p = pvals[idx]
        if _finite01(p) and p <= cutoff:
            selected.add(names[idx])
    return selected


def _bonferroni_reject_families(
    family_pvals: dict[str, float],
    *,
    alpha: float,
    m: int,
) -> set[str]:
    if not family_pvals:
        return set()
    try:
        alpha_f = float(alpha)
    except Exception:
        return set()
    if not (0.0 < alpha_f <= 1.0):
        return set()
    m_eff = max(int(m) if isinstance(m, int) else 0, len(family_pvals), 1)
    cutoff = alpha_f / m_eff
    return {fam for fam, p in family_pvals.items() if _finite01(p) and p <= cutoff}


def _extract_guard(report: dict[str, Any], name: str) -> dict[str, Any] | None:
    guards = report.get("guards")
    if isinstance(guards, list):
        for item in guards:
            if isinstance(item, dict) and item.get("name") == name:
                return item
    return None


def _extract_family_max_z(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    summary = metrics.get("family_z_summary")
    if isinstance(summary, dict):
        for fam, vals in summary.items():
            if not isinstance(vals, dict):
                continue
            z = vals.get("max")
            try:
                if z is not None and math.isfinite(float(z)):
                    out[str(fam)] = float(z)
            except Exception:
                continue
    q = metrics.get("family_z_quantiles")
    if isinstance(q, dict):
        for fam, vals in q.items():
            if not isinstance(vals, dict):
                continue
            z = vals.get("max")
            try:
                if z is not None and math.isfinite(float(z)):
                    out[str(fam)] = max(out.get(str(fam), float("-inf")), float(z))
            except Exception:
                continue
    return out


def _extract_multiple_testing(metrics: dict[str, Any]) -> dict[str, Any]:
    mt = metrics.get("multiple_testing")
    if not isinstance(mt, dict):
        return {}
    out: dict[str, Any] = {}
    method = mt.get("method")
    if isinstance(method, str) and method.strip():
        out["method"] = method.strip().lower()
    try:
        alpha = mt.get("alpha")
        if alpha is not None:
            out["alpha"] = float(alpha)
    except Exception:
        pass
    try:
        m_val = mt.get("m")
        if m_val is not None:
            out["m"] = int(m_val)
    except Exception:
        pass
    return out


def _selected_families_for_alpha(
    pvals: dict[str, float],
    *,
    method: str,
    alpha: float,
    m: int,
) -> set[str]:
    meth = (method or "").strip().lower()
    if meth == "bonferroni":
        return _bonferroni_reject_families(pvals, alpha=alpha, m=m)
    # Default: BH
    return _bh_reject_families(pvals, alpha=alpha, m=m)


def summarize_null_sweep_reports(
    reports: list[dict[str, Any]],
    *,
    tier: str,
    safety_margin: float = 0.05,
    target_any_warning_rate: float = 0.01,
) -> dict[str, Any]:
    """Summarize spectral null-sweep results and recommend κ/alpha.

    Inputs are run report dicts produced by `invarlock run` (or equivalent).
    """

    tier_norm = (tier or "").strip().lower() or "balanced"
    margin = float(safety_margin or 0.0)
    if not (0.0 <= margin <= 1.0):
        margin = 0.05
    target = float(target_any_warning_rate or 0.0)
    if not (0.0 <= target <= 1.0):
        target = 0.01

    family_max_z: dict[str, float] = defaultdict(lambda: float("-inf"))
    has_warning_default: list[bool] = []
    run_pvals: list[dict[str, float]] = []

    mt_method = "bh"
    mt_alpha = 0.05
    mt_m = 4

    selected_by_family: Counter[str] = Counter()
    candidate_by_family: Counter[str] = Counter()

    for report in reports:
        if not isinstance(report, dict):
            continue
        g = _extract_guard(report, "spectral") or {}
        metrics = g.get("metrics", {}) if isinstance(g.get("metrics"), dict) else {}
        mt = _extract_multiple_testing(metrics)
        if mt:
            mt_method = str(mt.get("method", mt_method))
            if mt.get("alpha") is not None:
                mt_alpha = float(mt.get("alpha"))
            if mt.get("m") is not None:
                mt_m = int(mt.get("m"))

        fam_z = _extract_family_max_z(metrics)
        for fam, z in fam_z.items():
            family_max_z[fam] = max(family_max_z[fam], float(z))

        selection = (
            metrics.get("multiple_testing_selection")
            if isinstance(metrics.get("multiple_testing_selection"), dict)
            else {}
        )
        pvals = selection.get("family_pvalues")
        if not isinstance(pvals, dict):
            pvals = {}
        parsed_pvals: dict[str, float] = {}
        for fam, p in pvals.items():
            try:
                pf = float(p)
            except Exception:
                continue
            if _finite01(pf):
                parsed_pvals[str(fam)] = pf
        run_pvals.append(parsed_pvals)

        families_selected = selection.get("families_selected")
        if isinstance(families_selected, list):
            for fam in families_selected:
                selected_by_family[str(fam)] += 1

        fam_counts = selection.get("family_violation_counts")
        if isinstance(fam_counts, dict):
            for fam, count in fam_counts.items():
                try:
                    candidate_by_family[str(fam)] += int(count)
                except Exception:
                    continue

        caps_applied = metrics.get("caps_applied")
        try:
            caps_applied_int = int(caps_applied) if caps_applied is not None else 0
        except Exception:
            caps_applied_int = 0
        violations = g.get("violations", [])
        has_warning_default.append(bool(caps_applied_int) or bool(violations))

    n = max(len(has_warning_default), 1)
    observed_any_rate = sum(1 for v in has_warning_default if v) / float(n)

    # κ recommendation: max observed z per family (+ margin), rounded for stable tiers.yaml diffs.
    rec_caps: dict[str, float] = {}
    for fam, z in sorted(family_max_z.items()):
        if not math.isfinite(z):
            continue
        kappa = z * (1.0 + margin)
        rec_caps[fam] = float(round(kappa, 3))

    # α calibration: choose the largest alpha that meets target_any_warning_rate.
    # This uses per-run family p-values (from spectral.multiple_testing_selection).
    def _rate_for_alpha(alpha: float) -> float:
        any_sel = 0
        for pvals in run_pvals:
            selected = _selected_families_for_alpha(
                pvals, method=mt_method, alpha=alpha, m=mt_m
            )
            any_sel += 1 if selected else 0
        return any_sel / float(max(len(run_pvals), 1))

    recommended_alpha = float(mt_alpha)
    if run_pvals and observed_any_rate > target:
        # Halving search is stable/deterministic and avoids dependency-heavy optimizers.
        alpha_grid: list[float] = []
        a = float(mt_alpha)
        for _ in range(20):
            if a <= 1e-6:
                break
            alpha_grid.append(a)
            a *= 0.5
        alpha_grid.append(1e-6)
        best = None
        for candidate in alpha_grid:
            rate = _rate_for_alpha(candidate)
            if rate <= target:
                best = candidate
                break
        if best is not None:
            recommended_alpha = float(best)

    return {
        "tier": tier_norm,
        "n_runs": int(len(has_warning_default)),
        "observed": {
            "any_warning_rate": float(observed_any_rate),
            "selected_by_family_runs": dict(selected_by_family),
            "candidate_violations_by_family_total": dict(candidate_by_family),
            "family_max_z": {
                k: float(v) for k, v in sorted(family_max_z.items()) if math.isfinite(v)
            },
        },
        "recommendations": {
            "family_caps": rec_caps,
            "multiple_testing": {
                "method": str(mt_method),
                "alpha": float(recommended_alpha),
                "m": int(mt_m),
            },
        },
    }


__all__ = ["summarize_null_sweep_reports"]
