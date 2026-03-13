from __future__ import annotations

import math
from typing import Any


def finite01(value: Any) -> bool:
    try:
        numeric = float(value)
        return math.isfinite(numeric) and 0.0 <= numeric <= 1.0
    except Exception:
        return False


def z_to_two_sided_pvalue(z: Any) -> float:
    try:
        zf = float(z)
        if not math.isfinite(zf):
            return 1.0
        return float(math.erfc(abs(zf) / math.sqrt(2.0)))
    except Exception:
        return 1.0


def bh_reject_families(
    family_pvals: dict[str, float], *, alpha: float, m: int
) -> set[str]:
    """BH family selection with denominator `m`."""
    if not family_pvals:
        return set()
    try:
        alpha_f = float(alpha)
    except Exception:
        alpha_f = 0.05
    if not (0.0 < alpha_f <= 1.0):
        return set()

    names = list(family_pvals.keys())
    pvals = [family_pvals[name] for name in names]
    n = len(pvals)
    m_eff = max(int(m) if isinstance(m, int) else 0, n, 1)
    order = sorted(
        range(n),
        key=lambda index: float("inf") if not finite01(pvals[index]) else pvals[index],
    )
    max_k = 0
    for rank, index in enumerate(order, start=1):
        pvalue = pvals[index]
        if not finite01(pvalue):
            continue
        if pvalue <= (alpha_f * rank) / m_eff:
            max_k = rank
    if max_k <= 0:
        return set()
    cutoff = (alpha_f * max_k) / m_eff
    return {
        names[index]
        for index in order
        if finite01(pvals[index]) and pvals[index] <= cutoff
    }


def bonferroni_reject_families(
    family_pvals: dict[str, float], *, alpha: float, m: int
) -> set[str]:
    if not family_pvals:
        return set()
    try:
        alpha_f = float(alpha)
    except Exception:
        alpha_f = 0.05
    if not (0.0 < alpha_f <= 1.0):
        return set()
    m_eff = max(int(m) if isinstance(m, int) else 0, len(family_pvals), 1)
    cutoff = alpha_f / m_eff
    return {
        family
        for family, pvalue in family_pvals.items()
        if finite01(pvalue) and pvalue <= cutoff
    }


def select_budgeted_violations(
    guard: Any, budgeted_violations: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply BH/Bonferroni selection at the family level."""
    mt = guard.multiple_testing if isinstance(guard.multiple_testing, dict) else {}
    method = str(mt.get("method", "bh")).lower()
    try:
        alpha = float(mt.get("alpha", 0.05) or 0.05)
    except Exception:
        alpha = 0.05
    m_raw = mt.get("m")
    m = None
    try:
        if m_raw is not None:
            m = int(m_raw)
    except Exception:
        m = None

    for violation in budgeted_violations:
        if violation.get("family"):
            continue
        module = violation.get("module")
        if isinstance(module, str):
            family = guard.module_family_map.get(module)
            if isinstance(family, str) and family:
                violation["family"] = family
                continue
        violation["family"] = "other"

    family_pvals: dict[str, float] = {}
    family_max_abs_z: dict[str, float] = {}
    family_counts: dict[str, int] = {}
    for violation in budgeted_violations:
        family = str(violation.get("family"))
        try:
            zf = float(violation.get("z_score"))
        except Exception:
            continue
        if not math.isfinite(zf):
            continue
        pvalue = z_to_two_sided_pvalue(zf)
        family_counts[family] = family_counts.get(family, 0) + 1
        current = family_pvals.get(family)
        if current is None or pvalue < current:
            family_pvals[family] = pvalue
            family_max_abs_z[family] = abs(zf)

    families_tested = sorted(family_pvals.keys())
    m_eff = m if isinstance(m, int) and m > 0 else len(families_tested)
    m_eff = max(m_eff, len(families_tested), 1)
    if isinstance(guard.multiple_testing, dict):
        guard.multiple_testing.setdefault("m", m_eff)

    if method in {"bh", "benjamini-hochberg", "benjamini_hochberg"}:
        selected_families = bh_reject_families(family_pvals, alpha=alpha, m=m_eff)
        applied_method = "bh"
    elif method in {"bonferroni", "bonf"}:
        selected_families = bonferroni_reject_families(
            family_pvals, alpha=alpha, m=m_eff
        )
        applied_method = "bonferroni"
    else:
        selected_families = bonferroni_reject_families(
            family_pvals, alpha=alpha, m=m_eff
        )
        applied_method = "bonferroni"

    selected: list[dict[str, Any]] = []
    default_selected_without_pvalue = 0
    for violation in budgeted_violations:
        family = (
            str(violation.get("family")) if violation.get("family") is not None else ""
        )
        z_val = violation.get("z_score")
        p_val: float | None = None
        try:
            zf = float(z_val)
        except Exception:
            zf = None
        if zf is not None and math.isfinite(zf):
            p_val = z_to_two_sided_pvalue(zf)
            is_selected = family in selected_families
        else:
            is_selected = True
            default_selected_without_pvalue += 1
        violation["p_value"] = p_val
        violation["selected"] = is_selected
        if is_selected:
            selected.append(violation)

    selection_metrics = {
        "method": applied_method,
        "alpha": alpha,
        "m": int(m_eff),
        "families_tested": families_tested,
        "families_selected": sorted(selected_families),
        "family_pvalues": {key: float(family_pvals[key]) for key in families_tested},
        "family_max_abs_z": {
            key: float(family_max_abs_z[key]) for key in families_tested
        },
        "family_violation_counts": dict(family_counts),
        "default_selected_without_pvalue": int(default_selected_without_pvalue),
    }
    return selected, selection_metrics


__all__ = [
    "bh_reject_families",
    "bonferroni_reject_families",
    "finite01",
    "select_budgeted_violations",
    "z_to_two_sided_pvalue",
]
