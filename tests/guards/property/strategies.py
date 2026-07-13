from __future__ import annotations

import math
from collections.abc import Mapping

from hypothesis import strategies as st

_FLOAT_COERCION_ERRORS = (TypeError, ValueError, OverflowError)


def _coerce_float(value: object, default: float) -> float:
    if not isinstance(value, bool | int | float | str):
        return float(default)
    try:
        return float(value)
    except _FLOAT_COERCION_ERRORS:
        return float(default)


def bh_select(pvals: list[float], alpha: float, *, m: int | None = None) -> list[bool]:
    n = len(pvals)
    if n == 0:
        return []
    alpha = float(alpha)
    if not (0.0 < alpha <= 1.0):
        return [False] * n

    m_eff = max(m if isinstance(m, int) and m > 0 else n, n, 1)
    order = sorted(
        range(n), key=lambda i: float("inf") if not _finite01(pvals[i]) else pvals[i]
    )
    rejs_sorted = [False] * n
    max_k = 0
    for rank, idx in enumerate(order, start=1):
        p = pvals[idx]
        if not _finite01(p):
            continue
        threshold = (alpha * rank) / m_eff
        if p <= threshold:
            max_k = rank

    if max_k > 0:
        cutoff = (alpha * max_k) / m_eff
        for idx in order:
            p = pvals[idx]
            if _finite01(p) and p <= cutoff:
                rejs_sorted[idx] = True
    return rejs_sorted


def spectral_decide(
    sigma_by_name: Mapping[str, float],
    default_denom_by_name: Mapping[str, float],
    family_of_name: Mapping[str, str],
    deadband: float,
    caps_by_family: Mapping[str, float],
    mtest: Mapping[str, object] | None = None,
) -> dict[str, object]:
    eps = 1e-12
    dead = max(float(deadband or 0.0), 0.0)

    names = list(sigma_by_name.keys())
    z_by_name: dict[str, float] = {}
    for name in names:
        s = float(sigma_by_name.get(name, 0.0) or 0.0)
        d = float(default_denom_by_name.get(name, 1.0) or 1.0)
        d = d if d > 0.0 else 1.0
        rel = (s / d) - 1.0
        z = 0.0
        if abs(rel) > dead:
            z = rel / max(dead, eps)
        z_by_name[name] = z

    def _p(z: float) -> float:
        return float(math.erfc(abs(z) / math.sqrt(2.0)))

    pvals = [_p(z_by_name[n]) for n in names]
    method_obj = (mtest or {}).get("method", "bh")
    method = str(method_obj).lower()
    alpha_obj = (mtest or {}).get("alpha", 0.05)
    alpha = _coerce_float(alpha_obj, 0.05)
    if method in {"bh", "benjamini-hochberg", "benjamini_hochberg"}:
        rejects = bh_select(pvals, alpha)
    elif method in {"bonferroni"}:
        cutoff = alpha / max(1, len(pvals))
        rejects = [bool(p <= cutoff) if _finite01(p) else False for p in pvals]
    else:
        rejects = [False] * len(pvals)

    fam_map = {n: str(family_of_name.get(n, "other")) for n in names}
    selected: list[str] = []
    per_family_counts: dict[str, int] = {}
    for name in sorted(names, key=lambda n: abs(z_by_name[n]), reverse=True):
        if not rejects[names.index(name)]:
            continue
        fam = fam_map[name]
        kappa = float(caps_by_family.get(fam, float("inf")) or float("inf"))
        curr = per_family_counts.get(fam, 0)
        if curr < int(math.ceil(kappa)):
            per_family_counts[fam] = curr + 1
            selected.append(name)

    return {
        "pass": len(selected) == 0,
        "selected": selected,
        "z_by_name": z_by_name,
        "per_family_counts": per_family_counts,
    }


def spectral_family_decide(
    z_by_name: Mapping[str, float],
    family_of_name: Mapping[str, str],
    mtest: Mapping[str, object] | None = None,
) -> dict[str, object]:
    names = list(z_by_name.keys())

    def _p(z: float) -> float:
        return float(math.erfc(abs(z) / math.sqrt(2.0)))

    family_pvals: dict[str, float] = {}
    family_counts: dict[str, int] = {}
    for name in names:
        family = str(family_of_name.get(name, "other") or "other")
        z_val = _coerce_float(z_by_name.get(name, 0.0), 0.0)
        if not math.isfinite(z_val):
            continue
        p_val = _p(z_val)
        current = family_pvals.get(family)
        if current is None or p_val < current:
            family_pvals[family] = p_val
        family_counts[family] = family_counts.get(family, 0) + 1

    families = list(family_pvals.keys())
    pvals = [family_pvals[family] for family in families]
    method_obj = (mtest or {}).get("method", "bh")
    method = str(method_obj).lower()
    alpha_obj = (mtest or {}).get("alpha", 0.05)
    alpha = _coerce_float(alpha_obj, 0.05)
    m_obj = (mtest or {}).get("m")
    try:
        m = int(m_obj) if m_obj is not None else len(families)
    except (TypeError, ValueError, OverflowError):
        m = len(families)
    m_eff = max(m, len(families), 1)
    if method in {"bh", "benjamini-hochberg", "benjamini_hochberg"}:
        rejects = bh_select(pvals, alpha, m=m_eff)
        applied_method = "bh"
    elif method in {"bonferroni", "bonf"}:
        cutoff = alpha / m_eff
        rejects = [bool(p <= cutoff) if _finite01(p) else False for p in pvals]
        applied_method = "bonferroni"
    else:
        cutoff = alpha / m_eff
        rejects = [bool(p <= cutoff) if _finite01(p) else False for p in pvals]
        applied_method = "bonferroni"

    selected_families = {
        family for family, reject in zip(families, rejects, strict=False) if reject
    }
    selected = [
        name
        for name in names
        if str(family_of_name.get(name, "other") or "other") in selected_families
    ]
    return {
        "pass": len(selected) == 0,
        "selected": selected,
        "families_selected": sorted(selected_families),
        "family_pvalues": family_pvals,
        "family_violation_counts": family_counts,
        "method": applied_method,
    }


def _finite01(p: object) -> bool:
    if not isinstance(p, int | float):
        return False
    value = _coerce_float(p, float("nan"))
    return math.isfinite(value) and (0.0 <= value <= 1.0)


def rmt_decide(
    baseline_by_family: Mapping[str, float],
    current_by_family: Mapping[str, float],
    epsilon_by_family: Mapping[str, float],
) -> dict[str, object]:
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
        delta_by_family[family] = (current / baseline) - 1.0

    ok = all(
        float(current_by_family.get(family, 0.0) or 0.0) <= allowed_by_family[family]
        for family in allowed_by_family
    )
    return {
        "pass": ok,
        "delta_by_family": delta_by_family,
        "allowed_by_family": allowed_by_family,
    }


def variance_decide(
    mean_delta: float,
    ci: tuple[float, float] | list[float],
    direction: str,
    min_effect: float,
    predictive_one_sided: bool,
) -> dict[str, object]:
    if not (isinstance(ci, tuple | list) and len(ci) == 2):
        return {"evaluated": False, "pass": True, "reason": "ci_unavailable"}
    lo, hi = float(ci[0]), float(ci[1])
    mu = float(mean_delta)
    me = float(min_effect or 0.0)

    dir_norm = (direction or "lower").strip().lower()
    if dir_norm == "higher":
        mu = -mu
        lo, hi = -hi, -lo

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


def families(min_families: int = 2, max_families: int = 5):
    fams = st.lists(
        st.sampled_from(["attn", "ffn", "embed", "other", "misc"]),
        min_size=min_families,
        max_size=max_families,
        unique=True,
    )
    return fams


@st.composite
def spectral_inputs(draw):
    family_names = draw(families())
    name_count = draw(st.integers(min_value=5, max_value=50))
    names = [f"m{i}" for i in range(name_count)]
    assignments = draw(
        st.lists(
            st.sampled_from(family_names),
            min_size=name_count,
            max_size=name_count,
        )
    )
    sigma_values = draw(
        st.lists(
            st.floats(
                min_value=1.0,
                max_value=1.5,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=name_count,
            max_size=name_count,
        )
    )
    deadband = draw(
        st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    cap = draw(
        st.floats(
            min_value=1.0,
            max_value=6.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    alpha = draw(
        st.floats(
            min_value=1e-6,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )

    family_of = dict(zip(names, assignments, strict=True))
    sigma = dict(zip(names, sigma_values, strict=True))
    denom = dict.fromkeys(names, 1.0)
    caps = dict.fromkeys(family_names, cap)
    mtest = {"method": "bh", "alpha": alpha}
    return sigma, denom, family_of, deadband, caps, mtest


@st.composite
def spectral_family_inputs(draw):
    family_names = draw(families(min_families=1, max_families=5))
    name_count = draw(st.integers(min_value=1, max_value=30))
    names = [f"module.{index}" for index in range(name_count)]
    assignments = draw(
        st.lists(
            st.sampled_from(family_names),
            min_size=name_count,
            max_size=name_count,
        )
    )
    z_values = draw(
        st.lists(
            st.floats(
                min_value=-8.0,
                max_value=8.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=name_count,
            max_size=name_count,
        )
    )
    method = draw(st.sampled_from(["bh", "bonferroni"]))
    alpha = draw(
        st.floats(
            min_value=1e-6,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return (
        dict(zip(names, z_values, strict=True)),
        dict(zip(names, assignments, strict=True)),
        {"method": method, "alpha": alpha},
    )


@st.composite
def rmt_inputs(draw):
    family_names = draw(families())
    family_count = len(family_names)
    baseline_values = draw(
        st.lists(
            st.integers(min_value=0, max_value=100),
            min_size=family_count,
            max_size=family_count,
        )
    )
    offsets = draw(
        st.lists(
            st.integers(min_value=-2, max_value=3),
            min_size=family_count,
            max_size=family_count,
        )
    )
    epsilon_values = draw(
        st.lists(
            st.floats(
                min_value=0.0,
                max_value=0.5,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=family_count,
            max_size=family_count,
        )
    )
    baseline = dict(zip(family_names, baseline_values, strict=True))
    current = {
        family: max(0, baseline[family] + offset)
        for family, offset in zip(family_names, offsets, strict=True)
    }
    epsilon = dict(zip(family_names, epsilon_values, strict=True))
    return baseline, current, epsilon


def variance_inputs():
    return st.builds(
        _build_var,
        st.floats(min_value=-0.01, max_value=0.01),
        st.floats(min_value=-0.02, max_value=0.0),
        st.floats(min_value=0.0, max_value=0.02),
        st.sampled_from(["lower", "higher"]),
        st.floats(min_value=0.0, max_value=0.02),
        st.booleans(),
    )


def _build_var(
    mu: float, lo: float, hi: float, direction: str, me: float, one_sided: bool
):
    lo2 = min(lo, hi)
    hi2 = max(lo, hi)
    return mu, (lo2, hi2), direction, float(me), bool(one_sided)
