from __future__ import annotations

import math
from collections.abc import Mapping


def bh_select(pvals: list[float], alpha: float) -> list[bool]:
    """
    Benjamini–Hochberg procedure. Returns boolean mask of rejections in input order.

    Preconditions: 0 < alpha <= 1; p in [0,1] or NaN (NaN => reject=False).
    """
    n = len(pvals)
    if n == 0:
        return []
    alpha = float(alpha)
    if not (0.0 < alpha <= 1.0):
        # Treat invalid alpha as no rejections to be conservative
        return [False] * n

    # Sort by p-value ascending while remembering original indices
    order = sorted(
        range(n), key=lambda i: (float("inf") if not _finite01(pvals[i]) else pvals[i])
    )
    rejs_sorted = [False] * n
    max_k = 0
    for rank, idx in enumerate(order, start=1):
        p = pvals[idx]
        if not _finite01(p):
            continue
        threshold = (alpha * rank) / n
        if p <= threshold:
            max_k = rank
    # Mark as rejected those with p <= (alpha * max_k / n)
    if max_k > 0:
        cutoff = (alpha * max_k) / n
        for idx in order:
            p = pvals[idx]
            if _finite01(p) and p <= cutoff:
                rejs_sorted[idx] = True
    # Return in original order
    return rejs_sorted


def spectral_decide(
    sigma_by_name: Mapping[str, float],
    default_denom_by_name: Mapping[str, float],
    family_of_name: Mapping[str, str],
    deadband: float,
    caps_by_family: Mapping[str, float],
    mtest: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """
    Pure decision kernel for spectral guard.

    - z_i = ((sigma_i / denom_i) - 1) / max(deadband, eps)
    - p_i = Phi(|z_i|) under standard normal tail (two-sided conservative mapping)
    - Multiple testing per method; then cap by family kappa.
    """
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

    # Map to two-sided p-values via complementary error function (normal)
    # p = 2 * (1 - Phi(|z|)) = erfc(|z| / sqrt(2))
    try:
        import math as _m

        def _p(z: float) -> float:
            return float(_m.erfc(abs(z) / math.sqrt(2.0)))
    except Exception:

        def _p(z: float) -> float:  # pragma: no cover
            return 1.0

    pvals = [_p(z_by_name[n]) for n in names]
    method_obj = (mtest or {}).get("method", "bh")
    method = str(method_obj).lower()
    alpha_obj = (mtest or {}).get("alpha", 0.05)
    try:
        alpha = float(alpha_obj)  # type: ignore[arg-type]
    except Exception:
        alpha = 0.05
    if method in {"bh", "benjamini-hochberg", "benjamini_hochberg"}:
        rejects = bh_select(pvals, alpha)
    elif method in {"bonferroni"}:
        cutoff = alpha / max(1, len(pvals))
        rejects = [bool(p <= cutoff) if _finite01(p) else False for p in pvals]
    else:
        # Unknown method: conservative
        rejects = [False] * len(pvals)

    # Apply per-family caps (kappa) after selection: greedily keep top-|z| per family up to kappa
    fam_map = {n: str(family_of_name.get(n, "other")) for n in names}
    selected: list[str] = []
    per_family_counts: dict[str, int] = {}
    # Sort by |z| descending; pick among rejected set
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


def _finite01(p: float) -> bool:
    try:
        return (
            (isinstance(p, int | float))
            and math.isfinite(p)
            and (0.0 <= float(p) <= 1.0)
        )
    except Exception:
        return False
