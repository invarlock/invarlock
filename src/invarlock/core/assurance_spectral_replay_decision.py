"""Verifier-owned spectral finding and multiple-testing replay."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .assurance_spectral_replay_common import _reject_families


def replay_selected_findings(
    *,
    baseline: Mapping[str, float],
    current: Mapping[str, float],
    families: Mapping[str, str],
    family_stats: Mapping[str, Mapping[str, float | int]],
    family_caps: Mapping[str, float],
    deadband: float,
    max_norm: float | None,
    method: str,
    alpha: float,
    configured_m: int,
    degeneracy_enabled: bool,
    baseline_degeneracy: Mapping[str, Mapping[str, float]],
    current_degeneracy: Mapping[str, Mapping[str, float]],
    thresholds: Mapping[str, tuple[float, float]],
) -> tuple[
    dict[str, float],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    z_scores: dict[str, float] = {}
    budgeted: list[dict[str, Any]] = []
    fatal: list[dict[str, Any]] = []
    for module in sorted(current):
        stats = family_stats[families[module]]
        std = float(stats["std"])
        if std > 0.0:
            z_value = (current[module] - float(stats["mean"])) / std
        else:
            denominator = baseline[module] if baseline[module] > 0.0 else 1.0
            relative_change = (current[module] / denominator) - 1.0
            z_value = (
                0.0
                if abs(relative_change) <= deadband
                else relative_change / (deadband if deadband > 0.0 else 1.0)
            )
        z_scores[module] = z_value
        family = families[module]
        cap = family_caps[family]
        if abs(z_value) > cap:
            budgeted.append(
                {
                    "type": "family_z_cap",
                    "severity": "budgeted",
                    "module": module,
                    "family": family,
                    "z_score": z_value,
                    "kappa": cap,
                    "sigma": current[module],
                    "baseline_sigma": baseline[module],
                }
            )
        if max_norm is not None and current[module] > max_norm:
            fatal.append(
                {
                    "type": "max_spectral_norm",
                    "severity": "fatal",
                    "module": module,
                    "family": family,
                    "current_sigma": current[module],
                    "threshold": max_norm,
                }
            )
        if degeneracy_enabled:
            for metric, violation_type, fields in (
                (
                    "stable_rank",
                    "degeneracy_stable_rank_drop",
                    ("stable_rank_base", "stable_rank_cur"),
                ),
                (
                    "norm_collapse",
                    "degeneracy_norm_collapse",
                    ("norm_collapse_base", "norm_collapse_cur"),
                ),
            ):
                base_value = float(baseline_degeneracy[module][metric])
                current_value = float(current_degeneracy[module][metric])
                if base_value <= 0.0:
                    continue
                ratio = current_value / max(base_value, 1e-12)
                warn_ratio, fatal_ratio = thresholds[metric]
                if math.isfinite(ratio) and ratio < warn_ratio:
                    finding = {
                        "type": violation_type,
                        "severity": "fatal" if ratio < fatal_ratio else "budgeted",
                        "module": module,
                        "family": family,
                        fields[0]: base_value,
                        fields[1]: current_value,
                        "ratio": ratio,
                        "warn_ratio": warn_ratio,
                        "fatal_ratio": fatal_ratio,
                    }
                    (fatal if finding["severity"] == "fatal" else budgeted).append(
                        finding
                    )

    family_pvalues: dict[str, float] = {}
    family_max_abs_z: dict[str, float] = {}
    family_counts: dict[str, int] = {}
    for finding in budgeted:
        if finding["type"] != "family_z_cap":
            continue
        family = str(finding["family"])
        z_value = float(finding["z_score"])
        p_value = math.erfc(abs(z_value) / math.sqrt(2.0))
        family_counts[family] = family_counts.get(family, 0) + 1
        if family not in family_pvalues or p_value < family_pvalues[family]:
            family_pvalues[family] = p_value
            family_max_abs_z[family] = abs(z_value)
    families_tested = sorted(family_pvalues)
    m_effective = max(configured_m, len(families_tested), 1)
    selected_families = _reject_families(
        family_pvalues, method=method, alpha=alpha, m=m_effective
    )
    selected_budgeted: list[dict[str, Any]] = []
    default_selected = 0
    for finding in budgeted:
        item: dict[str, Any] = dict(finding)
        if item["type"] == "family_z_cap":
            item["p_value"] = math.erfc(abs(float(item["z_score"])) / math.sqrt(2.0))
            item["selected"] = item["family"] in selected_families
        else:
            item["p_value"] = None
            item["selected"] = True
            default_selected += 1
        if item["selected"]:
            selected_budgeted.append(item)
    selection = {
        "method": method,
        "alpha": alpha,
        "m": m_effective,
        "families_tested": families_tested,
        "families_selected": sorted(selected_families),
        "family_pvalues": {
            family: family_pvalues[family] for family in families_tested
        },
        "family_max_abs_z": {
            family: family_max_abs_z[family] for family in families_tested
        },
        "family_violation_counts": family_counts,
        "default_selected_without_pvalue": default_selected,
    }
    return z_scores, budgeted, fatal, selection, [*fatal, *selected_budgeted]


__all__ = ["replay_selected_findings"]
