"""Regression coverage for SVD energy/error cap feasibility (synthetic)."""

from __future__ import annotations

import math


def _assert_cap_feasible(energy_keep: float, cap: float) -> None:
    floor = math.sqrt(max(0.0, 1.0 - float(energy_keep)))
    assert cap >= floor - 1e-9, (
        f"max_module_rel_err={cap} is below theoretical floor {floor:.6f} "
        f"for energy_keep={energy_keep}"
    )


def test_balanced_svd_caps_exceed_frobenius_floor():
    # Synthetic overrides (energy_keep, max_module_rel_err) representative of a
    # balanced SVD plan; should respect the Frobenius floor.
    overrides = [
        {"energy_keep": 0.95, "max_module_rel_err": 0.25},
        {"energy_keep": 0.90, "max_module_rel_err": 0.35},
        {"energy_keep": 0.85, "max_module_rel_err": 0.40},
    ]
    for override in overrides:
        _assert_cap_feasible(override["energy_keep"], override["max_module_rel_err"])


def test_conservative_svd_caps_exceed_frobenius_floor():
    # Synthetic overrides representative of a conservative plan
    overrides = [
        {"energy_keep": 0.98, "max_module_rel_err": 0.15},
        {"energy_keep": 0.96, "max_module_rel_err": 0.20},
    ]
    for override in overrides:
        _assert_cap_feasible(override["energy_keep"], override["max_module_rel_err"])
