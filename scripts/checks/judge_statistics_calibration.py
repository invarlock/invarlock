#!/usr/bin/env python3
"""Deterministically calibrate the bounded judge interval implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from statistics import NormalDist
from typing import Any

from invarlock.judge_measurements.statistics import METHOD_ID, hoeffding_interval

EXPERIMENTS = 1_900
ALPHA = Decimal("0.05")
COMPARISONS = 4
CALIBRATION_FAMILY_ALPHA = 0.05


@dataclass(frozen=True)
class Scenario:
    name: str
    unit_count: int
    mean: Decimal
    draw: Callable[[random.Random], Decimal]


def _seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big")


def _wilson(misses: int, total: int, *, comparisons: int) -> tuple[float, float]:
    rate = misses / total
    z = NormalDist().inv_cdf(1 - CALIBRATION_FAMILY_ALPHA / (2 * comparisons))
    z2 = z * z
    denominator = 1 + z2 / total
    center = (rate + z2 / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1 - rate) / total + z2 / (4 * total * total))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _scenarios() -> tuple[Scenario, ...]:
    def bernoulli(probability: str) -> Callable[[random.Random], Decimal]:
        threshold = float(probability)
        return lambda rng: Decimal(int(rng.random() < threshold))

    return (
        Scenario("bernoulli-005-n20", 20, Decimal("0.05"), bernoulli("0.05")),
        Scenario("bernoulli-050-n20", 20, Decimal("0.5"), bernoulli("0.5")),
        Scenario("bernoulli-095-n20", 20, Decimal("0.95"), bernoulli("0.95")),
        Scenario("bernoulli-005-n100", 100, Decimal("0.05"), bernoulli("0.05")),
        Scenario("bernoulli-050-n100", 100, Decimal("0.5"), bernoulli("0.5")),
        Scenario("bernoulli-095-n100", 100, Decimal("0.95"), bernoulli("0.95")),
        Scenario(
            "uniform-n40",
            40,
            Decimal("0.5"),
            lambda rng: Decimal(str(rng.random())),
        ),
        Scenario(
            "beta-2-8-n40",
            40,
            Decimal("0.2"),
            lambda rng: Decimal(str(rng.betavariate(2, 8))),
        ),
    )


def calibrate() -> dict[str, Any]:
    """Return stable Monte Carlo coverage and independent formula checks."""

    allocated_error = float(ALPHA) / COMPARISONS
    scenarios = _scenarios()
    results = []
    all_pass = True
    for scenario in scenarios:
        rng = random.Random(_seed(scenario.name))
        misses = 0
        formula_mismatches = 0
        for _ in range(EXPERIMENTS):
            values = [scenario.draw(rng) for _ in range(scenario.unit_count)]
            interval = hoeffding_interval(
                values,
                lower_bound=Decimal(0),
                upper_bound=Decimal(1),
                alpha=ALPHA,
                comparisons=COMPARISONS,
            )
            if not interval.lower <= scenario.mean <= interval.upper:
                misses += 1
            sample_mean = sum(map(float, values)) / scenario.unit_count
            radius = math.sqrt(
                math.log(2 * COMPARISONS / float(ALPHA)) / (2 * scenario.unit_count)
            )
            if not (
                float(interval.lower) <= max(0.0, sample_mean - radius) + 1e-14
                and float(interval.upper) >= min(1.0, sample_mean + radius) - 1e-14
            ):
                formula_mismatches += 1
        lower, upper = _wilson(misses, EXPERIMENTS, comparisons=len(scenarios))
        passed = formula_mismatches == 0 and upper <= allocated_error
        all_pass &= passed
        results.append(
            {
                "scenario": scenario.name,
                "unit_count": scenario.unit_count,
                "true_mean": str(scenario.mean),
                "misses": misses,
                "observed_miss_rate": f"{misses / EXPERIMENTS:.6f}",
                "miss_rate_wilson_simultaneous_95": [
                    f"{lower:.6f}",
                    f"{upper:.6f}",
                ],
                "formula_mismatches": formula_mismatches,
                "passed": passed,
            }
        )
    return {
        "format": "invarlock/judge-statistics-calibration-v1",
        "method": METHOD_ID,
        "experiments_per_scenario": EXPERIMENTS,
        "alpha": str(ALPHA),
        "comparison_family_size": COMPARISONS,
        "calibration_scenario_count": len(scenarios),
        "calibration_family_alpha": f"{CALIBRATION_FAMILY_ALPHA:.2f}",
        "allocated_error_per_comparison": f"{allocated_error:.6f}",
        "acceptance_rule": (
            "Each Wilson upper bound for the simulated miss rate, Bonferroni-"
            "adjusted to simultaneous 95% coverage across the calibration "
            "matrix, must not exceed alpha/family-size; every independently "
            "computed radius must be enclosed after outward quantization."
        ),
        "scenarios": results,
        "passed": all_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    payload = json.dumps(calibrate(), indent=2, sort_keys=True) + "\n"
    if args.check is not None:
        if args.check.read_text(encoding="utf-8") != payload:
            raise SystemExit(f"calibration result differs from {args.check}")
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
