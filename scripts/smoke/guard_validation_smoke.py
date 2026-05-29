#!/usr/bin/env python3
"""Deterministic synthetic smoke for guard calibration evidence."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GuardScenario:
    name: str
    threshold: float
    null_mean: float
    null_sd: float
    defect_shift: float


SCENARIOS = (
    GuardScenario(
        name="spectral",
        threshold=2.75,
        null_mean=0.0,
        null_sd=1.0,
        defect_shift=3.25,
    ),
    GuardScenario(
        name="rmt",
        threshold=0.18,
        null_mean=0.04,
        null_sd=0.04,
        defect_shift=0.24,
    ),
    GuardScenario(
        name="variance",
        threshold=0.12,
        null_mean=0.03,
        null_sd=0.03,
        defect_shift=0.18,
    ),
)
CALIBRATION_WINDOWS = (16, 32, 64, 128)
MODEL_FAMILIES = ("gpt2", "llama", "qwen")


def _sample_mean(
    rng: random.Random,
    *,
    mean: float,
    sd: float,
    windows: int,
) -> float:
    values = [rng.gauss(mean, sd) for _ in range(windows)]
    return sum(values) / float(windows)


def _estimate_rates(
    scenario: GuardScenario,
    *,
    windows: int,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    false_positives = 0
    true_positives = 0
    window_scale = math.sqrt(max(float(windows), 1.0))
    null_sd = scenario.null_sd * math.sqrt(32.0) / window_scale
    defect_sd = scenario.null_sd * math.sqrt(32.0) / window_scale
    for _ in range(replicates):
        null_score = _sample_mean(
            rng,
            mean=scenario.null_mean,
            sd=null_sd,
            windows=windows,
        )
        defect_score = _sample_mean(
            rng,
            mean=scenario.null_mean + scenario.defect_shift,
            sd=defect_sd,
            windows=windows,
        )
        false_positives += int(null_score >= scenario.threshold)
        true_positives += int(defect_score >= scenario.threshold)
    denominator = float(max(replicates, 1))
    return {
        "type_i_error": false_positives / denominator,
        "power": true_positives / denominator,
    }


def build_guard_validation_smoke(
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for window_index, windows in enumerate(CALIBRATION_WINDOWS):
            rates = _estimate_rates(
                scenario,
                windows=windows,
                replicates=replicates,
                seed=seed + scenario_index * 1000 + window_index,
            )
            rows.append(
                {
                    "guard": scenario.name,
                    "calibration_windows": windows,
                    "threshold": scenario.threshold,
                    **rates,
                }
            )
    family_rows = [
        {
            "model_family": family,
            "guards": [scenario.name for scenario in SCENARIOS],
            "status": "synthetic_smoke_only",
        }
        for family in MODEL_FAMILIES
    ]
    return {
        "schema": "invarlock/guard-validation-smoke-v1",
        "seed": seed,
        "replicates": replicates,
        "scope": "deterministic synthetic smoke; not empirical model-family proof",
        "rate_rows": rows,
        "model_family_sensitivity": family_rows,
    }


def _write_markdown(path: Path, payload: dict[str, object]) -> None:
    rows = payload["rate_rows"]
    assert isinstance(rows, list)
    lines = [
        "# Guard Validation Smoke",
        "",
        "This generated artifact is a deterministic synthetic smoke, not a",
        "replacement for empirical guard calibration on real checkpoints.",
        "",
        "| Guard | Windows | Type-I Error | Power |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        assert isinstance(row, dict)
        lines.append(
            "| {guard} | {windows} | {type_i:.3f} | {power:.3f} |".format(
                guard=row["guard"],
                windows=row["calibration_windows"],
                type_i=row["type_i_error"],
                power=row["power"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic guard-validation smoke artifacts."
    )
    parser.add_argument("--output-dir", default="artifacts/guard-validation")
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_guard_validation_smoke(
        replicates=max(int(args.replicates), 1),
        seed=int(args.seed),
    )
    json_path = output_dir / "guard-validation-smoke.json"
    md_path = output_dir / "guard-validation-smoke.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_markdown(md_path, payload)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
