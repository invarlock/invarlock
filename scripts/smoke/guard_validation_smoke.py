#!/usr/bin/env python3
"""Deterministic synthetic smoke for guard calibration evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from invarlock.guards.policies import (
    get_rmt_policy,
    get_spectral_policy,
    get_variance_policy,
)
from invarlock.guards.rmt_policy import compute_epsilon_violations
from invarlock.guards.spectral_detection import summarize_family_z_scores
from invarlock.guards.variance_policy import predictive_gate_outcome


@dataclass(frozen=True)
class GuardScenario:
    name: str
    threshold: float
    null_mean: float
    null_sd: float
    defect_shift: float
    production_entrypoint: str
    primitive_role: str


def _scenarios() -> tuple[GuardScenario, ...]:
    """Resolve smoke thresholds from the shipped policies under test."""

    spectral_threshold = float(
        get_spectral_policy("balanced")["family_caps"]["ffn"]["kappa"]
    )
    rmt_threshold = float(get_rmt_policy("balanced")["epsilon_by_family"]["ffn"])
    variance_threshold = float(get_variance_policy("aggressive")["min_effect_lognll"])
    return (
        GuardScenario(
            name="spectral",
            threshold=spectral_threshold,
            null_mean=0.0,
            null_sd=1.0,
            defect_shift=4.5,
            production_entrypoint=(
                "invarlock.guards.spectral_detection.summarize_family_z_scores"
            ),
            primitive_role="violation_summary",
        ),
        GuardScenario(
            name="rmt",
            threshold=rmt_threshold,
            null_mean=0.002,
            null_sd=0.002,
            defect_shift=0.02,
            production_entrypoint=(
                "invarlock.guards.rmt_policy.compute_epsilon_violations"
            ),
            primitive_role="violation_detection",
        ),
        GuardScenario(
            name="variance",
            threshold=variance_threshold,
            null_mean=0.005,
            null_sd=0.005,
            defect_shift=0.05,
            production_entrypoint=(
                "invarlock.guards.variance_policy.predictive_gate_outcome"
            ),
            primitive_role="gate_outcome",
        ),
    )


CALIBRATION_WINDOWS = (16, 32, 64, 128)
SCHEMA = "invarlock/guard-validation-smoke-v1"
SCOPE = (
    "deterministic synthetic production-primitive smoke; not empirical "
    "model-family proof or threshold calibration"
)
MAX_REPLICATES = 10_000
SOURCE_FILES = {
    "policy": "src/invarlock/guards/policies.py",
    "spectral": "src/invarlock/guards/spectral_detection.py",
    "rmt": "src/invarlock/guards/rmt_policy.py",
    "variance": "src/invarlock/guards/variance_policy.py",
}


def _sample_mean(
    rng: random.Random,
    *,
    mean: float,
    sd: float,
    windows: int,
) -> float:
    values = [rng.gauss(mean, sd) for _ in range(windows)]
    return sum(values) / float(windows)


def _guard_triggers(scenario: GuardScenario, score: float) -> bool:
    """Route each synthetic score through the named production primitive."""

    if scenario.name == "spectral":
        summary = summarize_family_z_scores(
            {"synthetic.ffn": score},
            {"synthetic.ffn": "ffn"},
            {"ffn": {"kappa": scenario.threshold}},
        )
        return int(summary["ffn"]["violations"]) > 0
    if scenario.name == "rmt":
        guard = SimpleNamespace(
            baseline_edge_risk_by_family={"ffn": 1.0},
            edge_risk_by_family={"ffn": 1.0 + score},
            epsilon_by_family={"ffn": scenario.threshold},
            epsilon_default=scenario.threshold,
        )
        return bool(compute_epsilon_violations(guard))
    if scenario.name == "variance":
        # The generated score is an improvement magnitude. The production
        # helper consumes final-minus-preview delta-log-loss, so improvement is
        # negative. Keep the synthetic interval strictly on the improving side.
        mean_delta = -score
        delta_ci = (mean_delta - 0.001, mean_delta)
        passed, _reason = predictive_gate_outcome(
            mean_delta,
            delta_ci,
            scenario.threshold,
            one_sided=True,
        )
        return passed
    raise ValueError(f"unsupported guard scenario: {scenario.name}")


def _estimate_outcomes(
    scenario: GuardScenario,
    *,
    windows: int,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    null_outcomes: list[bool] = []
    shifted_outcomes: list[bool] = []
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
        null_outcomes.append(_guard_triggers(scenario, null_score))
        shifted_outcomes.append(_guard_triggers(scenario, defect_score))
    false_positives = sum(null_outcomes)
    true_positives = sum(shifted_outcomes)
    denominator = float(replicates)
    return {
        "null_outcomes": null_outcomes,
        "shifted_outcomes": shifted_outcomes,
        "null_trigger_count": false_positives,
        "shifted_trigger_count": true_positives,
        "null_trigger_rate": false_positives / denominator,
        "shifted_trigger_rate": true_positives / denominator,
    }


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _canonical_digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _source_identity(repo_root: Path) -> dict[str, object]:
    producer_path = "scripts/smoke/guard_validation_smoke.py"
    return {
        "producer": {
            "path": producer_path,
            "sha256": _sha256_bytes((repo_root / producer_path).read_bytes()),
        },
        "policy": {
            "path": SOURCE_FILES["policy"],
            "sha256": _sha256_bytes((repo_root / SOURCE_FILES["policy"]).read_bytes()),
        },
        "primitives": {
            guard: {
                "path": SOURCE_FILES[guard],
                "sha256": _sha256_bytes((repo_root / SOURCE_FILES[guard]).read_bytes()),
            }
            for guard in ("spectral", "rmt", "variance")
        },
    }


def build_guard_validation_smoke(
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    if isinstance(replicates, bool) or not 1 <= replicates <= MAX_REPLICATES:
        raise ValueError(f"replicates must be in [1, {MAX_REPLICATES}]")
    if isinstance(seed, bool) or not -(2**63) <= seed < 2**63:
        raise ValueError("seed must be a signed 64-bit integer")
    rows: list[dict[str, object]] = []
    scenarios = _scenarios()
    for scenario_index, scenario in enumerate(scenarios):
        for window_index, windows in enumerate(CALIBRATION_WINDOWS):
            derived_seed = seed + scenario_index * 1000 + window_index
            outcomes = _estimate_outcomes(
                scenario,
                windows=windows,
                replicates=replicates,
                seed=derived_seed,
            )
            rows.append(
                {
                    "guard": scenario.name,
                    "calibration_windows": windows,
                    "threshold": scenario.threshold,
                    "production_entrypoint": scenario.production_entrypoint,
                    "primitive_role": scenario.primitive_role,
                    "derived_seed": derived_seed,
                    **outcomes,
                }
            )
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "seed": seed,
        "replicates": replicates,
        "scope": SCOPE,
        "source_identity": _source_identity(Path(__file__).resolve().parents[2]),
        "production_primitives": {
            scenario.name: {
                "entrypoint": scenario.production_entrypoint,
                "role": scenario.primitive_role,
            }
            for scenario in scenarios
        },
        "rate_rows": rows,
    }
    payload["evidence_sha256"] = _canonical_digest(payload)
    markdown = _render_markdown(payload)
    payload["markdown_sha256"] = _sha256_bytes(markdown.encode("utf-8"))
    return payload


def _render_markdown(payload: dict[str, object]) -> str:
    rows = payload["rate_rows"]
    assert isinstance(rows, list)
    lines = [
        "# Guard Validation Smoke",
        "",
        "This generated artifact is a deterministic synthetic smoke, not a",
        "replacement for empirical guard calibration on real checkpoints.",
        "",
        f"Evidence digest: `{payload['evidence_sha256']}`",
        "",
        "| Guard | Windows | Synthetic Null Trigger | Synthetic Shifted Trigger |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        assert isinstance(row, dict)
        lines.append(
            "| {guard} | {windows} | {null_rate:.3f} | {shifted_rate:.3f} |".format(
                guard=row["guard"],
                windows=row["calibration_windows"],
                null_rate=row["null_trigger_rate"],
                shifted_rate=row["shifted_trigger_rate"],
            )
        )
    return "\n".join(lines) + "\n"


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
    try:
        payload = build_guard_validation_smoke(
            replicates=int(args.replicates),
            seed=int(args.seed),
        )
    except ValueError as exc:
        parser.error(str(exc))
    json_path = output_dir / "guard-validation-smoke.json"
    md_path = output_dir / "guard-validation-smoke.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
