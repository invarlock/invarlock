from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def write_cert(
    path: Path,
    *,
    validation: dict[str, Any],
    degraded: bool = False,
    invariants_status: str = "pass",
    spectral_caps_applied: int | None = None,
    spectral_violations: list[tuple[str, str, float]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "validation": validation,
        "primary_metric": {"degraded": degraded, "invalid": degraded},
        "guard_overhead": {"evaluated": True},
        "invariants": {"status": invariants_status},
    }
    if spectral_caps_applied is not None or spectral_violations is not None:
        violations = spectral_violations or []
        payload["spectral"] = {
            "caps_applied": int(
                spectral_caps_applied
                if spectral_caps_applied is not None
                else len(violations)
            ),
            "violations": [
                {
                    "module": module,
                    "family": family,
                    "z_score": z_score,
                    "type": "family_z_cap",
                    "selected": True,
                }
                for module, family, z_score in violations
            ],
        }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_rmt_probe(path: Path, *, stable: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "probe": "rmt_cross_model_v1",
        "stable": stable,
        "epsilon_violations": [] if stable else [{"family": "ffn"}],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_ve_probe(
    path: Path,
    *,
    signal: bool,
    proposed_scales: int | None = None,
    would_enable: bool | None = None,
    ab_gain: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if proposed_scales is None:
        proposed_scales = 1 if signal else 0
    if would_enable is None:
        would_enable = signal
    if ab_gain is None:
        ab_gain = 0.01 if signal else 0.0
    payload = {
        "probe": "ve_probe_v1",
        "signal": signal,
        "would_enable": bool(would_enable),
        "proposed_scales": int(proposed_scales),
        "ab_gain": float(ab_gain),
        "ppl_no_ve": 10.0,
        "ppl_with_ve": 9.0 if signal else 10.0,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_guard_value_manifest(
    path: Path,
    *,
    scenario_id: str,
    category: str = "stress",
    primary_guard: str = "spectral",
    detectors_all_of: list[dict[str, Any]] | None = None,
) -> None:
    if detectors_all_of is None:
        detectors_all_of = [
            {
                "kind": "guard_signal_baseline_relative",
                "guard": primary_guard,
            }
        ]
    path.write_text(
        json.dumps(
            {
                "schema": "evidence_pack_scenarios_v1",
                "schema_version": 1,
                "scenarios": [
                    {
                        "id": scenario_id,
                        "category": category,
                        "failure_class": "test.guard_value",
                        "strictness": "must_detect",
                        "intent": "guard_value",
                        "primary_guard": primary_guard,
                        "generation": {"kind": "edit", "edit_spec": "noop"},
                        "requirements": {
                            "primary_guard_required": True,
                            "detectors_all_of": detectors_all_of,
                        },
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_verdict(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    script = repo_root / "scripts/evidence_packs/python/verdict_generator.py"
    subprocess.run(
        ["python3", str(script), "--output-dir", str(output_dir)],
        check=True,
        cwd=repo_root,
    )
    verdict_path = output_dir / "reports" / "final_verdict.json"
    return json.loads(verdict_path.read_text(encoding="utf-8"))


def run_verdict_with_manifest(
    repo_root: Path,
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    script = repo_root / "scripts/evidence_packs/python/verdict_generator.py"
    subprocess.run(
        [
            "python3",
            str(script),
            "--output-dir",
            str(output_dir),
            "--manifest",
            str(manifest_path),
        ],
        check=True,
        cwd=repo_root,
    )
    verdict_path = output_dir / "reports" / "final_verdict.json"
    return json.loads(verdict_path.read_text(encoding="utf-8"))
