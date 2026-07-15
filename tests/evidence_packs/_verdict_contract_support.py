from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from invarlock.evidence_pack_contracts.probes import build_probe_binding


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
        "run_id": "fixture-run",
        "meta": {"model_id": "fixture/model", "adapter": "hf_causal", "profile": "ci"},
        "context": {"runtime": {"execution_mode": "container"}},
        "provenance": {"provider_digest": {"ids_sha256": "fixture-provider"}},
        "validation": validation,
        "primary_metric": {"degraded": degraded, "invalid": degraded},
        "guard_metric_impact": {"evaluated": True},
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


def write_rmt_probe(
    path: Path, *, stable: bool, report_path: Path | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report_path = report_path or path.parent / "evaluation.report.json"
    report_raw = report_path.read_bytes()
    report = json.loads(report_raw)
    payload = {
        "schema": "invarlock/rmt-probe-v1",
        "probe": "rmt_cross_model_v1",
        "stable": stable,
        "passed": stable,
        "action": "continue" if stable else "abort",
        "stable_guard": stable,
        "epsilon_by_family": {"ffn": 0.01},
        "epsilon_default": 0.01,
        "epsilon_violations": []
        if stable
        else [
            {
                "family": "ffn",
                "module": "ffn",
                "edge_base": 1.0,
                "edge_cur": 2.0,
                "delta": 1.0,
                "allowed": 1.01,
                "epsilon": 0.01,
            }
        ],
        "violations": [],
        "metrics": {
            "stable": stable,
            "epsilon_default": 0.01,
            "epsilon_by_family": {"ffn": 0.01},
            "edge_base_by_family": {"ffn": 1.0},
            "edge_cur_by_family": {"ffn": 1.0 if stable else 2.0},
        },
        "binding": build_probe_binding(
            report, "sha256:" + hashlib.sha256(report_raw).hexdigest()
        ),
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
    report_path: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report_path = report_path or path.parent / "evaluation.report.json"
    report_raw = report_path.read_bytes()
    report = json.loads(report_raw)
    if proposed_scales is None:
        proposed_scales = 1 if signal else 0
    if would_enable is None:
        would_enable = signal
    if ab_gain is None:
        ab_gain = 0.01 if signal else 0.0
    ppl_no_ve = 10.0
    ppl_with_ve = ppl_no_ve * (1.0 - float(ab_gain))
    payload = {
        "schema": "invarlock/ve-probe-v1",
        "probe": "ve_probe_v1",
        "signal": signal,
        "signal_reasons": [] if signal else ["insufficient_signal"],
        "would_enable": bool(would_enable),
        "gate_reason": "enabled" if would_enable else "rejected",
        "proposed_scales": int(proposed_scales),
        "ab_gain": float(ab_gain),
        "ppl_no_ve": ppl_no_ve,
        "ppl_with_ve": ppl_with_ve,
        "abs_improvement": ppl_no_ve - ppl_with_ve,
        "ratio_ci": [0.8, 0.9] if signal else None,
        "predictive_gate": {
            "would_enable": bool(would_enable),
            "reason": "enabled" if would_enable else "rejected",
        },
        "calibration": {
            "windows": 12,
            "min_coverage": 10,
            "tier": "balanced",
            "profile": "ci",
        },
        "binding": build_probe_binding(
            report, "sha256:" + hashlib.sha256(report_raw).hexdigest()
        ),
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
    strictness: str = "must_detect",
    primary_guard_required: bool = True,
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
                        "strictness": strictness,
                        "intent": "guard_value",
                        "primary_guard": primary_guard,
                        "generation": {"kind": "edit", "edit_spec": "noop"},
                        "requirements": {
                            "primary_guard_required": primary_guard_required,
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
