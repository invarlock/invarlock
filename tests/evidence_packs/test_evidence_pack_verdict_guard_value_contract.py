from __future__ import annotations

import json
from pathlib import Path

from tests.evidence_packs._verdict_contract_support import (
    run_verdict_with_manifest,
    write_cert,
    write_guard_value_manifest,
    write_rmt_probe,
)


def test_verdict_contract_rejects_baseline_only_spectral_guard_value(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "run"
    manifest_path = tmp_path / "scenarios.json"
    write_guard_value_manifest(
        manifest_path,
        scenario_id="fp8_e5m2_stress",
        primary_guard="spectral",
        detectors_all_of=[
            {
                "kind": "validation_flag",
                "flag": "primary_metric_acceptable",
                "expected": True,
            },
            {
                "kind": "guard_signal_baseline_relative",
                "guard": "spectral",
                "min_new_modules": 1,
            },
        ],
    )

    model_dir = output_dir / "mistral-7b"
    baseline_path = (
        model_dir
        / "baseline_reports"
        / "ci_balanced_seq512_pv4_fn4"
        / "baseline_report.json"
    )
    subject_path = (
        model_dir / "reports" / "fp8_e5m2_stress" / "run_1" / "evaluation.report.json"
    )
    shared_violations = [
        ("model.layers.0.self_attn.q_proj", "attn", 7.7),
        ("model.layers.0.self_attn.k_proj", "attn", 3.2),
    ]
    validation = {
        "invariants_pass": True,
        "primary_metric_acceptable": True,
        "spectral_stable": True,
        "rmt_stable": True,
        "preview_final_drift_acceptable": True,
        "guard_overhead_acceptable": True,
    }
    write_cert(
        baseline_path,
        validation=validation,
        invariants_status="pass",
        spectral_caps_applied=2,
        spectral_violations=shared_violations,
    )
    write_cert(
        subject_path,
        validation=validation,
        invariants_status="pass",
        spectral_caps_applied=2,
        spectral_violations=shared_violations,
    )

    verdict = run_verdict_with_manifest(repo_root, output_dir, manifest_path)

    assert verdict["verdict"] == "FAIL"
    assert any(
        req.get("requirement") == "scenario_primary_guard_signal"
        and req.get("scenario") == "fp8_e5m2_stress"
        for req in verdict.get("failed_requirements", [])
    )
    assert any(
        req.get("requirement") == "scenario_expected_detectors"
        and req.get("scenario") == "fp8_e5m2_stress"
        for req in verdict.get("failed_requirements", [])
    )
    [record] = verdict["records"]
    assert record["detectors_hit"] is False
    assert record["primary_guard_hit"] is False
    spectral = record["guard_baseline_relative"]["spectral"]
    assert spectral["baseline_available"] is True
    assert spectral["new_caps_applied"] == 0
    assert spectral["delta_caps_applied"] == 0


def test_verdict_contract_accepts_new_spectral_cap_as_guard_value(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "run"
    manifest_path = tmp_path / "scenarios.json"
    write_guard_value_manifest(
        manifest_path,
        scenario_id="fp8_e5m2_stress",
        primary_guard="spectral",
        detectors_all_of=[
            {
                "kind": "validation_flag",
                "flag": "primary_metric_acceptable",
                "expected": True,
            },
            {
                "kind": "guard_signal_baseline_relative",
                "guard": "spectral",
                "min_new_modules": 1,
            },
        ],
    )

    model_dir = output_dir / "mistral-7b"
    baseline_path = (
        model_dir
        / "baseline_reports"
        / "ci_balanced_seq512_pv4_fn4"
        / "baseline_report.json"
    )
    subject_path = (
        model_dir / "reports" / "fp8_e5m2_stress" / "run_1" / "evaluation.report.json"
    )
    baseline_violations = [
        ("model.layers.0.self_attn.q_proj", "attn", 7.7),
        ("model.layers.0.self_attn.k_proj", "attn", 3.2),
    ]
    subject_violations = [
        *baseline_violations,
        ("model.layers.4.mlp.gate_proj", "ffn", 9.1),
    ]
    validation = {
        "invariants_pass": True,
        "primary_metric_acceptable": True,
        "spectral_stable": True,
        "rmt_stable": True,
        "preview_final_drift_acceptable": True,
        "guard_overhead_acceptable": True,
    }
    write_cert(
        baseline_path,
        validation=validation,
        invariants_status="pass",
        spectral_caps_applied=2,
        spectral_violations=baseline_violations,
    )
    write_cert(
        subject_path,
        validation=validation,
        invariants_status="pass",
        spectral_caps_applied=3,
        spectral_violations=subject_violations,
    )

    verdict = run_verdict_with_manifest(repo_root, output_dir, manifest_path)

    assert verdict["verdict"] == "PASS"
    [record] = verdict["records"]
    assert record["detectors_hit"] is True
    assert record["primary_guard_hit"] is True
    spectral = record["guard_baseline_relative"]["spectral"]
    assert spectral["new_caps_applied"] == 1
    assert spectral["delta_caps_applied"] == 1


def test_verdict_contract_rejects_guard_value_when_rmt_signal_is_baseline(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "run"
    manifest_path = tmp_path / "scenarios.json"
    write_guard_value_manifest(
        manifest_path,
        scenario_id="rmt_norm_noise",
        category="error_injection",
        primary_guard="rmt",
    )

    model_dir = output_dir / "mistral-7b"
    baseline_path = (
        model_dir
        / "baseline_reports"
        / "ci_balanced_seq512_pv4_fn4"
        / "baseline_report.json"
    )
    subject_path = (
        model_dir / "reports" / "errors" / "rmt_norm_noise" / "evaluation.report.json"
    )
    validation = {
        "invariants_pass": True,
        "primary_metric_acceptable": True,
        "spectral_stable": True,
        "rmt_stable": False,
        "preview_final_drift_acceptable": True,
        "guard_overhead_acceptable": True,
    }
    write_cert(baseline_path, validation=validation, invariants_status="pass")
    write_cert(subject_path, validation=validation, invariants_status="pass")

    verdict = run_verdict_with_manifest(repo_root, output_dir, manifest_path)

    assert verdict["verdict"] == "FAIL"
    assert any(
        req.get("requirement") == "scenario_primary_guard_signal"
        and req.get("scenario") == "rmt_norm_noise"
        for req in verdict.get("failed_requirements", [])
    )
    [record] = verdict["records"]
    assert record["detectors_hit"] is False
    assert record["primary_guard_hit"] is False
    rmt = record["guard_baseline_relative"]["rmt"]
    assert rmt["baseline_signal"] is True
    assert rmt["subject_signal"] is True
    assert rmt["relative_signal"] is False


def test_verdict_contract_supports_detectors_all_of(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "run"
    manifest_path = tmp_path / "scenarios.json"

    manifest_path.write_text(
        json.dumps(
            {
                "schema": "evidence_pack_scenarios_v1",
                "schema_version": 1,
                "scenarios": [
                    {
                        "id": "demo_error",
                        "category": "error_injection",
                        "failure_class": "demo",
                        "strictness": "must_detect",
                        "intent": "fault_detection",
                        "primary_guard": "rmt",
                        "generation": {"kind": "error", "error_type": "demo_error"},
                        "requirements": {
                            "primary_guard_required": True,
                            "detectors_any_of": [
                                {
                                    "kind": "rmt_probe",
                                    "field": "stable",
                                    "expected": False,
                                }
                            ],
                            "detectors_all_of": [
                                {
                                    "kind": "validation_flag",
                                    "flag": "primary_metric_acceptable",
                                    "expected": True,
                                }
                            ],
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

    model_dir = output_dir / "mistral-7b"
    cert_path = (
        model_dir / "reports" / "errors" / "demo_error" / "evaluation.report.json"
    )
    write_cert(
        cert_path,
        validation={
            "invariants_pass": True,
            # Make the AND clause fail (but keep the OR clause true via probe).
            "primary_metric_acceptable": False,
            "spectral_stable": True,
            "rmt_stable": True,
            "preview_final_drift_acceptable": True,
            "guard_overhead_acceptable": True,
        },
        invariants_status="pass",
    )
    write_rmt_probe(cert_path.parent / "rmt_probe.json", stable=False)

    verdict = run_verdict_with_manifest(repo_root, output_dir, manifest_path)

    assert verdict["verdict"] == "FAIL"
    assert any(
        req.get("requirement") == "error_injection_detected"
        and req.get("scenario") == "demo_error"
        for req in verdict.get("failed_requirements", [])
    )
