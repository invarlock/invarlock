from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def _write_cert(
    path: Path,
    *,
    validation: dict[str, Any],
    degraded: bool = False,
    invariants_status: str = "pass",
    spectral_caps_applied: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "validation": validation,
        "primary_metric": {"degraded": degraded, "invalid": degraded},
        "guard_overhead": {"evaluated": True},
        "invariants": {"status": invariants_status},
    }
    if spectral_caps_applied is not None:
        payload["spectral"] = {"caps_applied": int(spectral_caps_applied)}
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_rmt_probe(path: Path, *, stable: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "probe": "rmt_cross_model_v1",
        "stable": stable,
        "epsilon_violations": [] if stable else [{"family": "ffn"}],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_ve_probe(path: Path, *, signal: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "probe": "ve_probe_v1",
        "signal": signal,
        "would_enable": signal,
        "proposed_scales": 1 if signal else 0,
        "ab_gain": 0.01 if signal else 0.0,
        "ppl_no_ve": 10.0,
        "ppl_with_ve": 9.0 if signal else 10.0,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _run_verdict(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    script = repo_root / "scripts/proof_packs/python/verdict_generator.py"
    subprocess.run(
        ["python3", str(script), "--output-dir", str(output_dir)],
        check=True,
        cwd=repo_root,
    )
    verdict_path = output_dir / "reports" / "final_verdict.json"
    return json.loads(verdict_path.read_text(encoding="utf-8"))


def test_verdict_contract_clean_pass_catastrophic_fail_errors_detected(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "run"
    model_dir = output_dir / "mistral-7b"

    # Clean edits (4) => must PASS.
    for edit in (
        "quant_4bit_clean",
        "fp8_e5m2_clean",
        "prune_12pct_clean",
        "svd_rank32_l31_clean",
    ):
        _write_cert(
            model_dir / "reports" / edit / "run_1" / "evaluation.report.json",
            validation={
                "invariants_pass": True,
                "primary_metric_acceptable": True,
                "spectral_stable": True,
                "rmt_stable": True,
                "preview_final_drift_acceptable": True,
                "guard_overhead_acceptable": True,
            },
        )

    # Stress edits (4): two catastrophic required to FAIL; two informational.
    for edit in ("prune_50pct_stress", "svd_rank32_stress"):
        _write_cert(
            model_dir / "reports" / edit / "run_1" / "evaluation.report.json",
            validation={
                "invariants_pass": True,
                "primary_metric_acceptable": False,
                "spectral_stable": True,
                "rmt_stable": True,
                "preview_final_drift_acceptable": True,
                "guard_overhead_acceptable": True,
            },
        )
    for edit in ("quant_4bit_stress", "fp8_e5m2_stress"):
        _write_cert(
            model_dir / "reports" / edit / "run_1" / "evaluation.report.json",
            validation={
                "invariants_pass": True,
                "primary_metric_acceptable": True,
                "spectral_stable": False,
                "rmt_stable": True,
                "preview_final_drift_acceptable": True,
                "guard_overhead_acceptable": True,
            },
        )

    # Error injections (11) => must be detected (not PASS).
    for error_type in (
        "nan_injection",
        "inf_injection",
        "shape_mismatch",
        "missing_tensors",
        "extreme_quant",
        "scale_explosion",
        "rank_collapse",
        "norm_collapse",
        "weight_tying_break",
        "rmt_norm_noise",
        "spectral_moderate_scale",
    ):
        invariants_status = "fail"
        _write_cert(
            model_dir / "reports" / "errors" / error_type / "evaluation.report.json",
            validation={
                "invariants_pass": False,
                "primary_metric_acceptable": False,
                "spectral_stable": False,
                "rmt_stable": False,
                "preview_final_drift_acceptable": False,
                "guard_overhead_acceptable": True,
            },
            invariants_status=invariants_status,
        )

    ve_cert = (
        model_dir
        / "reports"
        / "errors"
        / "ve_mlp_scale_skew"
        / "evaluation.report.json"
    )
    _write_cert(
        ve_cert,
        validation={
            "invariants_pass": True,
            "primary_metric_acceptable": True,
            "spectral_stable": True,
            "rmt_stable": True,
            "preview_final_drift_acceptable": True,
            "guard_overhead_acceptable": True,
        },
        invariants_status="pass",
    )
    _write_ve_probe(ve_cert.parent / "ve_probe.json", signal=True)

    verdict = _run_verdict(repo_root, output_dir)
    assert verdict["verdict"] == "PASS"
    counts = verdict["counts"]
    assert counts["models_total"] == 1
    assert counts["clean_total"] == 4
    assert counts["stress_total"] == 4
    assert counts["error_injection_total"] == 12
    assert counts["informational_stress_signaled"] == 2
    assert counts["primary_guard_required_scenarios"] == 5
    assert counts["primary_guard_required_hits"] == 5

    guard_summary = verdict["guard_signal_summary"]
    assert guard_summary["records_total"] == 20
    signals = guard_summary["signals"]
    assert signals["primary_metric"]["flagged"] == 13
    assert signals["primary_metric"]["unique"] == 2
    assert signals["spectral"]["flagged"] == 13
    assert signals["spectral"]["unique"] == 2
    assert signals["rmt"]["flagged"] == 11
    assert signals["rmt"]["unique"] == 0
    assert signals["invariants"]["flagged"] == 11
    assert signals["invariants"]["unique"] == 0

    category = verdict["category_summary"]
    assert category["clean"]["reports"] == 4
    assert category["clean"]["any_flag"] == 0
    assert category["stress"]["reports"] == 4
    assert category["stress"]["any_flag"] == 4
    assert category["error_injection"]["reports"] == 12
    assert category["error_injection"]["any_flag"] == 11


def test_verdict_contract_reports_guard_signal_uniqueness(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "run"
    model_dir = output_dir / "mistral-7b"

    _write_cert(
        model_dir / "reports" / "quant_4bit_clean" / "run_1" / "evaluation.report.json",
        validation={
            "invariants_pass": True,
            "primary_metric_acceptable": True,
            "spectral_stable": True,
            "rmt_stable": True,
            "preview_final_drift_acceptable": True,
            "guard_overhead_acceptable": True,
        },
    )
    _write_cert(
        model_dir
        / "reports"
        / "prune_50pct_stress"
        / "run_1"
        / "evaluation.report.json",
        validation={
            "invariants_pass": True,
            "primary_metric_acceptable": False,
            "spectral_stable": True,
            "rmt_stable": True,
            "preview_final_drift_acceptable": True,
            "guard_overhead_acceptable": True,
        },
    )
    _write_cert(
        model_dir / "reports" / "fp8_e5m2_stress" / "run_1" / "evaluation.report.json",
        validation={
            "invariants_pass": True,
            "primary_metric_acceptable": True,
            "spectral_stable": False,
            "rmt_stable": True,
            "preview_final_drift_acceptable": True,
            "guard_overhead_acceptable": True,
        },
    )
    _write_cert(
        model_dir / "reports" / "errors" / "nan_injection" / "evaluation.report.json",
        validation={
            "invariants_pass": False,
            "primary_metric_acceptable": True,
            "spectral_stable": True,
            "rmt_stable": True,
            "preview_final_drift_acceptable": True,
            "guard_overhead_acceptable": True,
        },
        invariants_status="fail",
    )

    verdict = _run_verdict(repo_root, output_dir)
    summary = verdict["guard_signal_summary"]["signals"]
    assert summary["invariants"] == {"flagged": 1, "unique": 1}
    assert summary["primary_metric"] == {"flagged": 1, "unique": 1}
    assert summary["spectral"] == {"flagged": 1, "unique": 1}
    assert summary["rmt"] == {"flagged": 0, "unique": 0}


def test_verdict_contract_fails_closed_when_no_reports(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "empty"
    output_dir.mkdir(parents=True, exist_ok=True)

    verdict = _run_verdict(repo_root, output_dir)
    assert verdict["verdict"] == "FAIL"
    assert verdict["counts"]["models_total"] == 0
    assert any(
        req.get("requirement") == "evidence_present"
        for req in verdict.get("failed_requirements", [])
    )


def test_verdict_contract_enforces_informational_stress_signal_fraction(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "run"
    model_dir = output_dir / "mistral-7b"

    for edit in (
        "quant_4bit_clean",
        "fp8_e5m2_clean",
        "prune_12pct_clean",
        "svd_rank32_l31_clean",
    ):
        _write_cert(
            model_dir / "reports" / edit / "run_1" / "evaluation.report.json",
            validation={
                "invariants_pass": True,
                "primary_metric_acceptable": True,
                "spectral_stable": True,
                "rmt_stable": True,
                "preview_final_drift_acceptable": True,
                "guard_overhead_acceptable": True,
            },
        )

    for edit in ("prune_50pct_stress", "svd_rank32_stress"):
        _write_cert(
            model_dir / "reports" / edit / "run_1" / "evaluation.report.json",
            validation={
                "invariants_pass": True,
                "primary_metric_acceptable": False,
                "spectral_stable": True,
                "rmt_stable": True,
                "preview_final_drift_acceptable": True,
                "guard_overhead_acceptable": True,
            },
        )

    # Informational stress edits intentionally PASS here to drive signal fraction to 0.0.
    for edit in ("quant_4bit_stress", "fp8_e5m2_stress"):
        _write_cert(
            model_dir / "reports" / edit / "run_1" / "evaluation.report.json",
            validation={
                "invariants_pass": True,
                "primary_metric_acceptable": True,
                "spectral_stable": True,
                "rmt_stable": True,
                "preview_final_drift_acceptable": True,
                "guard_overhead_acceptable": True,
            },
        )

    for error_type in (
        "nan_injection",
        "inf_injection",
        "shape_mismatch",
        "missing_tensors",
        "extreme_quant",
        "scale_explosion",
        "rank_collapse",
        "norm_collapse",
        "weight_tying_break",
        "rmt_norm_noise",
        "spectral_moderate_scale",
        "ve_mlp_scale_skew",
    ):
        _write_cert(
            model_dir / "reports" / "errors" / error_type / "evaluation.report.json",
            validation={
                "invariants_pass": False,
                "primary_metric_acceptable": False,
                "spectral_stable": False,
                "rmt_stable": False,
                "preview_final_drift_acceptable": False,
                "guard_overhead_acceptable": True,
            },
            invariants_status="fail",
        )
        if error_type == "ve_mlp_scale_skew":
            _write_ve_probe(
                model_dir / "reports" / "errors" / error_type / "ve_probe.json",
                signal=True,
            )

    verdict = _run_verdict(repo_root, output_dir)
    assert verdict["verdict"] == "FAIL"
    assert verdict["counts"]["informational_stress_total"] == 2
    assert verdict["counts"]["informational_stress_signaled"] == 0
    assert any(
        req.get("requirement") == "informational_stress_min_signal_fraction"
        for req in verdict.get("failed_requirements", [])
    )


def test_verdict_contract_requires_primary_guard_signal_for_marked_scenarios(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "run"
    model_dir = output_dir / "mistral-7b"

    _write_cert(
        model_dir / "reports" / "errors" / "rmt_norm_noise" / "evaluation.report.json",
        validation={
            "invariants_pass": True,
            "primary_metric_acceptable": True,
            "spectral_stable": True,
            "rmt_stable": True,
            "preview_final_drift_acceptable": True,
            "guard_overhead_acceptable": True,
        },
        invariants_status="pass",
    )

    verdict = _run_verdict(repo_root, output_dir)
    assert verdict["verdict"] == "FAIL"
    assert any(
        req.get("requirement") == "scenario_primary_guard_signal"
        and req.get("scenario") == "rmt_norm_noise"
        for req in verdict.get("failed_requirements", [])
    )


def test_verdict_contract_accepts_rmt_probe_sidecar_as_primary_guard_signal(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "run"
    cert_path = (
        output_dir
        / "mistral-7b"
        / "reports"
        / "errors"
        / "rmt_norm_noise"
        / "evaluation.report.json"
    )

    _write_cert(
        cert_path,
        validation={
            "invariants_pass": True,
            "primary_metric_acceptable": True,
            "spectral_stable": True,
            "rmt_stable": True,
            "preview_final_drift_acceptable": True,
            "guard_overhead_acceptable": True,
        },
        invariants_status="pass",
    )
    _write_rmt_probe(cert_path.parent / "rmt_probe.json", stable=False)

    verdict = _run_verdict(repo_root, output_dir)
    assert verdict["verdict"] == "FAIL"
    assert not any(
        req.get("requirement") == "scenario_primary_guard_signal"
        and req.get("scenario") == "rmt_norm_noise"
        for req in verdict.get("failed_requirements", [])
    )


def test_verdict_contract_accepts_spectral_caps_applied_as_primary_guard_signal(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "run"
    cert_path = (
        output_dir
        / "mistral-7b"
        / "reports"
        / "errors"
        / "spectral_moderate_scale"
        / "evaluation.report.json"
    )

    _write_cert(
        cert_path,
        validation={
            "invariants_pass": True,
            "primary_metric_acceptable": True,
            "spectral_stable": True,
            "rmt_stable": True,
            "preview_final_drift_acceptable": True,
            "guard_overhead_acceptable": True,
        },
        invariants_status="pass",
        spectral_caps_applied=2,
    )

    verdict = _run_verdict(repo_root, output_dir)
    assert verdict["verdict"] == "FAIL"
    assert not any(
        req.get("requirement") == "scenario_primary_guard_signal"
        and req.get("scenario") == "spectral_moderate_scale"
        for req in verdict.get("failed_requirements", [])
    )


def test_verdict_contract_accepts_ve_probe_sidecar_as_primary_guard_signal(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "run"
    cert_path = (
        output_dir
        / "mistral-7b"
        / "reports"
        / "errors"
        / "ve_mlp_scale_skew"
        / "evaluation.report.json"
    )

    _write_cert(
        cert_path,
        validation={
            "invariants_pass": True,
            "primary_metric_acceptable": True,
            "spectral_stable": True,
            "rmt_stable": True,
            "preview_final_drift_acceptable": True,
            "guard_overhead_acceptable": True,
        },
        invariants_status="pass",
    )
    _write_ve_probe(cert_path.parent / "ve_probe.json", signal=True)

    verdict = _run_verdict(repo_root, output_dir)
    assert verdict["verdict"] == "FAIL"
    assert not any(
        req.get("requirement") == "scenario_primary_guard_signal"
        and req.get("scenario") == "ve_mlp_scale_skew"
        for req in verdict.get("failed_requirements", [])
    )


def test_verdict_contract_supports_detectors_all_of(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "run"
    manifest_path = tmp_path / "scenarios.json"

    manifest_path.write_text(
        json.dumps(
            {
                "schema": "proof_pack_scenarios_v1",
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
                            # OR matches (probe stable=false), but AND also requires PM acceptable.
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
    _write_cert(
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
    _write_rmt_probe(cert_path.parent / "rmt_probe.json", stable=False)

    script = repo_root / "scripts/proof_packs/python/verdict_generator.py"
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
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))

    assert verdict["verdict"] == "FAIL"
    assert any(
        req.get("requirement") == "error_injection_detected"
        and req.get("scenario") == "demo_error"
        for req in verdict.get("failed_requirements", [])
    )
