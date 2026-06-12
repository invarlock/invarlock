from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from scripts.evidence_packs.python.verdict.generator_helpers import _evaluate_report


def _write_cert(
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


def _write_ve_probe(
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


def _write_guard_value_manifest(
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


def _run_verdict(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    script = repo_root / "scripts/evidence_packs/python/verdict_generator.py"
    subprocess.run(
        ["python3", str(script), "--output-dir", str(output_dir)],
        check=True,
        cwd=repo_root,
    )
    verdict_path = output_dir / "reports" / "final_verdict.json"
    return json.loads(verdict_path.read_text(encoding="utf-8"))


def _run_verdict_with_manifest(
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


def test_verdict_report_fails_closed_on_missing_drift_flag() -> None:
    outcome = _evaluate_report(
        {
            "validation": {
                "invariants_pass": True,
                "primary_metric_acceptable": True,
                "spectral_stable": True,
                "rmt_stable": True,
                "guard_overhead_acceptable": True,
            },
            "primary_metric": {"degraded": False, "invalid": False},
            "guard_overhead": {"evaluated": False},
            "invariants": {"status": "pass"},
        }
    )

    assert outcome.passed is False
    assert "drift_fail" in outcome.reasons


def test_verdict_report_fails_closed_on_missing_evaluated_overhead_flag() -> None:
    outcome = _evaluate_report(
        {
            "validation": {
                "invariants_pass": True,
                "primary_metric_acceptable": True,
                "spectral_stable": True,
                "rmt_stable": True,
                "preview_final_drift_acceptable": True,
            },
            "primary_metric": {"degraded": False, "invalid": False},
            "guard_overhead": {"evaluated": True},
            "invariants": {"status": "pass"},
        }
    )

    assert outcome.passed is False
    assert "overhead_fail" in outcome.reasons


def test_verdict_contract_clean_pass_catastrophic_fail_errors_detected(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "run"
    model_dir = output_dir / "mistral-7b"
    shared_validation = {
        "invariants_pass": True,
        "primary_metric_acceptable": True,
        "spectral_stable": True,
        "rmt_stable": True,
        "preview_final_drift_acceptable": True,
        "guard_overhead_acceptable": True,
    }

    _write_cert(
        model_dir
        / "baseline_reports"
        / "ci_balanced_seq512_pv4_fn4"
        / "baseline_report.json",
        validation=shared_validation,
        invariants_status="pass",
        spectral_caps_applied=0,
        spectral_violations=[],
    )

    # Clean edits (4) => must PASS.
    for edit in (
        "quant_4bit_clean",
        "fp8_e5m2_clean",
        "prune_clean",
        "svd_rank32_clean",
    ):
        _write_cert(
            model_dir / "reports" / edit / "run_1" / "evaluation.report.json",
            validation=shared_validation,
        )

    # Stress edits (4): two catastrophic required to FAIL, one informational,
    # and FP8 as a PM-pass spectral-intervention demonstration.
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
    _write_cert(
        model_dir / "reports" / "quant_4bit_stress" / "run_1" / "evaluation.report.json",
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
        model_dir / "reports" / "fp8_e5m2_stress" / "run_1" / "evaluation.report.json",
        validation=shared_validation,
        spectral_caps_applied=2,
        spectral_violations=[
            ("model.layers.0.self_attn.q_proj", "attn", 7.7),
            ("model.layers.0.self_attn.k_proj", "attn", 3.2),
        ],
    )

    # Error injections (9) => must be detected (not PASS).
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

    rmt_cert = (
        model_dir / "reports" / "errors" / "rmt_norm_noise" / "evaluation.report.json"
    )
    _write_cert(
        rmt_cert,
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
    _write_rmt_probe(rmt_cert.parent / "rmt_probe.json", stable=False)

    spectral_cert = (
        model_dir
        / "reports"
        / "errors"
        / "spectral_moderate_scale"
        / "evaluation.report.json"
    )
    _write_cert(
        spectral_cert,
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
    # VE is a remediation guard: the evidence pack contract is that it runs and
    # produces a meaningful probe artifact. A conservative outcome (signal=false)
    # is acceptable as long as it proposed remediation scales.
    _write_ve_probe(
        ve_cert.parent / "ve_probe.json",
        signal=False,
        proposed_scales=32,
        would_enable=False,
        ab_gain=-0.1,
    )

    verdict = _run_verdict(repo_root, output_dir)
    assert verdict["verdict"] == "PASS"
    counts = verdict["counts"]
    assert counts["models_total"] == 1
    assert counts["clean_total"] == 4
    assert counts["stress_total"] == 4
    assert counts["error_injection_total"] == 12
    assert counts["informational_stress_signaled"] == 1
    assert counts["primary_guard_required_scenarios"] == 6
    assert counts["primary_guard_required_hits"] == 6

    guard_summary = verdict["guard_signal_summary"]
    assert guard_summary["records_total"] == 20
    signals = guard_summary["signals"]
    assert signals["primary_metric"]["flagged"] == 11
    assert signals["primary_metric"]["unique"] == 2
    assert signals["spectral"]["flagged"] == 10
    assert signals["spectral"]["unique"] == 1
    assert signals["rmt"]["flagged"] == 10
    assert signals["rmt"]["unique"] == 1
    assert signals["invariants"]["flagged"] == 9
    assert signals["invariants"]["unique"] == 0
    assert signals["variance"]["flagged"] == 1
    assert signals["variance"]["unique"] == 1

    interventions = verdict["guard_intervention_summary"]["signals"]
    assert interventions["spectral_caps"]["flagged"] == 2
    assert interventions["ve_signal"]["flagged"] == 1

    category = verdict["category_summary"]
    assert category["clean"]["reports"] == 4
    assert category["clean"]["any_flag"] == 0
    assert category["stress"]["reports"] == 4
    assert category["stress"]["any_flag"] == 3
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
            "spectral_stable": True,
            "rmt_stable": True,
            "preview_final_drift_acceptable": True,
            "guard_overhead_acceptable": True,
        },
        spectral_caps_applied=2,
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
    assert summary["spectral"] == {"flagged": 0, "unique": 0}
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
        "prune_clean",
        "svd_rank32_clean",
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

    # Informational stress edit intentionally PASSes here to drive signal
    # fraction to 0.0; FP8 still satisfies its required PM-pass guard signal.
    _write_cert(
        model_dir / "reports" / "quant_4bit_stress" / "run_1" / "evaluation.report.json",
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
        model_dir / "reports" / "fp8_e5m2_stress" / "run_1" / "evaluation.report.json",
        validation={
            "invariants_pass": True,
            "primary_metric_acceptable": True,
            "spectral_stable": True,
            "rmt_stable": True,
            "preview_final_drift_acceptable": True,
            "guard_overhead_acceptable": True,
        },
        spectral_caps_applied=2,
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
    assert verdict["counts"]["informational_stress_total"] == 1
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


def test_verdict_contract_rejects_baseline_only_spectral_guard_value(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "run"
    manifest_path = tmp_path / "scenarios.json"
    _write_guard_value_manifest(
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
    _write_cert(
        baseline_path,
        validation=validation,
        invariants_status="pass",
        spectral_caps_applied=2,
        spectral_violations=shared_violations,
    )
    _write_cert(
        subject_path,
        validation=validation,
        invariants_status="pass",
        spectral_caps_applied=2,
        spectral_violations=shared_violations,
    )

    verdict = _run_verdict_with_manifest(repo_root, output_dir, manifest_path)

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
    _write_guard_value_manifest(
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
    _write_cert(
        baseline_path,
        validation=validation,
        invariants_status="pass",
        spectral_caps_applied=2,
        spectral_violations=baseline_violations,
    )
    _write_cert(
        subject_path,
        validation=validation,
        invariants_status="pass",
        spectral_caps_applied=3,
        spectral_violations=subject_violations,
    )

    verdict = _run_verdict_with_manifest(repo_root, output_dir, manifest_path)

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
    _write_guard_value_manifest(
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
        model_dir
        / "reports"
        / "errors"
        / "rmt_norm_noise"
        / "evaluation.report.json"
    )
    validation = {
        "invariants_pass": True,
        "primary_metric_acceptable": True,
        "spectral_stable": True,
        "rmt_stable": False,
        "preview_final_drift_acceptable": True,
        "guard_overhead_acceptable": True,
    }
    _write_cert(baseline_path, validation=validation, invariants_status="pass")
    _write_cert(subject_path, validation=validation, invariants_status="pass")

    verdict = _run_verdict_with_manifest(repo_root, output_dir, manifest_path)

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
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))

    assert verdict["verdict"] == "FAIL"
    assert any(
        req.get("requirement") == "error_injection_detected"
        and req.get("scenario") == "demo_error"
        for req in verdict.get("failed_requirements", [])
    )
