from __future__ import annotations

from pathlib import Path

from invarlock import __version__
from scripts.evidence_packs.python.verdict.generator_helpers import _evaluate_report
from tests.evidence_packs._verdict_contract_support import (
    run_verdict,
    write_cert,
    write_rmt_probe,
    write_ve_probe,
)


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

    write_cert(
        model_dir
        / "baseline_reports"
        / "ci_balanced_seq512_pv4_fn4"
        / "baseline_report.json",
        validation=shared_validation,
        invariants_status="pass",
        spectral_caps_applied=0,
        spectral_violations=[],
    )

    # Clean edits (6) => must PASS.
    for edit in (
        "quant_4bit_clean",
        "fp8_e5m2_clean",
        "prune_clean",
        "svd_rank32_clean",
        "lora_rank4_clean",
        "fine_tune_step1_clean",
    ):
        write_cert(
            model_dir / "reports" / edit / "run_1" / "evaluation.report.json",
            validation=shared_validation,
        )

    # Stress edits (6): two catastrophic required to FAIL and four informational,
    # with FP8 retained as a historical PM-pass spectral-intervention report.
    for edit in ("prune_50pct_stress", "svd_rank32_stress"):
        write_cert(
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
    write_cert(
        model_dir
        / "reports"
        / "quant_4bit_stress"
        / "run_1"
        / "evaluation.report.json",
        validation={
            "invariants_pass": True,
            "primary_metric_acceptable": True,
            "spectral_stable": False,
            "rmt_stable": True,
            "preview_final_drift_acceptable": True,
            "guard_overhead_acceptable": True,
        },
    )
    write_cert(
        model_dir / "reports" / "fp8_e5m2_stress" / "run_1" / "evaluation.report.json",
        validation=shared_validation,
        spectral_caps_applied=2,
        spectral_violations=[
            ("model.layers.0.self_attn.q_proj", "attn", 7.7),
            ("model.layers.0.self_attn.k_proj", "attn", 3.2),
        ],
    )
    for edit in ("lora_rank8_stress", "fine_tune_step3_stress"):
        write_cert(
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
        write_cert(
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
    write_cert(
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
    write_rmt_probe(rmt_cert.parent / "rmt_probe.json", stable=False)

    targeted_rmt_cert = (
        model_dir
        / "reports"
        / "errors"
        / "rmt_norm_noise_l31_ffn_up_b030"
        / "evaluation.report.json"
    )
    write_cert(
        targeted_rmt_cert,
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
    write_rmt_probe(targeted_rmt_cert.parent / "rmt_probe.json", stable=False)

    spectral_cert = (
        model_dir
        / "reports"
        / "errors"
        / "spectral_moderate_scale"
        / "evaluation.report.json"
    )
    write_cert(
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
    targeted_spectral_cert = (
        model_dir
        / "reports"
        / "errors"
        / "spectral_moderate_scale_mlp_l31_up_s112"
        / "evaluation.report.json"
    )
    write_cert(
        targeted_spectral_cert,
        validation={
            "invariants_pass": True,
            "primary_metric_acceptable": True,
            "spectral_stable": True,
            "rmt_stable": True,
            "preview_final_drift_acceptable": True,
            "guard_overhead_acceptable": True,
        },
        invariants_status="pass",
        spectral_caps_applied=1,
        spectral_violations=[
            ("model.layers.31.mlp.up_proj", "ffn", 3.9),
        ],
    )

    ve_cert = (
        model_dir
        / "reports"
        / "errors"
        / "ve_mlp_scale_skew"
        / "evaluation.report.json"
    )
    write_cert(
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
    write_ve_probe(
        ve_cert.parent / "ve_probe.json",
        signal=False,
        proposed_scales=32,
        would_enable=False,
        ab_gain=-0.1,
    )

    targeted_ve_cert = (
        model_dir
        / "reports"
        / "errors"
        / "ve_mlp_scale_skew_l31_down_s090"
        / "evaluation.report.json"
    )
    write_cert(
        targeted_ve_cert,
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
    write_ve_probe(
        targeted_ve_cert.parent / "ve_probe.json",
        signal=False,
        proposed_scales=32,
        would_enable=False,
        ab_gain=-0.1,
    )

    verdict = run_verdict(repo_root, output_dir)
    verdict_text = (output_dir / "reports" / "final_verdict.txt").read_text(
        encoding="utf-8"
    )
    assert verdict["verdict"] == "PASS"
    assert verdict_text.startswith("INVARLOCK EVIDENCE PACK - FINAL VERDICT")
    assert f"InvarLock {__version__}" in verdict_text
    assert "Auditable verification for edited model checkpoints." in verdict_text
    counts = verdict["counts"]
    assert counts["models_total"] == 1
    assert counts["clean_total"] == 6
    assert counts["stress_total"] == 6
    assert counts["error_injection_total"] == 15
    assert counts["informational_stress_signaled"] == 3
    assert counts["primary_guard_required_scenarios"] == 8
    assert counts["primary_guard_required_hits"] == 8

    guard_summary = verdict["guard_signal_summary"]
    assert guard_summary["records_total"] == 27
    signals = guard_summary["signals"]
    assert signals["primary_metric"]["flagged"] == 13
    assert signals["primary_metric"]["unique"] == 4
    assert signals["spectral"]["flagged"] == 10
    assert signals["spectral"]["unique"] == 1
    assert signals["rmt"]["flagged"] == 11
    assert signals["rmt"]["unique"] == 2
    assert signals["invariants"]["flagged"] == 9
    assert signals["invariants"]["unique"] == 0
    assert signals["variance"]["flagged"] == 2
    assert signals["variance"]["unique"] == 2

    interventions = verdict["guard_intervention_summary"]["signals"]
    assert interventions["spectral_caps"]["flagged"] == 3
    assert interventions["ve_signal"]["flagged"] == 2

    category = verdict["category_summary"]
    assert category["clean"]["reports"] == 6
    assert category["clean"]["any_flag"] == 0
    assert category["stress"]["reports"] == 6
    assert category["stress"]["any_flag"] == 5
    assert category["error_injection"]["reports"] == 15
    assert category["error_injection"]["any_flag"] == 13
    assert verdict["manifest"]["path"] == "scripts/evidence_packs/scenarios.json"
    for record in verdict["records"]:
        assert not Path(record["path"]).is_absolute()
        baseline_report = record.get("baseline_report")
        if baseline_report:
            assert not Path(str(baseline_report)).is_absolute()


def test_verdict_contract_reports_guard_signal_uniqueness(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "run"
    model_dir = output_dir / "mistral-7b"

    write_cert(
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
    write_cert(
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
    write_cert(
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
    write_cert(
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

    verdict = run_verdict(repo_root, output_dir)
    summary = verdict["guard_signal_summary"]["signals"]
    assert summary["invariants"] == {"flagged": 1, "unique": 1}
    assert summary["primary_metric"] == {"flagged": 1, "unique": 1}
    assert summary["spectral"] == {"flagged": 0, "unique": 0}
    assert summary["rmt"] == {"flagged": 0, "unique": 0}


def test_verdict_contract_fails_closed_when_no_reports(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "empty"
    output_dir.mkdir(parents=True, exist_ok=True)

    verdict = run_verdict(repo_root, output_dir)
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
        write_cert(
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
        write_cert(
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

    # Informational stress edits intentionally PASS here to drive signal
    # fraction to 0.0; FP8 is historical context, not a required guard signal.
    write_cert(
        model_dir
        / "reports"
        / "quant_4bit_stress"
        / "run_1"
        / "evaluation.report.json",
        validation={
            "invariants_pass": True,
            "primary_metric_acceptable": True,
            "spectral_stable": True,
            "rmt_stable": True,
            "preview_final_drift_acceptable": True,
            "guard_overhead_acceptable": True,
        },
    )
    write_cert(
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
        write_cert(
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
            write_ve_probe(
                model_dir / "reports" / "errors" / error_type / "ve_probe.json",
                signal=True,
            )

    verdict = run_verdict(repo_root, output_dir)
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

    write_cert(
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

    verdict = run_verdict(repo_root, output_dir)
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

    write_cert(
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
    write_rmt_probe(cert_path.parent / "rmt_probe.json", stable=False)

    verdict = run_verdict(repo_root, output_dir)
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

    write_cert(
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

    verdict = run_verdict(repo_root, output_dir)
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

    write_cert(
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
    write_ve_probe(cert_path.parent / "ve_probe.json", signal=True)

    verdict = run_verdict(repo_root, output_dir)
    assert verdict["verdict"] == "FAIL"
    assert not any(
        req.get("requirement") == "scenario_primary_guard_signal"
        and req.get("scenario") == "ve_mlp_scale_skew"
        for req in verdict.get("failed_requirements", [])
    )
