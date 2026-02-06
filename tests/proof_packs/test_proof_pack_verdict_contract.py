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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "validation": validation,
        "primary_metric": {"degraded": degraded, "invalid": degraded},
        "guard_overhead": {"evaluated": True},
        "invariants": {"status": invariants_status},
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

    verdict = _run_verdict(repo_root, output_dir)
    assert verdict["verdict"] == "PASS"
    counts = verdict["counts"]
    assert counts["models_total"] == 1
    assert counts["clean_total"] == 4
    assert counts["stress_total"] == 4
    assert counts["error_injection_total"] == 9


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


def test_verdict_contract_enforces_informational_stress_fraction(
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

    # Informational stress edits intentionally PASS here to drive fail fraction to 0.0.
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

    verdict = _run_verdict(repo_root, output_dir)
    assert verdict["verdict"] == "FAIL"
    assert verdict["counts"]["informational_stress_total"] == 2
    assert verdict["counts"]["informational_stress_fail"] == 0
    assert any(
        req.get("requirement") == "informational_stress_min_fail_fraction"
        for req in verdict.get("failed_requirements", [])
    )
