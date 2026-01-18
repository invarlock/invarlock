from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def _write_cert(path: Path, *, validation: dict[str, Any], degraded: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "validation": validation,
        "primary_metric": {"degraded": degraded, "invalid": degraded},
        "guard_overhead": {"evaluated": True},
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_verdict_contract_clean_pass_catastrophic_fail_errors_detected(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "run"
    model_dir = output_dir / "mistral-7b"

    # Clean edits (4) => must PASS.
    for edit in (
        "quant_8bit_clean",
        "fp8_e4m3_clean",
        "prune_12pct_clean",
        "svd_rank256_clean",
    ):
        _write_cert(
            model_dir / "certificates" / edit / "run_1" / "evaluation.cert.json",
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
            model_dir / "certificates" / edit / "run_1" / "evaluation.cert.json",
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
            model_dir / "certificates" / edit / "run_1" / "evaluation.cert.json",
            validation={
                "invariants_pass": True,
                "primary_metric_acceptable": True,
                "spectral_stable": False,
                "rmt_stable": True,
                "preview_final_drift_acceptable": True,
                "guard_overhead_acceptable": True,
            },
        )

    # Error injections (5) => must be detected (not PASS).
    for error_type in (
        "nan_injection",
        "inf_injection",
        "extreme_quant",
        "scale_explosion",
        "weight_tying_break",
    ):
        _write_cert(
            model_dir / "certificates" / "errors" / error_type / "evaluation.cert.json",
            validation={
                "invariants_pass": False,
                "primary_metric_acceptable": False,
                "spectral_stable": False,
                "rmt_stable": False,
                "preview_final_drift_acceptable": False,
                "guard_overhead_acceptable": True,
            },
        )

    script = repo_root / "scripts/proof_packs/python/verdict_generator.py"
    subprocess.run(
        ["python3", str(script), "--output-dir", str(output_dir)],
        check=True,
        cwd=repo_root,
    )

    verdict_path = output_dir / "reports" / "final_verdict.json"
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    assert verdict["verdict"] == "PASS"

