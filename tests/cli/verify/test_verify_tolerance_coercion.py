from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock.reporting.verify_contract import VerifyOutcome, run_verify_reports


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _mk_cert_accuracy_ok() -> dict:
    return {
        "schema_version": "v1",
        "run_id": "r",
        "artifacts": {"generated_at": "t"},
        "plugins": {},
        "meta": {},
        "dataset": {
            "provider": "unit",
            "seq_len": 8,
            "windows": {
                "preview": 1,
                "final": 1,
                "stats": {
                    "window_match_fraction": 1.0,
                    "window_overlap_fraction": 0.0,
                    "coverage": {"preview": {"used": 1}, "final": {"used": 1}},
                    "paired_windows": 1,
                },
            },
        },
        "primary_metric": {
            "kind": "accuracy",
            "final": 0.8,
            "preview": 0.8,
            "delta_vs_baseline_pp": 0.0,
            "display_ci": [0.8, 0.8],
        },
        "metrics": {"classification": {"n_correct": 8, "n_total": 10}},
        "baseline_ref": {"primary_metric": {"kind": "accuracy", "final": 0.75}},
    }


@pytest.mark.parametrize(
    "unsafe_tolerance",
    ["not-a-float", float("nan"), float("inf"), -1e-12, 1e-6, 1.0],
)
def test_verify_rejects_unsafe_recompute_tolerance(
    tmp_path: Path, unsafe_tolerance: object
) -> None:
    cert = _mk_cert_accuracy_ok()
    p = _write(tmp_path / "acc.json", cert)

    result = run_verify_reports(
        [p],
        profile="dev",
        json_mode=True,
        tolerance=unsafe_tolerance,  # type: ignore[arg-type]
    )

    assert result.outcome == VerifyOutcome.POLICY_FAIL
    assert "finite number between 0 and 1e-9" in str(result.error)
