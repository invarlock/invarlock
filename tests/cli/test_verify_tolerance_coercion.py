from __future__ import annotations

import json
from pathlib import Path

import typer

from invarlock.cli.commands.verify import verify_command


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _mk_cert_accuracy_ok() -> dict:
    return {
        "schema_version": "v1",
        "run_id": "r",
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
            "ratio_vs_baseline": 0.0,
            "display_ci": [0.8, 0.8],
        },
        "metrics": {"classification": {"n_correct": 8, "n_total": 10}},
        "baseline_ref": {"primary_metric": {"kind": "accuracy", "final": 0.75}},
    }


def test_verify_tolerance_coercion_string_ok(tmp_path: Path) -> None:
    cert = _mk_cert_accuracy_ok()
    p = _write(tmp_path / "acc.json", cert)
    # Pass a non-float tolerance; the command should coerce or fallback without crashing
    try:
        verify_command(
            [p], baseline=None, profile="dev", json_out=True, tolerance="not-a-float"
        )  # type: ignore[arg-type]
    except typer.Exit as e:
        assert getattr(e, "exit_code", getattr(e, "code", None)) == 0
