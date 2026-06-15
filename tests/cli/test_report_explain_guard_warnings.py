from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import typer

from invarlock.cli.commands.explain_gates import explain_gates_command
from invarlock.cli.commands.verify import verify_command


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _cert_with_warning() -> dict:
    return {
        "schema_version": "v1",
        "run_id": "r-warning",
        "artifacts": {"generated_at": "t"},
        "plugins": {},
        "meta": {},
        "provenance": {"provider_digest": {"ids_sha256": "subject-ids"}},
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
            "kind": "ppl_causal",
            "preview": 9.0,
            "final": 9.0,
            "ratio_vs_baseline": 1.0,
            "display_ci": [1.0, 1.0],
        },
        "evaluation_windows": {
            "final": {"logloss": [math.log(9.0)], "token_counts": [1]}
        },
        "baseline_ref": {"primary_metric": {"kind": "ppl_causal", "final": 9.0}},
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
            "guard_warnings_present": True,
            "guard_warning_policy_acceptable": True,
        },
        "guard_warnings": {
            "present": True,
            "warning_count": 1,
            "warnings": [
                {
                    "guard": "spectral",
                    "kind": "new_capped_module",
                    "severity": "warning",
                    "family": "ffn",
                    "module": "layers.31.mlp.up_proj",
                    "baseline": {"capped": False},
                    "subject": {"capped": True, "z_score": 9.7},
                    "policy_gate": "pass",
                    "message": (
                        "Policy passes, but subject has a new capped module versus baseline."
                    ),
                }
            ],
        },
    }


def test_warning_policy_fail_turns_warning_into_verify_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cert_path = _write(tmp_path / "warning.json", _cert_with_warning())

    with pytest.raises(typer.Exit) as exc:
        verify_command(
            [cert_path],
            profile="dev",
            json_out=True,
            runtime_provenance="host",
            warning_policy="fail",
        )

    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"] == {"ok": False, "reason": "policy_fail"}
    assert payload["results"][0]["warning_count"] == 1
    assert getattr(exc.value, "exit_code", getattr(exc.value, "code", None)) == 1


def test_warning_policy_pass_keeps_default_verify_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cert_path = _write(tmp_path / "warning.json", _cert_with_warning())

    verify_command(
        [cert_path],
        profile="dev",
        json_out=False,
        runtime_provenance="host",
    )

    out = capsys.readouterr().out
    assert "Guard warnings present: 1" in out
    assert "VERIFY OK" in out


def test_report_explain_describes_guard_warning_separately_from_policy_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    subject = _write(tmp_path / "subject.report.json", {"ok": True})
    baseline = _write(tmp_path / "baseline.report.json", {"ok": True})
    monkeypatch.setattr(
        "invarlock.cli.commands.explain_gates.make_report",
        lambda _subject, _baseline: _cert_with_warning(),
    )

    explain_gates_command(
        subject_report=str(subject),
        baseline_report=str(baseline),
    )
    out = capsys.readouterr().out
    assert "Report Outline" in out
    assert "Guard Warnings: 1 [warn]; source=guard_warnings.warning_count" in out
    assert "Guard Warnings" in out
    assert "policy: pass" in out
    assert "strict warning mode" in out
