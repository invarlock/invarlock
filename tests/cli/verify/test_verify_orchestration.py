from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import typer

from invarlock.cli.commands.verify import verify_command
from invarlock.reporting import verify_adapter_family
from invarlock.reporting import verify_contract as verify_mod
from invarlock.reporting.verify_contract import VerifyOutcome


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _ppl_cert() -> dict:
    return {
        "schema_version": "v1",
        "run_id": "r-ppl",
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
        },
    }


def _accuracy_cert() -> dict:
    return {
        "schema_version": "v1",
        "run_id": "r-acc",
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
            "kind": "accuracy",
            "final": 0.8,
            "delta_vs_baseline_pp": 0.0,
            "display_ci": [0.8, 0.8],
        },
        "metrics": {"classification": {"n_correct": 8, "n_total": 10}},
        "baseline_ref": {"primary_metric": {"kind": "accuracy", "final": 0.8}},
        "validation": {
            "primary_metric_acceptable": True,
            "preview_final_drift_acceptable": True,
            "invariants_pass": True,
            "spectral_stable": True,
            "rmt_stable": True,
        },
    }


@pytest.mark.parametrize(
    "baseline_payload",
    [
        {"provenance": []},
        {"provenance": {"provider_digest": []}},
    ],
)
def test_verify_ignores_non_dict_baseline_provider_digest(
    tmp_path: Path,
    baseline_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cert_path = _write(tmp_path / "subject.json", _ppl_cert())
    baseline_path = _write(tmp_path / "baseline.json", baseline_payload)
    monkeypatch.setattr(
        verify_mod, "_validate_evaluation_report_payload", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        verify_adapter_family,
        "warn_adapter_family_mismatch",
        lambda *args, **kwargs: None,
        raising=True,
    )

    with pytest.raises(typer.Exit) as exc:
        verify_command([cert_path], baseline=baseline_path, profile="ci", json_out=True)

    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"] == {"ok": True, "reason": "ok"}
    assert "resolution" not in payload
    assert getattr(exc.value, "exit_code", getattr(exc.value, "code", None)) == 0


def test_verify_accuracy_missing_aggregates_after_provider_digest(
    tmp_path: Path,
) -> None:
    cert = _accuracy_cert()
    cert["metrics"]["classification"] = {}
    cert_path = _write(tmp_path / "missing-aggregates.json", cert)

    with pytest.raises(typer.Exit) as exc:
        verify_command([cert_path], baseline=None, profile="release", json_out=True)

    assert getattr(exc.value, "exit_code", getattr(exc.value, "code", None)) == 2


def test_verify_dev_rejects_incomplete_ppl_final_window(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cert = _ppl_cert()
    cert["evaluation_windows"] = {"final": {}}
    cert_path = _write(tmp_path / "incomplete-final.json", cert)

    with pytest.raises(SystemExit) as exc:
        verify_command([cert_path], baseline=None, profile="dev", json_out=False)

    out = capsys.readouterr().out
    assert exc.value.code == 1
    assert "requires complete final logloss and token-count evidence" in out
    assert "[FAIL]" in out


def test_verify_dev_rejects_non_finite_raw_ppl_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cert = _ppl_cert()
    cert["evaluation_windows"] = {"final": {"logloss": ["bad"], "token_counts": [1]}}
    cert_path = _write(tmp_path / "bad-float.json", cert)

    with pytest.raises(SystemExit) as exc:
        verify_command([cert_path], baseline=None, profile="dev", json_out=False)

    out = capsys.readouterr().out
    assert exc.value.code == 1
    assert "evaluation_windows.final.logloss[0] must be finite" in out
    assert "[FAIL]" in out


def test_verify_human_success_line_swallows_render_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cert_path = _write(tmp_path / "subject.json", _ppl_cert())

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("render failed")

    monkeypatch.setattr(
        verify_mod._verify_output, "build_verify_success_line", _boom, raising=True
    )

    verify_command([cert_path], baseline=None, profile="dev", json_out=False)

    out = capsys.readouterr().out
    assert "PASS" in out
    assert "VERIFY OK" not in out


def test_verify_human_generic_exception_prints_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cert_path = _write(tmp_path / "subject.json", _ppl_cert())

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(
        verify_mod, "_validate_evaluation_report_payload", _boom, raising=True
    )

    with pytest.raises(SystemExit) as exc:
        verify_command([cert_path], baseline=None, profile="dev", json_out=False)

    out = capsys.readouterr().out
    assert "Verification failed: unexpected failure" in out
    assert getattr(exc.value, "exit_code", getattr(exc.value, "code", None)) == 1


def test_verify_reports_contract_returns_structured_payload_without_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cert_path = _write(tmp_path / "subject.json", _ppl_cert())

    result = verify_mod.verify_reports_contract(
        [cert_path],
        baseline=None,
        profile="dev",
        json_mode=True,
    )

    assert result.outcome == VerifyOutcome.OK
    payload = result.payload
    assert payload["summary"] == {"ok": True, "reason": "ok"}
    assert "resolution" not in payload
    assert capsys.readouterr().out == ""


def test_run_verify_reports_collects_human_lines_without_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cert_path = _write(tmp_path / "subject.json", _ppl_cert())

    result = verify_mod.run_verify_reports(
        [cert_path],
        baseline=None,
        profile="dev",
        json_mode=False,
    )

    assert result.outcome == VerifyOutcome.OK
    assert any(
        diag.level == "pass" and "subject.json" in diag.message
        for diag in result.diagnostics
    )
    assert any(
        diag.level == "info" and "VERIFY OK" in diag.message
        for diag in result.diagnostics
    )
    assert capsys.readouterr().out == ""
