from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.evidence_catalog import catalog_digest
from tests.reporting._support_evidence_pack_paths import (
    _build_pack as _build_evidence_pack,
)
from tests.reporting._support_evidence_pack_paths import (
    _build_report_payload,
    _successful_verify_result,
    evidence_pack_mod,
)

_ALLOW_UNVERIFIED_PROVENANCE_ENV = {"INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE": "1"}
ROOT = Path(__file__).resolve().parents[2]


def _build_pack(pack_dir: Path, *, report_rel_path: str) -> Path:
    return _build_evidence_pack(
        pack_dir,
        report_rel_path=report_rel_path,
        report_payload=_build_report_payload(),
    )


def test_evidence_pack_help_lists_verify() -> None:
    result = CliRunner().invoke(app, ["advanced", "evidence-pack", "--help"])
    assert result.exit_code == 0
    assert "verify" in result.output
    assert "inspect" in result.output
    assert "verify-set" in result.output
    assert "build" not in result.output
    assert "keygen" not in result.output
    assert "seal" not in result.output


def test_evidence_catalog_validate_emits_only_stable_public_fields() -> None:
    catalog = ROOT / "contracts" / "evidence_catalog_v1.json"

    result = CliRunner().invoke(
        app, ["advanced", "evidence-catalog", "validate", str(catalog), "--json"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload == {
        "format_version": "evidence-catalog-validate-v1",
        "ok": True,
        "catalog_digest": catalog_digest(catalog),
        "entry_count": 39,
        "entry_ids": sorted(payload["entry_ids"]),
        "errors": [],
    }
    assert str(ROOT) not in result.stdout


def test_evidence_catalog_validate_redacts_invalid_input_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing-catalog.json"

    result = CliRunner().invoke(
        app, ["advanced", "evidence-catalog", "validate", str(missing), "--json"]
    )

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout) == {
        "format_version": "evidence-catalog-validate-v1",
        "ok": False,
        "catalog_digest": None,
        "entry_count": 0,
        "entry_ids": [],
        "errors": ["catalog_invalid"],
    }
    assert str(tmp_path) not in result.stdout


def test_evidence_pack_verify_strict_rejects_pack_without_signed_baseline(
    monkeypatch, tmp_path: Path
) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )
    json_out = tmp_path / "verify.json"

    monkeypatch.setattr(
        "invarlock.evidence_pack._run_verify_command",
        lambda reports, profile, report_assurance="report": _successful_verify_result(
            reports
        ),
        raising=False,
    )

    result = CliRunner().invoke(
        app,
        [
            "advanced",
            "evidence-pack",
            "verify",
            str(pack_dir),
            "--json",
            "--json-out",
            str(json_out),
            "--report-assurance",
            "strict",
        ],
        env=_ALLOW_UNVERIFIED_PROVENANCE_ENV,
    )

    assert result.exit_code == 6, result.output
    payload = json.loads(result.stdout.strip())
    assert payload["format_version"] == "evidence-pack-verify-v1"
    assert payload["ok"] is False
    assert payload["report_assurance"] == "strict"
    assert any("verification_baselines" in error for error in payload["errors"])
    assert not json_out.exists()


def test_evidence_pack_verify_human_success(monkeypatch, tmp_path: Path) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )
    monkeypatch.setattr(
        "invarlock.evidence_pack._run_verify_command",
        lambda reports, profile, report_assurance="report": _successful_verify_result(
            reports
        ),
        raising=False,
    )

    result = CliRunner().invoke(
        app,
        ["advanced", "evidence-pack", "verify", str(pack_dir)],
        env=_ALLOW_UNVERIFIED_PROVENANCE_ENV,
    )

    assert result.exit_code == 0, result.output
    assert "warning(s)" in result.output
    assert "Evidence pack verified" in result.output


def test_evidence_pack_verify_human_failure_renders_errors(
    monkeypatch, tmp_path: Path
) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )
    monkeypatch.setattr(
        "invarlock.cli.commands.evidence_pack.verify_evidence_pack",
        lambda *args, **kwargs: SimpleNamespace(
            payload={
                "pack": str(pack_dir),
                "ok": False,
                "warnings": [],
                "errors": ["bad pack"],
            },
            status=evidence_pack_mod.EvidencePackStatus.INTEGRITY,
        ),
        raising=False,
    )

    result = CliRunner().invoke(
        app, ["advanced", "evidence-pack", "verify", str(pack_dir)]
    )

    assert result.exit_code == 6, result.output
    assert "Evidence pack verification failed" in result.output
    assert "bad pack" in result.output


def test_evidence_pack_verify_integrity_only_is_not_report_verification(
    monkeypatch, tmp_path: Path
) -> None:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    monkeypatch.setattr(
        "invarlock.cli.commands.evidence_pack.verify_evidence_pack",
        lambda *args, **kwargs: SimpleNamespace(
            payload={
                "pack": str(pack_dir),
                "ok": False,
                "integrity_ok": True,
                "reports_verified": False,
                "verification_scope": "integrity_only",
                "assurance_status": "not_verified",
                "warnings": [],
                "errors": [],
            },
            status=evidence_pack_mod.EvidencePackStatus.INTEGRITY_ONLY,
        ),
        raising=False,
    )

    result = CliRunner().invoke(
        app,
        ["advanced", "evidence-pack", "verify", str(pack_dir), "--skip-verify"],
    )

    assert result.exit_code == evidence_pack_mod.EvidencePackStatus.INTEGRITY_ONLY
    assert "integrity inspection completed" in result.output
    assert "not report assurance" in result.output
    assert "Evidence pack verified" not in result.output
    assert "[PASS]" not in result.output


def test_evidence_pack_verify_json_round_trip_with_verify_payload(
    monkeypatch, tmp_path: Path
) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )
    monkeypatch.setattr(
        "invarlock.evidence_pack._run_verify_command",
        lambda reports, profile, report_assurance="report": _successful_verify_result(
            reports
        ),
        raising=False,
    )

    result = CliRunner().invoke(
        app,
        ["advanced", "evidence-pack", "verify", str(pack_dir), "--json"],
        env=_ALLOW_UNVERIFIED_PROVENANCE_ENV,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout.strip())
    assert payload["format_version"] == "evidence-pack-verify-v1"
    assert payload["ok"] is True
    assert payload["verify"]["format_version"] == "verify-v1"
    assert payload["verify"]["ok"] is True


def test_evidence_pack_verify_json_round_trip_with_real_nested_verify(
    tmp_path: Path,
) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )

    result = CliRunner().invoke(
        app,
        ["advanced", "evidence-pack", "verify", str(pack_dir), "--json"],
        env=_ALLOW_UNVERIFIED_PROVENANCE_ENV,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout.strip())
    assert payload["format_version"] == "evidence-pack-verify-v1"
    assert payload["ok"] is True
    assert payload["verify"]["format_version"] == "verify-v1"
    assert payload["verify"]["summary"]["reason"] == "ok"
    assert payload["verify"]["summary"]["ok"] is True


def test_evidence_pack_verify_rejects_missing_pack(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["advanced", "evidence-pack", "verify", str(tmp_path / "missing"), "--json"],
    )

    assert result.exit_code == 3
    payload = json.loads(result.stdout.strip())
    assert payload["ok"] is False
    assert "resolution" not in payload


def test_evidence_pack_verify_human_failure(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["advanced", "evidence-pack", "verify", str(tmp_path / "missing")]
    )

    assert result.exit_code == 3
    assert "Evidence pack verification failed" in result.output
    assert "Pack directory not found" in result.output


def test_evidence_pack_inspect_json_summary(tmp_path: Path) -> None:
    pack_dir = _build_pack(
        tmp_path / "pack",
        report_rel_path="reports/model/clean/noop/evaluation.report.json",
    )

    result = CliRunner().invoke(
        app, ["advanced", "evidence-pack", "inspect", str(pack_dir), "--json"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout.strip())
    assert payload["format_version"] == "evidence-pack-inspect-v1"
    assert payload["ok"] is True
    assert payload["reports"]["total"] == 1
    assert payload["reports"]["clean"] == 1
    assert payload["reports"]["errors"] == 0
    assert payload["signature"]["present"] is False
    assert payload["integrity"]["checksums_bound"] is True
    assert payload["integrity"]["manifest_provenance_ok"] is True
    assert payload["strict_ready"] is False
    assert payload["evidence_level"] == "medium"
    assert any(
        "manifest.signature.json missing" in issue for issue in payload["issues"]
    )


def test_evidence_pack_inspect_human_success_and_failure(
    monkeypatch, tmp_path: Path
) -> None:
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()

    monkeypatch.setattr(
        "invarlock.cli.commands.evidence_pack.inspect_evidence_pack",
        lambda path: SimpleNamespace(
            payload={
                "pack": str(path),
                "ok": True,
                "issues": ["unsigned"],
            },
            status=evidence_pack_mod.EvidencePackStatus.OK,
        ),
        raising=False,
    )
    result = CliRunner().invoke(
        app, ["advanced", "evidence-pack", "inspect", str(pack_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "Evidence pack inspected" in result.output
    assert "unsigned" in result.output

    monkeypatch.setattr(
        "invarlock.cli.commands.evidence_pack.inspect_evidence_pack",
        lambda path: SimpleNamespace(
            payload={
                "pack": str(path),
                "ok": False,
                "issues": ["missing manifest"],
            },
            status=evidence_pack_mod.EvidencePackStatus.FORMAT,
        ),
        raising=False,
    )
    result = CliRunner().invoke(
        app, ["advanced", "evidence-pack", "inspect", str(pack_dir)]
    )
    assert result.exit_code == 4, result.output
    assert "Evidence pack inspection failed" in result.output
    assert "missing manifest" in result.output
