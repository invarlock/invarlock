from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import invarlock.evidence_pack as evidence_pack_mod
from invarlock.cli.app import app


def test_evidence_pack_build_forwards_report_assurance_and_release_review(
    monkeypatch, tmp_path: Path
) -> None:
    final_verdict = tmp_path / "final_verdict.json"
    report = tmp_path / "evaluation.report.json"
    final_verdict.write_text('{"verdict":"PASS"}', encoding="utf-8")
    report.write_text("{}", encoding="utf-8")
    seen: dict[str, object] = {}

    def _fake_build(*args, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            payload={
                "pack": str(tmp_path / "pack"),
                "ok": True,
                "warnings": [],
                "errors": [],
                "reports": {"total": 1},
                "verify": {"ok": True},
                "files": {"hashed": 2},
            },
            status=evidence_pack_mod.EvidencePackStatus.OK,
        )

    monkeypatch.setattr(
        "invarlock.cli.commands.evidence_pack.build_evidence_pack",
        _fake_build,
        raising=False,
    )
    result = CliRunner().invoke(
        app,
        [
            "advanced",
            "evidence-pack",
            "build",
            str(tmp_path / "pack"),
            "--final-verdict",
            str(final_verdict),
            "--report",
            str(report),
            "--profile",
            "ci",
            "--report-assurance",
            "strict",
            "--release-review",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["profile"] == "ci"
    assert seen["report_assurance"] == "strict"
    assert seen["release_review"] is True
