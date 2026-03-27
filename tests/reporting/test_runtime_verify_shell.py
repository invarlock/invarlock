from __future__ import annotations

import json
from pathlib import Path

from invarlock import runtime_verify


def test_runtime_verify_json_success(monkeypatch, capsys, tmp_path: Path) -> None:
    report = tmp_path / "evaluation.report.json"
    manifest = tmp_path / "runtime.manifest.json"
    monkeypatch.setattr(runtime_verify, "_verify_report_manifest", lambda *_: [])

    exit_code = runtime_verify.main(
        ["--report", str(report), "--manifest", str(manifest), "--json"]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "ok": True,
        "errors": [],
        "report": str(report),
        "manifest": str(manifest),
    }


def test_runtime_verify_json_failure(monkeypatch, capsys, tmp_path: Path) -> None:
    report = tmp_path / "evaluation.report.json"
    manifest = tmp_path / "runtime.manifest.json"
    monkeypatch.setattr(
        runtime_verify, "_verify_report_manifest", lambda *_: ["bad digest"]
    )

    exit_code = runtime_verify.main(
        ["--report", str(report), "--manifest", str(manifest), "--json"]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "ok": False,
        "errors": ["bad digest"],
        "report": str(report),
        "manifest": str(manifest),
    }


def test_runtime_verify_human_success(monkeypatch, capsys, tmp_path: Path) -> None:
    report = tmp_path / "evaluation.report.json"
    manifest = tmp_path / "runtime.manifest.json"
    monkeypatch.setattr(runtime_verify, "_verify_report_manifest", lambda *_: [])

    exit_code = runtime_verify.main(
        ["--report", str(report), "--manifest", str(manifest)]
    )

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == (
        f"runtime verify ok report={report} manifest={manifest}"
    )


def test_runtime_verify_human_failure(monkeypatch, capsys, tmp_path: Path) -> None:
    report = tmp_path / "evaluation.report.json"
    manifest = tmp_path / "runtime.manifest.json"
    monkeypatch.setattr(
        runtime_verify,
        "_verify_report_manifest",
        lambda *_: ["bad digest", "missing runtime"],
    )

    exit_code = runtime_verify.main(
        ["--report", str(report), "--manifest", str(manifest)]
    )

    assert exit_code == 1
    assert capsys.readouterr().out.splitlines() == ["bad digest", "missing runtime"]
