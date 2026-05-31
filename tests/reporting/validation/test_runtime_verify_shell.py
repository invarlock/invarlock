from __future__ import annotations

from pathlib import Path

from invarlock import runtime_verify


def test_runtime_verify_success(monkeypatch, tmp_path: Path) -> None:
    report = tmp_path / "evaluation.report.json"
    manifest = tmp_path / "runtime.manifest.json"
    monkeypatch.setattr(runtime_verify, "verify_report_manifest", lambda *_: [])

    result = runtime_verify.verify_runtime_manifest(report, manifest)
    assert result.ok is True
    assert result.errors == ()
    assert result.report == str(report)
    assert result.manifest == str(manifest)


def test_runtime_verify_reports_verifier_errors(monkeypatch, tmp_path: Path) -> None:
    report = tmp_path / "evaluation.report.json"
    manifest = tmp_path / "runtime.manifest.json"
    monkeypatch.setattr(
        runtime_verify,
        "verify_report_manifest",
        lambda *_: ["bad digest", "missing runtime"],
    )

    result = runtime_verify.verify_runtime_manifest(report, manifest)
    assert result.ok is False
    assert result.errors == ("bad digest", "missing runtime")
