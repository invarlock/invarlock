from __future__ import annotations

from pathlib import Path

from invarlock import runtime_verify
from invarlock.runtime_security import (
    CONTAINER_EXECUTION_ENV,
    RUNTIME_IMAGE_DIGEST_ENV,
    RUNTIME_IMAGE_ENV,
    write_runtime_manifest,
)

_TEST_IMAGE_DIGEST = "sha256:" + ("a" * 64)


def _write_bound_inputs(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    monkeypatch.setenv(CONTAINER_EXECUTION_ENV, "1")
    monkeypatch.setenv(RUNTIME_IMAGE_ENV, "ghcr.io/invarlock/runtime:test")
    monkeypatch.setenv(RUNTIME_IMAGE_DIGEST_ENV, _TEST_IMAGE_DIGEST)
    report = tmp_path / "evaluation.report.json"
    report.write_text('{"schema_version":"v1"}\n', encoding="utf-8")
    return report, write_runtime_manifest(report)


def test_runtime_verify_success(monkeypatch, tmp_path: Path) -> None:
    report, manifest = _write_bound_inputs(tmp_path, monkeypatch)

    result = runtime_verify.verify_runtime_manifest(
        report,
        manifest,
        expected_image_digest=_TEST_IMAGE_DIGEST,
    )
    assert result.ok is True
    assert result.errors == ()
    assert result.report == str(report)
    assert result.manifest == str(manifest)
    assert result.binding_verified is True
    assert result.expected_digest_matched is True


def test_runtime_verify_reports_verifier_errors(monkeypatch, tmp_path: Path) -> None:
    report, manifest = _write_bound_inputs(tmp_path, monkeypatch)
    report.write_text('{"schema_version":"v2"}\n', encoding="utf-8")

    result = runtime_verify.verify_runtime_manifest(report, manifest)
    assert result.ok is False
    assert any("report digest mismatch" in error for error in result.errors)
    assert result.binding_verified is False
