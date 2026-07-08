from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from invarlock.public_contracts import load_runtime_manifest_schema
from invarlock.runtime_security import (
    CONTAINER_EXECUTION_ENV,
    RUNTIME_IMAGE_DIGEST_ENV,
    RUNTIME_IMAGE_ENV,
    write_runtime_manifest,
)
from invarlock.runtime_verify import verify_report_manifest

_VALID_TEST_IMAGE_DIGEST = "sha256:" + ("a" * 64)


def _write_valid_report_and_manifest(
    tmp_path: Path, *, report_bytes: bytes | None = None
) -> tuple[Path, Path]:
    report_path = tmp_path / "evaluation.report.json"
    payload = b'{"schema_version":"v1"}\n' if report_bytes is None else report_bytes
    report_path.write_bytes(payload)
    manifest_path = write_runtime_manifest(
        report_path,
        config_payload={"model": {"id": "gpt2"}},
        extra={"profile": "ci"},
    )
    return report_path, manifest_path


def test_runtime_manifest_fixture_matches_public_contract() -> None:
    schema = load_runtime_manifest_schema()
    manifest = json.loads(
        (
            Path.cwd()
            / "tests"
            / "fixtures"
            / "runtime_provenance"
            / "runtime.manifest.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(instance=manifest, schema=schema)
    assert manifest["manifest_version"] == 1
    assert manifest["verifier_contract_version"] == "runtime-manifest-v1"


def test_write_runtime_manifest_matches_public_contract(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(CONTAINER_EXECUTION_ENV, "1")
    monkeypatch.setenv(RUNTIME_IMAGE_ENV, "ghcr.io/invarlock/invarlock-runtime:test")
    monkeypatch.setenv(RUNTIME_IMAGE_DIGEST_ENV, _VALID_TEST_IMAGE_DIGEST)

    report_path = tmp_path / "evaluation.report.json"
    report_path.write_text('{"schema_version":"v1"}\n', encoding="utf-8")

    manifest_path = write_runtime_manifest(
        report_path,
        config_payload={"model": {"id": "gpt2"}},
        extra={"profile": "ci"},
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    jsonschema.validate(instance=manifest, schema=load_runtime_manifest_schema())
    assert manifest["context"] == {"profile": "ci"}


def test_runtime_verifier_rejects_schema_invalid_manifest(tmp_path: Path) -> None:
    report_path = tmp_path / "evaluation.report.json"
    report_path.write_text('{"schema_version":"v1"}\n', encoding="utf-8")

    manifest_path = tmp_path / "runtime.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "execution_mode": "container",
                "report": {
                    "filename": report_path.name,
                    "path": str(report_path),
                    "sha256": "0" * 64,
                },
                "runtime": {
                    "container_execution": True,
                    "image_digest": _VALID_TEST_IMAGE_DIGEST,
                    "image_ref": "ghcr.io/invarlock/invarlock-runtime:test",
                    "allow_network": False,
                    "allow_remote_code": False,
                    "allow_third_party_plugins": False,
                },
                "verifier_contract_version": "runtime-manifest-v1",
            }
        ),
        encoding="utf-8",
    )

    errors = verify_report_manifest(report_path, manifest_path)

    assert any("runtime manifest schema validation failed" in error for error in errors)


def test_runtime_verifier_reports_unreadable_report(tmp_path: Path) -> None:
    missing_report = tmp_path / "missing.report.json"
    manifest_path = tmp_path / "runtime.manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    assert verify_report_manifest(missing_report, manifest_path) == [
        f"unable to read report: [Errno 2] No such file or directory: '{missing_report}'"
    ]


def test_runtime_verifier_reports_unreadable_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(CONTAINER_EXECUTION_ENV, "1")
    monkeypatch.setenv(RUNTIME_IMAGE_ENV, "ghcr.io/invarlock/invarlock-runtime:test")
    monkeypatch.setenv(RUNTIME_IMAGE_DIGEST_ENV, _VALID_TEST_IMAGE_DIGEST)
    report_path, manifest_path = _write_valid_report_and_manifest(tmp_path)
    manifest_path.unlink()

    errors = verify_report_manifest(report_path, manifest_path)

    assert len(errors) == 1
    assert errors[0].startswith("unable to read manifest:")


def test_runtime_verifier_reports_invalid_manifest_json(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(CONTAINER_EXECUTION_ENV, "1")
    monkeypatch.setenv(RUNTIME_IMAGE_ENV, "ghcr.io/invarlock/invarlock-runtime:test")
    monkeypatch.setenv(RUNTIME_IMAGE_DIGEST_ENV, _VALID_TEST_IMAGE_DIGEST)
    report_path = tmp_path / "evaluation.report.json"
    report_path.write_text('{"schema_version":"v1"}\n', encoding="utf-8")
    manifest_path = tmp_path / "runtime.manifest.json"
    manifest_path.write_text("{not-json", encoding="utf-8")

    errors = verify_report_manifest(report_path, manifest_path)

    assert len(errors) == 1
    assert errors[0].startswith("unable to parse manifest:")


def test_runtime_verifier_rejects_non_object_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(CONTAINER_EXECUTION_ENV, "1")
    monkeypatch.setenv(RUNTIME_IMAGE_ENV, "ghcr.io/invarlock/invarlock-runtime:test")
    monkeypatch.setenv(RUNTIME_IMAGE_DIGEST_ENV, _VALID_TEST_IMAGE_DIGEST)
    report_path = tmp_path / "evaluation.report.json"
    report_path.write_text('{"schema_version":"v1"}\n', encoding="utf-8")
    manifest_path = tmp_path / "runtime.manifest.json"
    manifest_path.write_text('["not-an-object"]', encoding="utf-8")

    assert verify_report_manifest(report_path, manifest_path) == [
        "manifest payload must be a JSON object"
    ]


def test_runtime_verifier_reports_missing_schema(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(CONTAINER_EXECUTION_ENV, "1")
    monkeypatch.setenv(RUNTIME_IMAGE_ENV, "ghcr.io/invarlock/invarlock-runtime:test")
    monkeypatch.setenv(RUNTIME_IMAGE_DIGEST_ENV, _VALID_TEST_IMAGE_DIGEST)
    report_path, manifest_path = _write_valid_report_and_manifest(tmp_path)
    monkeypatch.setattr(
        "invarlock.runtime_verify.load_runtime_manifest_schema",
        lambda: {},
    )

    assert verify_report_manifest(report_path, manifest_path) == [
        "runtime manifest schema is unavailable"
    ]


def test_runtime_verifier_accepts_valid_manifest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(CONTAINER_EXECUTION_ENV, "1")
    monkeypatch.setenv(RUNTIME_IMAGE_ENV, "ghcr.io/invarlock/invarlock-runtime:test")
    monkeypatch.setenv(RUNTIME_IMAGE_DIGEST_ENV, _VALID_TEST_IMAGE_DIGEST)
    report_path, manifest_path = _write_valid_report_and_manifest(tmp_path)

    assert verify_report_manifest(report_path, manifest_path) == []


def test_runtime_verifier_reports_contract_runtime_and_digest_mismatches(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(CONTAINER_EXECUTION_ENV, "1")
    monkeypatch.setenv(RUNTIME_IMAGE_ENV, "ghcr.io/invarlock/invarlock-runtime:test")
    monkeypatch.setenv(RUNTIME_IMAGE_DIGEST_ENV, _VALID_TEST_IMAGE_DIGEST)
    report_path, manifest_path = _write_valid_report_and_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["verifier_contract_version"] = "runtime-manifest-v0"
    manifest["execution_mode"] = "local"
    manifest["runtime"]["container_execution"] = False
    manifest["runtime"]["image_digest"] = ""
    manifest["report"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "invarlock.runtime_verify.jsonschema.validate",
        lambda instance, schema: None,
    )

    errors = verify_report_manifest(report_path, manifest_path)

    assert "unexpected verifier contract version: runtime-manifest-v0" in errors
    assert 'execution_mode must be "container", got local' in errors
    assert "runtime.container_execution must be true" in errors
    assert "runtime.image_digest must be present" in errors
    assert any(error.startswith("report digest mismatch:") for error in errors)


def test_runtime_verifier_reports_missing_report_sha_and_empty_report(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(CONTAINER_EXECUTION_ENV, "1")
    monkeypatch.setenv(RUNTIME_IMAGE_ENV, "ghcr.io/invarlock/invarlock-runtime:test")
    monkeypatch.setenv(RUNTIME_IMAGE_DIGEST_ENV, _VALID_TEST_IMAGE_DIGEST)
    report_path, manifest_path = _write_valid_report_and_manifest(
        tmp_path, report_bytes=b""
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["report"].pop("sha256", None)
    manifest["runtime"] = "invalid"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "invarlock.runtime_verify.jsonschema.validate",
        lambda instance, schema: None,
    )

    errors = verify_report_manifest(report_path, manifest_path)

    assert "runtime.container_execution must be true" in errors
    assert "runtime.image_digest must be present" in errors
    assert "manifest is missing report.sha256" in errors
    assert "report file is empty" in errors
