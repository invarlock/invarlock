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


def test_runtime_manifest_fixture_matches_public_contract() -> None:
    schema = load_runtime_manifest_schema()
    manifest = json.loads(
        (
            Path.cwd()
            / "tests"
            / "fixtures"
            / "runtime_attestation"
            / "runtime.manifest.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(instance=manifest, schema=schema)


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
