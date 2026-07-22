from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import jsonschema
import pytest

from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.public_contracts import load_runtime_manifest_schema
from invarlock.runtime_manifest import write_runtime_manifest
from invarlock.runtime_security_helpers import (
    RuntimeManifestExecution,
    RuntimeProviderManifestFiles,
)
from invarlock.runtime_verify import verify_report_manifest

_DIGEST = "sha256:" + "a" * 64


def _inputs(tmp_path: Path) -> tuple[Path, Path, RuntimeProviderManifestFiles]:
    report = tmp_path / "report.json"
    report.write_text('{"ok":true}\n', encoding="utf-8")
    config = tmp_path / "run.yaml"
    config.write_text("provider: fixture\n", encoding="utf-8")
    receipt = tmp_path / "runtime-provider.receipt.json"
    observation = tmp_path / "runtime-scoring.observation.json"
    identity = tmp_path / "model-artifact.identity.json"
    for path in (receipt, observation, identity):
        path.write_text("{}\n", encoding="utf-8")
    return report, config, RuntimeProviderManifestFiles(receipt, observation, identity)


def _execution() -> RuntimeManifestExecution:
    return RuntimeManifestExecution(
        execution_mode="container",
        container_execution=True,
        image_ref="registry.example/runtime:release",
        image_digest=_DIGEST,
        allow_network=False,
        allow_remote_code=False,
        allow_third_party_plugins=False,
    )


def test_writer_emits_the_only_canonical_runtime_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, config, sidecars = _inputs(tmp_path)
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", "sha256:" + "f" * 64)

    path = write_runtime_manifest(
        report,
        config_path=config,
        provider_files=sidecars,
        execution=_execution(),
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))

    jsonschema.validate(manifest, load_runtime_manifest_schema())
    assert manifest["manifest_version"] == 1
    assert manifest["verifier_contract_version"] == "runtime-manifest-v1"
    assert manifest["outer_container"]["image_ref"] == (
        f"registry.example/runtime:release@{_DIGEST}"
    )
    assert (
        manifest["report"]["sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "execution",
    [
        replace(_execution(), execution_mode="host"),
        replace(_execution(), container_execution=False),
        replace(_execution(), image_digest=None),
        replace(_execution(), image_digest="sha256:BAD"),
    ],
)
def test_writer_rejects_untrusted_execution(
    tmp_path: Path,
    execution: RuntimeManifestExecution,
) -> None:
    report, config, sidecars = _inputs(tmp_path)

    with pytest.raises(ValueError, match="container execution|image digest"):
        write_runtime_manifest(
            report,
            config_path=config,
            provider_files=sidecars,
            execution=execution,
        )


def test_writer_rejects_collisions_and_existing_destination(tmp_path: Path) -> None:
    report, config, sidecars = _inputs(tmp_path)
    collision = RuntimeProviderManifestFiles(
        receipt=report,
        scoring_observation=sidecars.scoring_observation,
        artifact_identity=sidecars.artifact_identity,
    )
    with pytest.raises(ValueError, match="distinct"):
        write_runtime_manifest(
            report,
            config_path=config,
            provider_files=collision,
            execution=_execution(),
        )

    write_runtime_manifest(
        report,
        config_path=config,
        provider_files=sidecars,
        execution=_execution(),
    )
    with pytest.raises(ValueError, match="must not already exist"):
        write_runtime_manifest(
            report,
            config_path=config,
            provider_files=sidecars,
            execution=_execution(),
        )


def test_writer_rejects_nonportable_image_reference(tmp_path: Path) -> None:
    report, config, sidecars = _inputs(tmp_path)

    with pytest.raises(RuntimeError, match="portable reference"):
        write_runtime_manifest(
            report,
            config_path=config,
            provider_files=sidecars,
            execution=replace(_execution(), image_ref="/private/runtime"),
        )


def test_verifier_rejects_any_noncanonical_manifest_version(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "runtime.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_version": 9,
                "verifier_contract_version": "unsupported",
            }
        ),
        encoding="utf-8",
    )

    assert verify_report_manifest(report, manifest) == [
        "unsupported runtime manifest version: manifest_version=9"
    ]


def test_verifier_rejects_image_reference_digest_disagreement(tmp_path: Path) -> None:
    report, config, sidecars = _inputs(tmp_path)
    manifest_path = write_runtime_manifest(
        report,
        config_path=config,
        provider_files=sidecars,
        execution=_execution(),
    )
    manifest = json.loads(manifest_path.read_bytes())
    manifest["outer_container"]["image_ref"] = (
        "registry.example/runtime@sha256:" + "b" * 64
    )
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    errors = verify_report_manifest(report, manifest_path)

    assert "outer_container.image_ref must bind image_digest" in errors
