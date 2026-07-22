from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock.runtime_manifest import write_runtime_manifest
from invarlock.runtime_security_helpers import RuntimeProviderManifestFiles

_DIGEST = "sha256:" + "d" * 64


def _files(root: Path) -> tuple[Path, RuntimeProviderManifestFiles]:
    report = root / "report.json"
    report.write_text('{"ok":true}\n', encoding="utf-8")
    receipt = root / "runtime-provider.receipt.json"
    observation = root / "runtime-scoring.observation.json"
    identity = root / "model-artifact.identity.json"
    for path in (receipt, observation, identity):
        path.write_text("{}\n", encoding="utf-8")
    return report, RuntimeProviderManifestFiles(receipt, observation, identity)


def _container_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INVARLOCK_CONTAINER_EXECUTION", "true")
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE", "registry/runtime:release")
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", _DIGEST)


def test_manifest_can_bind_inline_configuration_from_current_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, sidecars = _files(tmp_path)
    _container_environment(monkeypatch)
    monkeypatch.setenv("INVARLOCK_ALLOW_NETWORK", "on")
    monkeypatch.setenv("INVARLOCK_ALLOW_REMOTE_CODE", "yes")
    monkeypatch.setenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", "1")

    manifest_path = write_runtime_manifest(
        report,
        provider_files=sidecars,
        config_payload={"provider": "hf_transformers", "batch_size": 1},
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["config"]["source"] == "inline"
    assert manifest["config"]["path"] is None
    assert manifest["config"]["sha256"] is not None
    assert manifest["outer_container"] == {
        "allow_network": True,
        "allow_remote_code": True,
        "allow_third_party_plugins": True,
        "container_execution": True,
        "image_digest": _DIGEST,
        "image_ref": f"registry/runtime:release@{_DIGEST}",
    }


def test_manifest_records_missing_configuration_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, sidecars = _files(tmp_path)
    _container_environment(monkeypatch)

    manifest = json.loads(
        write_runtime_manifest(report, provider_files=sidecars).read_text(
            encoding="utf-8"
        )
    )

    assert manifest["config"] == {"path": None, "sha256": None, "source": "missing"}


def test_manifest_rejects_sidecars_or_config_outside_report_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, sidecars = _files(tmp_path)
    _container_environment(monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_receipt = outside / "receipt.json"
    outside_receipt.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="receipt must be a sibling"):
        write_runtime_manifest(
            report,
            provider_files=RuntimeProviderManifestFiles(
                outside_receipt,
                sidecars.scoring_observation,
                sidecars.artifact_identity,
            ),
        )

    outside_config = outside / "run.yaml"
    outside_config.write_text("provider: hf_transformers\n", encoding="utf-8")
    with pytest.raises(ValueError, match="config_path must be a sibling"):
        write_runtime_manifest(
            report,
            provider_files=sidecars,
            config_path=outside_config,
        )


def test_manifest_rejects_duplicate_or_reserved_binding_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, sidecars = _files(tmp_path)
    _container_environment(monkeypatch)

    with pytest.raises(ValueError, match="binding files must be distinct"):
        write_runtime_manifest(
            report,
            provider_files=RuntimeProviderManifestFiles(
                sidecars.receipt,
                sidecars.receipt,
                sidecars.artifact_identity,
            ),
        )

    with pytest.raises(ValueError, match="file config must be distinct"):
        write_runtime_manifest(
            report,
            provider_files=sidecars,
            config_path=report,
        )
