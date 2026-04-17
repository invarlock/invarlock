from __future__ import annotations

import json
from pathlib import Path

import pytest

import invarlock.runtime_security as runtime_security
import invarlock.runtime_security_helpers as runtime_security_helpers


def test_write_runtime_manifest_records_runtime_context(
    monkeypatch, tmp_path: Path
) -> None:
    report_path = tmp_path / "evaluation.report.json"
    report_path.write_text('{"ok": true}\n', encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("profile: release\n", encoding="utf-8")

    monkeypatch.setattr(
        runtime_security_helpers,
        "current_execution_mode",
        lambda: "container",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image",
        lambda: "ghcr.io/invarlock/invarlock-runtime:test",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image_digest",
        lambda: "sha256:container",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "running_inside_container",
        lambda: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "network_allowed",
        lambda: False,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "remote_code_allowed",
        lambda: False,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "third_party_plugins_allowed",
        lambda: False,
        raising=True,
    )

    manifest_path = runtime_security.write_runtime_manifest(
        report_path,
        config_path=config_path,
        extra={"note": "demo", "path": report_path},
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["execution_mode"] == "container"
    assert (
        payload["runtime"]["image_ref"]
        == "ghcr.io/invarlock/invarlock-runtime:test@sha256:container"
    )
    assert payload["runtime"]["image_digest"] == "sha256:container"
    assert payload["runtime"]["container_execution"] is True
    assert payload["config"]["source"] == "file"
    assert payload["context"]["note"] == "demo"
    assert payload["context"]["path"] == str(report_path)


def test_write_runtime_manifest_omits_context_for_empty_extra(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report_path = tmp_path / "evaluation.report.json"
    report_path.write_text('{"ok": true}\n', encoding="utf-8")

    monkeypatch.setattr(
        runtime_security_helpers,
        "current_execution_mode",
        lambda: "host",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image",
        lambda: runtime_security.RUNTIME_IMAGE_LOCAL_DEFAULT,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image_digest",
        lambda: None,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "running_inside_container",
        lambda: False,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "network_allowed",
        lambda: False,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "remote_code_allowed",
        lambda: False,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "third_party_plugins_allowed",
        lambda: False,
        raising=True,
    )

    manifest_path = runtime_security.write_runtime_manifest(report_path, extra={})
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "context" not in payload


def test_write_runtime_manifest_rejects_mutable_remote_ref_without_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report_path = tmp_path / "evaluation.report.json"
    report_path.write_text('{"ok": true}\n', encoding="utf-8")

    monkeypatch.setattr(
        runtime_security_helpers,
        "current_execution_mode",
        lambda: "container",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image",
        lambda: runtime_security.RUNTIME_IMAGE_DEFAULT,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "resolve_runtime_image_digest",
        lambda: None,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "running_inside_container",
        lambda: True,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "network_allowed",
        lambda: False,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "remote_code_allowed",
        lambda: False,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "third_party_plugins_allowed",
        lambda: False,
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security,
        "unverified_provenance_allowed",
        lambda: False,
        raising=True,
    )

    with pytest.raises(RuntimeError, match="digest-pinned runtime image"):
        runtime_security.write_runtime_manifest(report_path)


def test_write_runtime_manifest_honors_execution_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report_path = tmp_path / "evaluation.report.json"
    report_path.write_text('{"ok": true}\n', encoding="utf-8")

    monkeypatch.setattr(
        runtime_security_helpers,
        "current_execution_mode",
        lambda: "host-bypass",
        raising=True,
    )
    monkeypatch.setattr(
        runtime_security_helpers,
        "running_inside_container",
        lambda: False,
        raising=True,
    )

    manifest_path = runtime_security.write_runtime_manifest(
        report_path,
        execution=runtime_security.RuntimeManifestExecution(
            execution_mode="container",
            container_execution=True,
            image_ref="ghcr.io/invarlock/invarlock-runtime:test",
            image_digest="sha256:container",
            allow_network=True,
            allow_remote_code=False,
            allow_third_party_plugins=False,
        ),
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["execution_mode"] == "container"
    assert payload["runtime"]["container_execution"] is True
    assert payload["runtime"]["image_digest"] == "sha256:container"
    assert (
        payload["runtime"]["image_ref"]
        == "ghcr.io/invarlock/invarlock-runtime:test@sha256:container"
    )
