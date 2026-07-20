from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import invarlock.runtime_verify as runtime_verify

_DIGEST = "sha256:" + "a" * 64


def test_verifier_fails_closed_when_report_or_manifest_cannot_be_loaded(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    manifest = tmp_path / "runtime.manifest.json"

    missing_report = runtime_verify.verify_runtime_manifest(
        report,
        manifest,
        expected_image_digest="not-a-digest",
    )
    assert missing_report.ok is False
    assert missing_report.trust_status == "failed"
    assert any("unable to read report" in error for error in missing_report.errors)
    assert any(
        "expected runtime image digest" in error for error in missing_report.errors
    )

    report.write_text("{}\n", encoding="utf-8")
    assert runtime_verify.verify_report_manifest(report, manifest)[0].startswith(
        "unable to read manifest"
    )

    manifest.write_text("not-json\n", encoding="utf-8")
    assert runtime_verify.verify_report_manifest(report, manifest)[0].startswith(
        "unable to parse manifest"
    )

    manifest.write_text("[]\n", encoding="utf-8")
    assert runtime_verify.verify_report_manifest(report, manifest) == [
        "manifest payload must be a JSON object"
    ]


def test_expected_image_digest_validation_distinguishes_invalid_and_mismatch() -> None:
    assert runtime_verify._normalize_expected_image_digest(None) is None
    assert runtime_verify._normalize_expected_image_digest(_DIGEST.upper()) == _DIGEST
    assert (
        runtime_verify._expected_image_digest_errors(
            declared_image_digest=_DIGEST,
            expected_image_digest=None,
        )
        == []
    )
    assert runtime_verify._expected_image_digest_errors(
        declared_image_digest=_DIGEST,
        expected_image_digest="sha256:BAD",
    ) == ["expected runtime image digest must match sha256:<64 lowercase hex chars>"]
    assert runtime_verify._expected_image_digest_errors(
        declared_image_digest=None,
        expected_image_digest=_DIGEST,
    ) == [f"runtime image digest mismatch: manifest=<missing> expected={_DIGEST}"]


def test_declared_image_digest_uses_only_validated_manifest_object() -> None:
    assert runtime_verify._declared_image_digest({}) is None
    assert runtime_verify._declared_image_digest({"outer_container": []}) is None
    assert (
        runtime_verify._declared_image_digest(
            {"outer_container": {"image_digest": _DIGEST.upper()}}
        )
        == _DIGEST
    )


def test_loaded_manifest_reports_every_runtime_binding_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "report.json"
    manifest_path = tmp_path / "runtime.manifest.json"
    monkeypatch.setattr(runtime_verify, "load_runtime_manifest_schema", lambda: {})
    assert runtime_verify._verify_loaded_report_manifest(
        b"{}",
        {"manifest_version": runtime_verify.RUNTIME_MANIFEST_VERSION},
        report_path=report,
        manifest_path=manifest_path,
    ) == ["runtime manifest schema is unavailable"]

    monkeypatch.setattr(
        runtime_verify,
        "load_runtime_manifest_schema",
        lambda: {"type": "object", "required": ["required"]},
    )
    assert (
        "schema validation failed"
        in runtime_verify._verify_loaded_report_manifest(
            b"{}",
            {"manifest_version": runtime_verify.RUNTIME_MANIFEST_VERSION},
            report_path=report,
            manifest_path=manifest_path,
        )[0]
    )

    monkeypatch.setattr(
        runtime_verify, "load_runtime_manifest_schema", lambda: {"type": "object"}
    )
    manifest: dict[str, object] = {
        "manifest_version": runtime_verify.RUNTIME_MANIFEST_VERSION,
        "verifier_contract_version": "old",
        "execution_mode": "host",
        "outer_container": {
            "container_execution": False,
            "image_digest": "",
            "image_ref": "different",
            "allow_network": True,
            "allow_remote_code": True,
            "allow_third_party_plugins": True,
        },
        "report": {},
        "runtime_provider": {},
        "config": {"source": "inline"},
    }
    errors = runtime_verify._verify_loaded_report_manifest(
        b"",
        manifest,
        report_path=report,
        manifest_path=manifest_path,
        require_strict_runtime=True,
        expected_image_digest=_DIGEST,
    )
    for fragment in (
        "unexpected verifier contract version",
        "execution_mode",
        "container_execution",
        "image_digest must be present",
        "allow_network",
        "allow_remote_code",
        "allow_third_party_plugins",
        "missing report.sha256",
        "report file is empty",
        "verifiable file config",
        "report.path",
        "report.filename",
        "runtime_provider.receipt reference is missing",
        "runtime image digest mismatch",
    ):
        assert any(fragment in error for error in errors), (fragment, errors)


def test_loaded_manifest_checks_report_digest_and_image_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runtime_verify, "load_runtime_manifest_schema", lambda: {"type": "object"}
    )
    report = tmp_path / "report.json"
    manifest_path = tmp_path / "runtime.manifest.json"
    manifest: dict[str, object] = {
        "manifest_version": runtime_verify.RUNTIME_MANIFEST_VERSION,
        "verifier_contract_version": runtime_verify.RUNTIME_VERIFIER_CONTRACT_VERSION,
        "execution_mode": "container",
        "outer_container": {
            "container_execution": True,
            "image_digest": _DIGEST,
            "image_ref": "repository@sha256:" + "b" * 64,
        },
        "report": {
            "sha256": "0" * 64,
            "path": "other.json",
            "filename": "other.json",
        },
        "runtime_provider": {},
        "config": {"source": "inline"},
    }
    errors = runtime_verify._verify_loaded_report_manifest(
        b"{}",
        manifest,
        report_path=report,
        manifest_path=manifest_path,
    )
    assert "outer_container.image_ref must bind image_digest" in errors
    assert any("report digest mismatch" in error for error in errors)


def test_reference_collision_checks_reject_all_reserved_aliases(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    manifest_path = tmp_path / "runtime.manifest.json"
    assert runtime_verify._reference_collision_errors(
        {}, report_path=report, manifest_path=manifest_path
    ) == ["runtime_provider bindings are missing"]

    manifest: dict[str, object] = {
        "report": {"path": "report.json", "filename": "report.json"},
        "runtime_provider": {
            "receipt": {"filename": "report.json"},
            "scoring_observation": {"filename": "shared.json"},
            "artifact_identity": {"filename": "shared.json"},
        },
        "config": {
            "source": "file",
            "path": "shared.json",
        },
    }

    errors = runtime_verify._reference_collision_errors(
        manifest,
        report_path=report,
        manifest_path=manifest_path,
    )

    assert "runtime provider binding filenames must be distinct" in errors
    assert any("collide with the report" in error for error in errors)
    assert "file config collides with a runtime provider binding" in errors

    cast_config = manifest["config"]
    assert isinstance(cast_config, dict)
    cast_config["path"] = "runtime.manifest.json"
    assert "file config collides with the report or manifest" in (
        runtime_verify._reference_collision_errors(
            manifest,
            report_path=report,
            manifest_path=manifest_path,
        )
    )


def test_bound_json_reader_rejects_missing_tampered_and_malformed_sidecars(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "runtime.manifest.json"
    schema: dict[str, object] = {
        "type": "object",
        "required": ["ok"],
        "properties": {"ok": {"const": True}},
        "additionalProperties": False,
    }

    assert runtime_verify._read_bound_json_object(
        None, manifest_path=manifest, label="sidecar", schema=schema
    )[1] == ["sidecar reference is missing"]
    assert runtime_verify._read_bound_json_object(
        {"filename": 7}, manifest_path=manifest, label="sidecar", schema=schema
    )[1] == ["sidecar reference is invalid"]
    assert runtime_verify._read_bound_json_object(
        {"filename": "missing.json", "sha256": "0" * 64},
        manifest_path=manifest,
        label="sidecar",
        schema=schema,
    )[1][0].startswith("unable to read sidecar")

    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text('{"ok":true}\n', encoding="utf-8")
    assert (
        "digest mismatch"
        in runtime_verify._read_bound_json_object(
            {"filename": sidecar.name, "sha256": "0" * 64},
            manifest_path=manifest,
            label="sidecar",
            schema=schema,
        )[1][0]
    )

    sidecar.write_text("not-json\n", encoding="utf-8")
    digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    assert runtime_verify._read_bound_json_object(
        {"filename": sidecar.name, "sha256": digest},
        manifest_path=manifest,
        label="sidecar",
        schema=schema,
    )[1][0].startswith("unable to parse sidecar")

    sidecar.write_text("[]\n", encoding="utf-8")
    digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    assert runtime_verify._read_bound_json_object(
        {"filename": sidecar.name, "sha256": digest},
        manifest_path=manifest,
        label="sidecar",
        schema=schema,
    )[1] == ["sidecar must decode to a JSON object"]

    sidecar.write_text('{"ok":false}\n', encoding="utf-8")
    digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    assert (
        "schema validation failed"
        in runtime_verify._read_bound_json_object(
            {"filename": sidecar.name, "sha256": digest},
            manifest_path=manifest,
            label="sidecar",
            schema=schema,
        )[1][0]
    )

    sidecar.write_text('{"ok":true}\n', encoding="utf-8")
    digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    bound, errors = runtime_verify._read_bound_json_object(
        {"filename": sidecar.name, "sha256": digest},
        manifest_path=manifest,
        label="sidecar",
        schema=schema,
    )
    assert errors == []
    assert bound is not None
    assert bound.value == {"ok": True}
    assert bound.sha256 == digest


def test_file_config_binding_authenticates_exact_sibling_bytes(tmp_path: Path) -> None:
    manifest_path = tmp_path / "runtime.manifest.json"
    config = tmp_path / "run.yaml"
    config.write_bytes(b"provider: fixture\n")
    digest = hashlib.sha256(config.read_bytes()).hexdigest()

    assert (
        runtime_verify._verify_file_config_binding(
            {"config": {"source": "inline"}}, manifest_path=manifest_path
        )
        == []
    )
    assert (
        runtime_verify._verify_file_config_binding(
            {"config": {"source": "file", "path": config.name, "sha256": digest}},
            manifest_path=manifest_path,
        )
        == []
    )

    config.write_bytes(b"provider: tampered\n")
    assert (
        "digest mismatch"
        in runtime_verify._verify_file_config_binding(
            {"config": {"source": "file", "path": config.name, "sha256": digest}},
            manifest_path=manifest_path,
        )[0]
    )

    assert runtime_verify._verify_sibling_digest_reference(
        None, manifest_path=manifest_path, label="config"
    ) == ["config reference is missing"]
    assert runtime_verify._verify_sibling_digest_reference(
        {"filename": 7}, manifest_path=manifest_path, label="config"
    ) == ["config reference is invalid"]
    assert runtime_verify._verify_sibling_digest_reference(
        {"filename": "missing", "sha256": "0" * 64},
        manifest_path=manifest_path,
        label="config",
    )[0].startswith("unable to read config")


def test_provider_cross_bindings_report_all_disagreements() -> None:
    artifact = {"artifact_format": "hf_snapshot"}
    evidence = runtime_verify._RuntimeProviderEvidence(
        receipt=runtime_verify._BoundJsonObject(
            b"{}",
            {
                "artifact_identity": {"artifact_format": "gguf"},
                "scoring_observation_sha256": "wrong",
                "plugin": {"name": "receipt"},
                "capabilities": {
                    "provider_name": "capability",
                    "artifact_formats": ["gguf"],
                },
                "outer_image_digest": "wrong",
            },
            "receipt-digest",
        ),
        scoring_observation=runtime_verify._BoundJsonObject(
            b"{}",
            {
                "provider_name": "observation",
                "artifact_identity_sha256": "wrong",
            },
            "observation-digest",
        ),
        artifact_identity=runtime_verify._BoundJsonObject(
            b"{}", artifact, "artifact-digest"
        ),
    )
    errors = runtime_verify._runtime_provider_cross_binding_errors(
        evidence,
        manifest={"outer_container": {"image_digest": _DIGEST}},
    )
    for fragment in (
        "artifact_identity does not match",
        "scoring_observation_sha256",
        "provider names do not agree",
        "artifact_identity_sha256",
        "outer_image_digest",
        "artifact format",
    ):
        assert any(fragment in error for error in errors), (fragment, errors)


@pytest.mark.parametrize(
    ("expected", "binding_errors", "trust_status", "matched"),
    [
        (_DIGEST, [], "expected_image_digest_matched", True),
        (None, [], "manifest_bound", False),
        ("sha256:" + "b" * 64, [], "failed", False),
        (_DIGEST, ["binding failed"], "failed", False),
    ],
)
def test_snapshot_trust_status_requires_both_binding_and_expected_digest(
    monkeypatch: pytest.MonkeyPatch,
    expected: str | None,
    binding_errors: list[str],
    trust_status: str,
    matched: bool,
) -> None:
    monkeypatch.setattr(
        runtime_verify,
        "_verify_loaded_report_manifest",
        lambda *_args, **_kwargs: binding_errors,
    )
    manifest: dict[str, object] = {"outer_container": {"image_digest": _DIGEST}}

    result = runtime_verify.verify_runtime_manifest_snapshot(
        json.dumps({"ok": True}).encode(),
        manifest,
        report="report.json",
        manifest="runtime.manifest.json",
        expected_image_digest=expected,
    )

    assert result.binding_verified is (not binding_errors)
    assert result.expected_digest_matched is matched
    assert result.trust_status == trust_status
    assert result.ok is (not result.errors)
