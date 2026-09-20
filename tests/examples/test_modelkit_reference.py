"""Verify the retained rejected ModelKit comparison without model execution."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "examples/integrations/modelkit-handoff/references/mistral-7b"
spec = importlib.util.spec_from_file_location(
    "modelkit_reference", REFERENCE / "replay.py"
)
assert spec and spec.loader
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def test_signed_rejection_replays_without_becoming_acceptance(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["replay.py"])
    assert reference.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] and result["integrity_ok"] and result["receipt_ok"]
    assert result["verification_status"] == 7
    assert result["policy_verdict"] == "fail"
    assert result["historical_recipient_accepted"] is False
    assert result["envelope_evidence_bound"] and result["subject_digest_bound"]
    assert result["envelope_errors"] == [
        "technical verdict does not satisfy recipient policy"
    ]


def test_archive_inventory_is_bounded_and_contains_only_public_reference_bytes():
    archive = REFERENCE / "reference.zip"
    provenance = json.loads((REFERENCE / "provenance.json").read_bytes())
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == reference.ARCHIVE_SHA256
    assert (REFERENCE / "reference.sha256").read_text().split()[
        0
    ] == reference.ARCHIVE_SHA256
    assert provenance["archive"]["sha256"] == reference.ARCHIVE_SHA256
    assert (
        archive.stat().st_size
        == provenance["archive"]["bytes"]
        < reference.ARCHIVE_LIMIT
    )
    with zipfile.ZipFile(archive) as stream:
        manifest = json.loads(stream.read("manifest.json"))
        names = stream.namelist()
        assert names == sorted(set(names))
        assert set(names) == set(manifest["files"]) | {"manifest.json"}
        assert len(names) == provenance["archive"]["file_count"]
        assert (
            sum(x.file_size for x in stream.infolist())
            == provenance["archive"]["expanded_bytes"]
            < 4 * 1024**2
        )
        for info in stream.infolist():
            path = Path(info.filename)
            assert stat.S_ISREG(info.external_attr >> 16)
            assert not path.is_absolute() and ".." not in path.parts
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert path.suffix not in {".gguf", ".safetensors", ".bin", ".tar", ".key"}
            assert "blobs" not in path.parts
            payload = stream.read(info)
            assert b"PRIVATE KEY" not in payload
            if info.filename != "manifest.json":
                assert manifest["files"][info.filename] == {
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
        report = json.loads(stream.read("evidence/reports/evaluation.report.json"))
        assert report["record_count"] == 128
        assert report["baseline"]["mean_score"] == 94 / 128
        assert report["subject"]["mean_score"] == 93 / 128
        assert report["verdict"] == "fail"
        assert report["sample_qualification"]["passed"] is True
        assert report["uncertainty"]["lower"] < -2 < report["uncertainty"]["upper"]


def test_changed_archive_is_rejected_before_extraction(tmp_path, monkeypatch):
    archive = tmp_path / "changed.zip"
    archive.write_bytes((REFERENCE / "reference.zip").read_bytes() + b"changed")

    def no_open(*args, **kwargs):
        pytest.fail("archive was opened before authenticating its fixed digest")

    monkeypatch.setattr(zipfile, "ZipFile", no_open)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        reference.replay(archive)


def test_authenticated_snapshot_survives_path_replacement(tmp_path, monkeypatch):
    archive = tmp_path / "reference.zip"
    archive.write_bytes((REFERENCE / "reference.zip").read_bytes())
    original = reference.read_regular_file_bytes

    def replace_after_read(path, **kwargs):
        snapshot = original(path, **kwargs)
        path.write_bytes(b"changed after authentication")
        return snapshot

    monkeypatch.setattr(reference, "read_regular_file_bytes", replace_after_read)
    assert reference.replay(archive)["ok"]


@pytest.mark.parametrize("kind", ["oversized", "symlink"])
def test_unsafe_archive_source_is_rejected(tmp_path, kind):
    archive = tmp_path / "reference.zip"
    if kind == "oversized":
        archive.write_bytes(b"x" * (reference.ARCHIVE_LIMIT + 1))
        message = "size limit"
    else:
        archive.symlink_to(REFERENCE / "reference.zip")
        message = "symlink"
    with pytest.raises(ValueError, match=message):
        reference.replay(archive)


@pytest.mark.parametrize(
    "mutation",
    ["receipt-signature", "recipient-trust", "expected-acceptance", "subject-digest"],
)
def test_independent_replay_rejects_changed_binding(tmp_path, mutation):
    with zipfile.ZipFile(REFERENCE / "reference.zip") as stream:
        stream.extractall(tmp_path)
    if mutation == "receipt-signature":
        path = tmp_path / "verification.receipt.json"
        value = json.loads(path.read_bytes())
        value["signature"]["value"] = "A" * 86 + "=="
    elif mutation == "recipient-trust":
        path = tmp_path / "recipient-policy.json"
        value = json.loads(path.read_bytes())
        value["trusted_signers"][0]["status"] = "revoked"
    else:
        path = tmp_path / "reference.json"
        value = json.loads(path.read_bytes())
        if mutation == "expected-acceptance":
            value["expected"]["recipient_accepted"] = True
        else:
            value["model_content_digests"]["subject"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(value))
    assert reference.verify_snapshot(tmp_path)["ok"] is False


def test_cli_failure_status_when_replay_checks_fail(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["replay.py"])
    monkeypatch.setattr(reference, "replay", lambda path: {"ok": False})
    assert reference.main() == 1
    assert json.loads(capsys.readouterr().out) == {"ok": False}


def test_retained_delivery_metadata_binds_packages_and_preserves_rejections():
    """Check retained observations and descriptors; large layer bytes are absent."""
    provenance = json.loads((REFERENCE / "provenance.json").read_bytes())
    with zipfile.ZipFile(REFERENCE / "reference.zip") as stream:
        result = json.loads(stream.read("delivery/result.json"))
        inventory = json.loads(stream.read("delivery/package-metadata/inventory.json"))
        assert len(inventory) == 8
        original = result["accepted"]
        repacked = result["scenarios"]["repacked_same_contents"]
        for decision in (original, repacked):
            assert decision["technical_integrity_ok"]
            assert decision["envelope_evidence_bound"]
            assert decision["technical_policy_verdict"] == "fail"
            assert decision["accepted"] is False
            assert decision["exit_code"] == 1
        for role in ("baseline", "subject"):
            assert result["packages"][role] == provenance["delivery"]["packages"][role]
            assert (
                result["packages"][role]["original"]
                != result["packages"][role]["repacked"]
            )
            for kind, decision in (("original", original), ("repacked", repacked)):
                prefix = f"{role}-{kind}"
                manifest_bytes = stream.read(
                    f"delivery/package-metadata/{prefix}.manifest.json"
                )
                config_bytes = stream.read(
                    f"delivery/package-metadata/{prefix}.config.json"
                )
                for suffix, payload in (
                    ("manifest", manifest_bytes),
                    ("config", config_bytes),
                ):
                    assert inventory[f"{prefix}.{suffix}.json"] == {
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                manifest = json.loads(manifest_bytes)
                package = decision["packages"][role]
                assert (
                    "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
                    == result["packages"][role][kind]
                )
                assert (
                    manifest["config"]["digest"]
                    == "sha256:" + hashlib.sha256(config_bytes).hexdigest()
                )
                assert manifest["config"]["size"] == len(config_bytes)
                assert manifest["layers"][0]["digest"] == package["layer_digest"]
                assert (
                    package["artifact_content_digest"]
                    == "sha256:" + provenance["models"][role]["sha256"]
                )
                assert package["model_bytes"] == provenance["models"][role]["bytes"]
            assert (
                original["packages"][role]["artifact_content_digest"]
                == repacked["packages"][role]["artifact_content_digest"]
            )
        expected_errors = {
            "wrong_runtime": "embedded subject runtime identity does not match caller runtime anchor",
            "revoked_signer": "envelope signer is revoked by recipient policy",
            "altered_evidence": "manifest.json is not canonical JSON",
            "candidate_changed": "actual candidate content differs from expected content",
        }
        for name, message in expected_errors.items():
            assert message in result["scenarios"][name]["errors"]
        assert (
            "No such file or directory"
            in result["scenarios"]["missing_package"]["errors"][0]
        )
        assert {
            name: value["exit_code"] for name, value in result["scenarios"].items()
        } == provenance["delivery"]["scenario_exit_codes"]


def test_retained_engine_smokes_use_same_subject_and_keep_exact_output():
    provenance = json.loads((REFERENCE / "provenance.json").read_bytes())
    with zipfile.ZipFile(REFERENCE / "reference.zip") as stream:
        result = json.loads(stream.read("delivery/podman-result.json"))
        for engine in ("docker", "podman"):
            observed = result[engine]
            assert observed["container_engine"] == engine
            assert observed["recipient_accepted"] is False
            assert (
                observed["artifact_digest"]
                == "sha256:" + provenance["models"]["subject"]["sha256"]
            )
            assert observed["generated_text"] == " a city of many faces. It is\n\n"
            assert (
                hashlib.sha256(observed["generated_text"].encode()).hexdigest()
                == provenance["delivery"]["raw_output_sha256"]
            )
            before = json.loads(
                stream.read(
                    f"delivery/before-{'inference' if engine == 'docker' else 'podman'}.json"
                )
            )
            assert (
                before["technical_integrity_ok"] and before["envelope_evidence_bound"]
            )
            assert before["accepted"] is False
        transport = json.loads(stream.read("delivery/image-transport.json"))
        assert result["docker"]["image_id"] == transport["source_execution_id"]
        assert result["podman"]["image_id"] == transport["podman_execution_id"]
        assert transport["source_execution_id"] != transport["podman_execution_id"]
        assert transport["recorded_checks"][
            "docker_podman_configurations_and_layers_identical"
        ]
        assert len(transport["layers"]) == len(transport["rootfs_diff_ids"]) == 12


def test_render_provenance_matches_retained_html():
    provenance = json.loads((REFERENCE / "provenance.json").read_bytes())
    assert (
        hashlib.sha256((REFERENCE / "report.html").read_bytes()).hexdigest()
        == provenance["render"]["html_sha256"]
    )
