from __future__ import annotations

import json
from pathlib import Path

from invarlock.public_contracts import load_public_evidence_index
from tests._repo_root import REPO_ROOT


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_gpt2_public_artifact_package_binds_checkpoint_refs_and_verifiers() -> None:
    package_dir = (
        REPO_ROOT / "public_evidence" / "published_basis" / "gpt2" / "artifact_package"
    )
    artifact = _load_json(package_dir / "artifact_package.json")
    refs = _load_json(package_dir / "checkpoint_refs.json")

    assert artifact["schema"] == "invarlock.public_evidence.artifact_package.v1"
    assert refs["schema"] == "invarlock.public_evidence.checkpoint_refs.v1"
    assert refs["weights_vendored"] is False
    assert refs["checkpoint_materialization"] == "external_reference"

    for role in ("baseline_checkpoint", "subject_checkpoint"):
        checkpoint = refs[role]
        assert isinstance(checkpoint, dict)
        assert checkpoint["materialization"] == "external_reference"
        assert checkpoint["model_id"] == "sshleifer/tiny-gpt2"

    for key in (
        "evaluation_report",
        "runtime_manifest",
        "signed_evidence_pack",
        "checkpoint_refs",
    ):
        rel_path = artifact[key]
        assert isinstance(rel_path, str)
        assert (package_dir / rel_path).exists()

    signature = _load_json(
        REPO_ROOT
        / "public_evidence"
        / "published_basis"
        / "gpt2"
        / "evidence_pack"
        / "manifest.signature.json"
    )
    assert artifact["expected_fingerprint"] == signature["signing_key_fingerprint"]

    commands = artifact["verifier_commands"]
    assert isinstance(commands, list)
    signed_pack_commands = [
        entry
        for entry in commands
        if isinstance(entry, dict) and entry.get("name") == "signed-evidence-pack"
    ]
    assert len(signed_pack_commands) == 1
    signed_command = signed_pack_commands[0]["command"]
    assert isinstance(signed_command, list)
    assert "--expected-fingerprint" in signed_command
    assert artifact["expected_fingerprint"] in signed_command
    assert signed_pack_commands[0]["expected_authenticity"] == "pinned"


def test_packaged_gpt2_public_artifact_package_is_indexed() -> None:
    repo_package = (
        REPO_ROOT / "public_evidence" / "published_basis" / "gpt2" / "artifact_package"
    )
    index = load_public_evidence_index()
    gpt2 = next(entry for entry in index["entries"] if entry["slug"] == "gpt2")
    artifact_package = gpt2["artifacts"]["artifact_package"]

    assert artifact_package["kind"] == "directory"
    assert artifact_package["path"] == (
        "public_evidence/published_basis/gpt2/artifact_package"
    )
    assert artifact_package["file_count"] == len(
        [path for path in repo_package.rglob("*") if path.is_file()]
    )
    assert artifact_package["size_bytes"] == sum(
        path.stat().st_size for path in repo_package.rglob("*") if path.is_file()
    )
