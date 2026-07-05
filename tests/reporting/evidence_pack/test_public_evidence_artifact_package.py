from __future__ import annotations

from invarlock.public_contracts import load_public_evidence_index


def _gpt2_index_entry() -> dict[str, object]:
    index = load_public_evidence_index()
    return next(entry for entry in index["entries"] if entry["slug"] == "gpt2")


def test_gpt2_public_artifact_package_binds_checkpoint_refs_and_verifiers() -> None:
    gpt2 = _gpt2_index_entry()
    artifact_package = gpt2["artifacts"]["artifact_package"]

    assert gpt2["expected_fingerprint"].startswith("sha256:")
    assert artifact_package["kind"] == "directory"
    assert artifact_package["file_count"] == 3
    control_hashes = artifact_package["control_hashes"]
    assert control_hashes["artifact_package.json"].startswith("sha256:")
    assert control_hashes["checkpoint_refs.json"].startswith("sha256:")
    external = artifact_package["external_asset"]
    assert external["archive_path"] == (
        "public_evidence/published_basis/gpt2/artifact_package"
    )
    assert external["url"].startswith("https://github.com/invarlock/invarlock/")


def test_packaged_gpt2_public_artifact_package_is_indexed() -> None:
    gpt2 = _gpt2_index_entry()
    artifact_package = gpt2["artifacts"]["artifact_package"]

    assert artifact_package["kind"] == "directory"
    assert artifact_package["path"] == (
        "public_evidence/published_basis/gpt2/artifact_package"
    )
    assert artifact_package["file_count"] == 3
    assert artifact_package["size_bytes"] > 0
    assert artifact_package["external_asset"]["archive_path"] == (
        "public_evidence/published_basis/gpt2/artifact_package"
    )
