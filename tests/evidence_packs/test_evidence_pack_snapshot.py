from __future__ import annotations

import os
from pathlib import Path

import pytest

from invarlock.evidence_pack_snapshot import PackSnapshot


def _pack(tmp_path: Path) -> Path:
    root = tmp_path / "pack"
    root.mkdir()
    (root / "manifest.json").write_text(
        '{"format":"evidence-pack-v1"}\n', encoding="utf-8"
    )
    nested = root / "records"
    nested.mkdir()
    (nested / "paired.json").write_text('{"records":[]}\n', encoding="utf-8")
    return root


def test_pack_snapshot_materializes_authenticated_immutable_bytes(
    tmp_path: Path,
) -> None:
    root = _pack(tmp_path)

    snapshot, errors = PackSnapshot.capture(root)

    assert errors == []
    assert snapshot is not None
    assert snapshot.files.inventory == frozenset(
        {"manifest.json", "records/paired.json"}
    )
    assert snapshot.files.parsed_json["manifest.json"] == {"format": "evidence-pack-v1"}
    manifest = snapshot.files.entry("manifest.json")
    assert manifest is not None
    assert manifest.read_bytes() == (root / "manifest.json").read_bytes()
    assert snapshot.files.entry("missing") is None

    with snapshot.files.materialized() as materialized:
        assert (materialized / "manifest.json").read_bytes() == manifest.read_bytes()
        assert snapshot.files.materialized_stability_errors(materialized) == []
    assert not Path(snapshot.files.storage.name).exists()


def test_pack_snapshot_rejects_unsafe_root_and_entries(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    assert PackSnapshot.capture(missing)[0] is None

    real = _pack(tmp_path)
    alias = tmp_path / "pack-link"
    alias.symlink_to(real, target_is_directory=True)
    snapshot, errors = PackSnapshot.capture(alias)
    assert snapshot is None
    assert "not found or unsafe" in " ".join(errors)

    target = tmp_path / "outside.json"
    target.write_text("{}", encoding="utf-8")
    (real / "linked.json").symlink_to(target)
    snapshot, errors = PackSnapshot.capture(real)
    assert snapshot is None
    assert "must not contain symlinks" in " ".join(errors)


def test_pack_snapshot_rejects_non_regular_entry(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("named pipes are unavailable")
    root = _pack(tmp_path)
    os.mkfifo(root / "unsafe.fifo")

    snapshot, errors = PackSnapshot.capture(root)

    assert snapshot is None
    assert "only regular files and directories" in " ".join(errors)


def test_snapshot_detects_source_mutation_and_inventory_drift(tmp_path: Path) -> None:
    root = _pack(tmp_path)
    snapshot, errors = PackSnapshot.capture(root)
    assert snapshot is not None and errors == []

    (root / "manifest.json").write_text('{"format":"tampered"}\n', encoding="utf-8")
    (root / "extra.json").write_text("{}\n", encoding="utf-8")

    joined = "\n".join(snapshot.stability_errors())
    assert "changed after capture" in joined
    assert "extra=['extra.json']" in joined
    snapshot.files.cleanup()


def test_materialized_snapshot_detects_added_removed_symlinked_and_changed_files(
    tmp_path: Path,
) -> None:
    root = _pack(tmp_path)
    snapshot, errors = PackSnapshot.capture(root)
    assert snapshot is not None and errors == []
    storage = Path(snapshot.files.storage.name)

    (storage / "extra.txt").write_text("extra", encoding="utf-8")
    assert "inventory changed" in " ".join(
        snapshot.files.materialized_stability_errors(storage)
    )
    (storage / "extra.txt").unlink()

    manifest = storage / "manifest.json"
    manifest.chmod(0o600)
    manifest.write_text('{"format":"tampered"}\n', encoding="utf-8")
    assert "bytes changed" in " ".join(
        snapshot.files.materialized_stability_errors(storage)
    )

    manifest.unlink()
    manifest.symlink_to(root / "manifest.json")
    assert "became unsafe" in " ".join(
        snapshot.files.materialized_stability_errors(storage)
    )
    snapshot.files.cleanup()


def test_materialization_refuses_tampered_private_snapshot(tmp_path: Path) -> None:
    root = _pack(tmp_path)
    snapshot, errors = PackSnapshot.capture(root)
    assert snapshot is not None and errors == []
    entry = snapshot.files.entry("manifest.json")
    assert entry is not None
    entry.snapshot_path.chmod(0o600)
    entry.snapshot_path.write_text('{"format":"tampered"}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="snapshot digest changed"):
        with snapshot.files.materialized():
            pass


def test_structural_json_validation_can_be_disabled(tmp_path: Path) -> None:
    root = _pack(tmp_path)
    (root / "manifest.json").write_text("[]\n", encoding="utf-8")

    parsed_snapshot, errors = PackSnapshot.capture(root)
    assert parsed_snapshot is not None and errors == []
    assert "manifest.json" not in parsed_snapshot.files.parsed_json
    parsed_snapshot.files.cleanup()

    opaque_snapshot, errors = PackSnapshot.capture(root, validate_structural_json=False)
    assert opaque_snapshot is not None and errors == []
    assert opaque_snapshot.files.entry("manifest.json").json_error == "not requested"  # type: ignore[union-attr]
    opaque_snapshot.files.cleanup()
