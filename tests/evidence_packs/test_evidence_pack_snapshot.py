from __future__ import annotations

import os
from pathlib import Path

import pytest

from invarlock import evidence_pack_snapshot as snapshot_module
from invarlock.evidence_pack_snapshot import PackSnapshot


def _pack(tmp_path: Path) -> Path:
    root = tmp_path / "pack"
    root.mkdir()
    (root / "manifest.json").write_text(
        '{"format":"invarlock/evidence-pack-v1"}\n', encoding="utf-8"
    )
    nested = root / "records"
    nested.mkdir()
    (nested / "paired.json").write_text('{"records":[]}\n', encoding="utf-8")
    return root


def test_snapshot_identity_rejects_nonregular_nodes(tmp_path: Path) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    target = tmp_path / "target"
    target.write_text("bytes", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(target)

    assert snapshot_module._identity(directory) is None
    assert snapshot_module._identity(link) is None


def test_capture_entry_rejects_missing_and_replaced_source(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    destination = tmp_path / "snapshot" / "missing.json"
    expected = snapshot_module.FileIdentity(1, 2, 3, 4, 5, 6)

    entry, error = snapshot_module._capture_entry(
        missing,
        relative_path="missing.json",
        snapshot_path=destination,
        expected_identity=expected,
        max_bytes=100,
    )
    assert entry is None
    assert error == "snapshot input is missing or not a regular file: missing.json"

    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    entry, error = snapshot_module._capture_entry(
        source,
        relative_path="source.json",
        snapshot_path=tmp_path / "snapshot" / "source.json",
        expected_identity=expected,
        max_bytes=100,
    )
    assert entry is None
    assert error == "snapshot input changed before capture: source.json"


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
    assert snapshot.files.parsed_json["manifest.json"] == {
        "format": "invarlock/evidence-pack-v1"
    }
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


def test_materialization_refuses_unsafe_private_snapshot_entry(tmp_path: Path) -> None:
    root = _pack(tmp_path)
    snapshot, errors = PackSnapshot.capture(root)
    assert snapshot is not None and errors == []
    entry = snapshot.files.entry("manifest.json")
    assert entry is not None
    entry.snapshot_path.unlink()
    entry.snapshot_path.symlink_to(root / "manifest.json")

    with pytest.raises(RuntimeError, match="snapshot file became unsafe"):
        with snapshot.files.materialized():
            pass


def test_materialized_stability_accepts_mode_drift_when_bytes_still_match(
    tmp_path: Path,
) -> None:
    root = _pack(tmp_path)
    snapshot, errors = PackSnapshot.capture(root)
    assert snapshot is not None and errors == []
    storage = Path(snapshot.files.storage.name)
    (storage / "manifest.json").chmod(0o600)

    assert snapshot.files.materialized_stability_errors(storage) == []
    snapshot.files.cleanup()


def test_snapshot_cleanup_is_idempotent_after_backing_tree_removal(
    tmp_path: Path,
) -> None:
    root = _pack(tmp_path)
    snapshot, errors = PackSnapshot.capture(root)
    assert snapshot is not None and errors == []

    snapshot.files.storage.cleanup()
    snapshot.files.cleanup()


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


def test_pack_snapshot_rejects_entry_and_file_count_over_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _pack(tmp_path)
    monkeypatch.setattr(snapshot_module, "MAX_PACK_SNAPSHOT_ENTRIES", 2)

    snapshot, errors = PackSnapshot.capture(root)

    assert snapshot is None
    assert "2-entry snapshot limit" in " ".join(errors)

    monkeypatch.setattr(snapshot_module, "MAX_PACK_SNAPSHOT_ENTRIES", 20)
    monkeypatch.setattr(snapshot_module, "MAX_PACK_SNAPSHOT_FILES", 1)
    snapshot, errors = PackSnapshot.capture(root)
    assert snapshot is None
    assert "1-file snapshot limit" in " ".join(errors)


def test_pack_snapshot_rejects_file_and_total_bytes_before_copying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _pack(tmp_path)
    copied: list[Path] = []

    def record_copy(*args, **kwargs):
        copied.append(args[0])
        raise AssertionError("over-budget files must not be copied")

    monkeypatch.setattr(snapshot_module, "MAX_PACK_SNAPSHOT_FILE_BYTES", 1)
    monkeypatch.setattr(snapshot_module, "copy_regular_file_snapshot", record_copy)
    snapshot, errors = PackSnapshot.capture(root)
    assert snapshot is None
    assert "file manifest.json exceeds the 1-byte" in " ".join(errors)
    assert copied == []

    monkeypatch.setattr(snapshot_module, "MAX_PACK_SNAPSHOT_FILE_BYTES", 1024)
    monkeypatch.setattr(snapshot_module, "MAX_PACK_SNAPSHOT_TOTAL_BYTES", 1)
    snapshot, errors = PackSnapshot.capture(root)
    assert snapshot is None
    assert "1-byte total snapshot limit" in " ".join(errors)
    assert copied == []


def test_pack_snapshot_bounds_a_file_that_grows_after_enumeration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "pack"
    root.mkdir()
    manifest = root / "manifest.json"
    manifest.write_bytes(b"x")
    real_copy = snapshot_module.copy_regular_file_snapshot

    def grow_before_copy(source: Path, destination: Path, **kwargs):
        source.write_bytes(b"12345")
        return real_copy(source, destination, **kwargs)

    monkeypatch.setattr(snapshot_module, "MAX_PACK_SNAPSHOT_FILE_BYTES", 4)
    monkeypatch.setattr(snapshot_module, "copy_regular_file_snapshot", grow_before_copy)

    snapshot, errors = PackSnapshot.capture(root)

    assert snapshot is None
    assert errors == ["unable to snapshot input safely: manifest.json"]


def test_capture_and_stability_fail_if_the_snapshot_or_root_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _pack(tmp_path)
    monkeypatch.setattr(
        snapshot_module.ImmutableFileSnapshot,
        "stability_errors",
        lambda _self: ["pack snapshot changed after capture"],
    )
    snapshot, errors = PackSnapshot.capture(root)
    assert snapshot is None
    assert errors == ["pack snapshot changed after capture"]

    monkeypatch.undo()
    snapshot, errors = PackSnapshot.capture(root)
    assert snapshot is not None and errors == []
    root.rename(tmp_path / "moved-pack")
    assert "root changed after capture" in " ".join(snapshot.stability_errors())
    snapshot.files.cleanup()
