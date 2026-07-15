from __future__ import annotations

import errno
from pathlib import Path

import pytest

from invarlock.filesystem import atomic_directory
from invarlock.filesystem.atomic_directory import (
    AtomicDirectoryExistsError,
    AtomicDirectoryPublicationError,
    publish_directory_no_replace,
)


def _staging(tmp_path: Path) -> Path:
    path = tmp_path / "staging"
    path.mkdir()
    (path / "payload.txt").write_text("complete", encoding="utf-8")
    return path


def test_atomic_directory_publication_moves_complete_tree(tmp_path: Path) -> None:
    staging = _staging(tmp_path)
    destination = tmp_path / "published"

    publish_directory_no_replace(staging, destination)

    assert not staging.exists()
    assert (destination / "payload.txt").read_text(encoding="utf-8") == "complete"


def test_atomic_directory_publication_never_replaces_existing_destination(
    tmp_path: Path,
) -> None:
    staging = _staging(tmp_path)
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "owner.txt").write_text("existing", encoding="utf-8")

    with pytest.raises(AtomicDirectoryExistsError, match="refusing to replace"):
        publish_directory_no_replace(staging, destination)

    assert (staging / "payload.txt").read_text(encoding="utf-8") == "complete"
    assert (destination / "owner.txt").read_text(encoding="utf-8") == "existing"


def test_destination_created_after_parent_open_wins_without_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = _staging(tmp_path)
    destination = tmp_path / "published"
    kernel_rename = atomic_directory._rename_no_replace
    raced = False

    def race_then_rename(**kwargs: object) -> int:
        nonlocal raced
        raced = True
        destination.mkdir()
        (destination / "owner.txt").write_text("racer", encoding="utf-8")
        return kernel_rename(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(atomic_directory, "_rename_no_replace", race_then_rename)

    with pytest.raises(AtomicDirectoryExistsError, match="refusing to replace"):
        publish_directory_no_replace(staging, destination)

    assert raced is True
    assert (staging / "payload.txt").read_text(encoding="utf-8") == "complete"
    assert (destination / "owner.txt").read_text(encoding="utf-8") == "racer"


def test_staging_replaced_after_open_is_rolled_back_from_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = _staging(tmp_path)
    trusted = tmp_path / "trusted-opened-staging"
    destination = tmp_path / "published"
    kernel_rename = atomic_directory._rename_no_replace
    swapped = False

    def swap_once_then_rename(**kwargs: object) -> int:
        nonlocal swapped
        if not swapped:
            swapped = True
            staging.rename(trusted)
            staging.mkdir()
            (staging / "payload.txt").write_text("attacker", encoding="utf-8")
        return kernel_rename(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        atomic_directory,
        "_rename_no_replace",
        swap_once_then_rename,
    )

    with pytest.raises(
        AtomicDirectoryPublicationError,
        match="identity does not match staging",
    ):
        publish_directory_no_replace(staging, destination)

    assert swapped is True
    assert not destination.exists()
    assert (trusted / "payload.txt").read_text(encoding="utf-8") == "complete"
    assert (staging / "payload.txt").read_text(encoding="utf-8") == "attacker"


def test_destination_parent_swapped_after_open_rolls_publication_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_parent = tmp_path / "source-parent"
    destination_parent = tmp_path / "destination-parent"
    source_parent.mkdir()
    destination_parent.mkdir()
    staging = _staging(source_parent)
    destination = destination_parent / "published"
    opened_destination_parent = tmp_path / "opened-destination-parent"
    kernel_rename = atomic_directory._rename_no_replace
    swapped = False

    def swap_once_then_rename(**kwargs: object) -> int:
        nonlocal swapped
        if not swapped:
            swapped = True
            destination_parent.rename(opened_destination_parent)
            destination_parent.mkdir()
        return kernel_rename(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        atomic_directory,
        "_rename_no_replace",
        swap_once_then_rename,
    )

    with pytest.raises(
        AtomicDirectoryPublicationError,
        match="destination parent identity changed",
    ):
        publish_directory_no_replace(staging, destination)

    assert not destination.exists()
    assert not (opened_destination_parent / "published").exists()
    assert (staging / "payload.txt").read_text(encoding="utf-8") == "complete"


def test_source_parent_swapped_after_open_rolls_publication_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_parent = tmp_path / "source-parent"
    destination_parent = tmp_path / "destination-parent"
    source_parent.mkdir()
    destination_parent.mkdir()
    staging = _staging(source_parent)
    destination = destination_parent / "published"
    opened_source_parent = tmp_path / "opened-source-parent"
    kernel_rename = atomic_directory._rename_no_replace
    swapped = False

    def swap_once_then_rename(**kwargs: object) -> int:
        nonlocal swapped
        if not swapped:
            swapped = True
            source_parent.rename(opened_source_parent)
            source_parent.mkdir()
            attacker = source_parent / "staging"
            attacker.mkdir()
            (attacker / "payload.txt").write_text("attacker", encoding="utf-8")
        return kernel_rename(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        atomic_directory,
        "_rename_no_replace",
        swap_once_then_rename,
    )

    with pytest.raises(
        AtomicDirectoryPublicationError,
        match="staging parent identity changed",
    ):
        publish_directory_no_replace(staging, destination)

    assert not destination.exists()
    assert (opened_source_parent / "staging" / "payload.txt").read_text(
        encoding="utf-8"
    ) == "complete"
    assert (source_parent / "staging" / "payload.txt").read_text(
        encoding="utf-8"
    ) == "attacker"


@pytest.mark.parametrize("which", ["staging", "destination"])
def test_atomic_directory_publication_rejects_symlink_parent(
    tmp_path: Path, which: str
) -> None:
    source_parent = tmp_path / "source-parent"
    destination_parent = tmp_path / "destination-parent"
    source_parent.mkdir()
    destination_parent.mkdir()
    staging = source_parent / "staging"
    staging.mkdir()

    linked_parent = tmp_path / f"{which}-parent-link"
    target = source_parent if which == "staging" else destination_parent
    linked_parent.symlink_to(target, target_is_directory=True)
    if which == "staging":
        staging = linked_parent / "staging"
        destination = destination_parent / "published"
    else:
        destination = linked_parent / "published"

    with pytest.raises(
        AtomicDirectoryPublicationError,
        match=f"{which} parent must be an existing non-symlink directory",
    ):
        publish_directory_no_replace(staging, destination)

    assert (source_parent / "staging").is_dir()
    assert not (destination_parent / "published").exists()


def test_atomic_directory_publication_rejects_symlink_parent_ancestor(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real"
    nested = real_parent / "nested"
    nested.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(
        AtomicDirectoryPublicationError,
        match="destination parent must be an existing non-symlink directory",
    ):
        publish_directory_no_replace(staging, alias / "nested" / "published")

    assert staging.is_dir()
    assert not (nested / "published").exists()


def test_atomic_directory_publication_rejects_non_directory_parent(
    tmp_path: Path,
) -> None:
    staging = _staging(tmp_path)
    parent = tmp_path / "not-a-directory"
    parent.write_text("file", encoding="utf-8")

    with pytest.raises(
        AtomicDirectoryPublicationError,
        match="destination parent must be an existing non-symlink directory",
    ):
        publish_directory_no_replace(staging, parent / "published")

    assert staging.is_dir()


def test_atomic_directory_publication_rejects_symlink_staging(
    tmp_path: Path,
) -> None:
    real_staging = _staging(tmp_path)
    staging = tmp_path / "staging-link"
    staging.symlink_to(real_staging, target_is_directory=True)

    with pytest.raises(
        AtomicDirectoryPublicationError,
        match="staging must be an existing non-symlink directory",
    ):
        publish_directory_no_replace(staging, tmp_path / "published")

    assert real_staging.is_dir()
    assert staging.is_symlink()


def test_atomic_directory_publication_rejects_existing_destination_symlink(
    tmp_path: Path,
) -> None:
    staging = _staging(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    destination = tmp_path / "published"
    destination.symlink_to(existing, target_is_directory=True)

    with pytest.raises(AtomicDirectoryExistsError, match="refusing to replace"):
        publish_directory_no_replace(staging, destination)

    assert staging.is_dir()
    assert destination.is_symlink()


def test_platform_primitive_unavailable_fails_without_path_rename_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = _staging(tmp_path)
    destination = tmp_path / "published"

    class _NoRenameLibC:
        pass

    monkeypatch.setattr(atomic_directory.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        atomic_directory.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: _NoRenameLibC(),
    )

    with pytest.raises(AtomicDirectoryPublicationError, match="unavailable on Linux"):
        publish_directory_no_replace(staging, destination)

    assert staging.is_dir()
    assert not destination.exists()


def test_descriptor_relative_primitive_receives_open_parent_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = _staging(tmp_path)
    destination = tmp_path / "published"
    observed: dict[str, object] = {}

    def fail_as_existing(**kwargs: object) -> int:
        observed.update(kwargs)
        atomic_directory.ctypes.set_errno(errno.EEXIST)
        return -1

    monkeypatch.setattr(atomic_directory, "_rename_no_replace", fail_as_existing)

    with pytest.raises(AtomicDirectoryExistsError):
        publish_directory_no_replace(staging, destination)

    assert isinstance(observed["source_parent_fd"], int)
    assert observed["source_parent_fd"] >= 0  # type: ignore[operator]
    assert isinstance(observed["destination_parent_fd"], int)
    assert observed["destination_parent_fd"] >= 0  # type: ignore[operator]
    assert observed["source_name"] == b"staging"
    assert observed["destination_name"] == b"published"
