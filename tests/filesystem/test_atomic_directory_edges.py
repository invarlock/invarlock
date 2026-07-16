from __future__ import annotations

import ctypes
import errno
from pathlib import Path

import pytest

from invarlock.filesystem import atomic_directory
from invarlock.filesystem.atomic_directory import (
    AtomicDirectoryPublicationError,
    publish_directory_no_replace,
)


def test_relative_directory_publication_uses_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    staging = Path("staging")
    staging.mkdir()
    (staging / "payload").write_text("complete", encoding="utf-8")

    publish_directory_no_replace(staging, Path("published"))

    assert not staging.exists()
    assert Path("published/payload").read_text(encoding="utf-8") == "complete"


def test_publication_rejects_same_source_and_destination(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(AtomicDirectoryPublicationError, match="must name different"):
        publish_directory_no_replace(staging, staging)


def test_publication_rejects_missing_or_non_directory_staging(tmp_path: Path) -> None:
    with pytest.raises(AtomicDirectoryPublicationError, match="staging must be"):
        publish_directory_no_replace(tmp_path / "missing", tmp_path / "published")

    regular_file = tmp_path / "file"
    regular_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(AtomicDirectoryPublicationError, match="staging must be"):
        publish_directory_no_replace(regular_file, tmp_path / "published")


def test_unexpected_kernel_rename_error_is_not_misclassified_as_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()

    def denied(**_kwargs: object) -> int:
        ctypes.set_errno(errno.EACCES)
        return -1

    monkeypatch.setattr(atomic_directory, "_rename_no_replace", denied)

    with pytest.raises(AtomicDirectoryPublicationError, match="Permission denied"):
        publish_directory_no_replace(staging, tmp_path / "published")
    assert staging.is_dir()


def test_unsupported_platform_fails_without_nonexclusive_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(atomic_directory.platform, "system", lambda: "OtherOS")

    with pytest.raises(AtomicDirectoryPublicationError, match="unsupported"):
        publish_directory_no_replace(staging, tmp_path / "published")
    assert staging.is_dir()


def test_missing_secure_directory_flags_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.delattr(atomic_directory.os, "O_DIRECTORY", raising=False)

    with pytest.raises(AtomicDirectoryPublicationError, match="secure descriptor"):
        publish_directory_no_replace(staging, tmp_path / "published")
