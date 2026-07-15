"""Immutable filesystem snapshots for evidence-pack verification."""

from __future__ import annotations

import hashlib
import shutil
import stat
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from invarlock.evidence_pack_contracts.probes import (
    PROBE_FILENAMES,
    ProbeValidationError,
    validate_probe_payload,
)
from invarlock.evidence_pack_json import StrictJsonError, load_json_object
from invarlock.runtime_security import RUNTIME_MANIFEST_FILENAME


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    mode: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class SnapshotEntry:
    relative_path: str
    source_path: Path
    snapshot_path: Path
    sha256: str
    identity: FileIdentity
    snapshot_identity: FileIdentity
    parsed_json: Any | None
    json_error: str | None

    def read_bytes(self) -> bytes:
        """Read the authenticated snapshot bytes, not the mutable source path."""

        return self.snapshot_path.read_bytes()


def _identity(path: Path) -> FileIdentity | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        return None
    return FileIdentity(
        device=info.st_dev,
        inode=info.st_ino,
        mode=info.st_mode,
        size=info.st_size,
        modified_ns=info.st_mtime_ns,
        changed_ns=info.st_ctime_ns,
    )


def _parse_json_once(path: Path, *, label: str) -> tuple[Any | None, str | None]:
    try:
        return load_json_object(path, label=label), None
    except StrictJsonError as exc:
        return None, str(exc)


def _sha256_path(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _capture_entry(
    source_path: Path,
    *,
    relative_path: str,
    snapshot_path: Path,
    parse_json: bool = False,
) -> tuple[SnapshotEntry | None, str | None]:
    before = _identity(source_path)
    if before is None:
        return None, f"snapshot input is missing or not a regular file: {source_path}"
    try:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.touch(exist_ok=False)
        shutil.copyfile(source_path, snapshot_path)
        digest = _sha256_path(snapshot_path)
    except OSError as exc:
        return None, f"unable to snapshot input {source_path}: {exc}"
    after = _identity(source_path)
    if after != before:
        return None, f"snapshot input changed while being captured: {source_path}"
    parsed_json, json_error = (
        _parse_json_once(snapshot_path, label=relative_path)
        if parse_json
        else (None, "not requested")
    )
    snapshot_path.chmod(0o444)
    snapshot_identity = _identity(snapshot_path)
    assert snapshot_identity is not None
    return (
        SnapshotEntry(
            relative_path=relative_path,
            source_path=source_path,
            snapshot_path=snapshot_path,
            sha256=digest,
            identity=before,
            snapshot_identity=snapshot_identity,
            parsed_json=parsed_json,
            json_error=json_error,
        ),
        None,
    )


def _manifest_reference_paths(payload: dict[str, Any]) -> set[str]:
    """Return manifest-declared JSON inputs that influence verification."""

    paths: set[str] = set()

    def add_reference(value: Any) -> None:
        if not isinstance(value, dict):
            return
        relative_path = value.get("path")
        if (
            isinstance(relative_path, str)
            and relative_path
            and relative_path.lower().endswith(".json")
        ):
            paths.add(relative_path)

    add_reference(payload.get("subject"))
    add_reference(payload.get("environment"))
    add_reference(payload.get("verification_policy_pack"))
    invocation = payload.get("invocation")
    if isinstance(invocation, dict):
        add_reference(invocation.get("config_source"))
    for field in ("materials", "verification_baselines"):
        values = payload.get(field)
        if isinstance(values, list):
            for value in values:
                add_reference(value)
    return paths


def _structural_json_paths(entries: list[SnapshotEntry]) -> set[str]:
    """Identify pack files whose JSON semantics drive package verification."""

    paths = {
        "metadata/manifest.json",
        "manifest.signature.json",
        "metadata/manifest.signature.json",
        "metadata/scenarios.json",
    }
    for entry in entries:
        relative = entry.relative_path
        filename = PurePosixPath(relative).name
        if relative.startswith("results/") and filename == "final_verdict.json":
            paths.add(relative)
        if relative.startswith("reports/") and filename in {
            "evaluation.report.json",
            RUNTIME_MANIFEST_FILENAME,
            *PROBE_FILENAMES,
        }:
            paths.add(relative)
        if relative.startswith("baselines/") and filename in {
            "evaluation.report.json",
            RUNTIME_MANIFEST_FILENAME,
        }:
            paths.add(relative)
    return paths


def _validate_pack_json_inputs(
    entries: list[SnapshotEntry],
) -> tuple[list[SnapshotEntry], dict[str, Any], list[str]]:
    """Strictly parse all JSON that can affect package verification decisions."""

    by_relative = {entry.relative_path: entry for entry in entries}
    manifest = by_relative.get("manifest.json")
    required = _structural_json_paths(entries)
    if manifest is not None and isinstance(manifest.parsed_json, dict):
        required.update(_manifest_reference_paths(manifest.parsed_json))

    parsed: dict[str, Any] = {
        entry.relative_path: entry.parsed_json
        for entry in entries
        if entry.json_error is None
    }
    replacements: dict[str, SnapshotEntry] = {}
    errors: list[str] = []
    for relative in sorted(required):
        # ``manifest.json`` is parsed during capture so its existing format
        # diagnostics are returned by the verifier's format phase.
        if relative == "manifest.json":
            continue
        entry = by_relative.get(relative)
        if entry is None:
            continue
        payload, error = _parse_json_once(
            entry.snapshot_path,
            label=f"evidence-pack JSON input {relative}",
        )
        if error is not None:
            errors.append(f"unsafe evidence-pack JSON input {relative}: {error}")
            continue
        parsed[relative] = payload
        if PurePosixPath(relative).name in PROBE_FILENAMES:
            try:
                validate_probe_payload(PurePosixPath(relative).name, payload)
            except ProbeValidationError as exc:
                errors.append(f"unsafe evidence-pack JSON input {relative}: {exc}")
                continue
        replacements[relative] = replace(
            entry,
            parsed_json=payload,
            json_error=None,
        )
    if replacements:
        entries = [replacements.get(entry.relative_path, entry) for entry in entries]
    return entries, parsed, errors


@dataclass(frozen=True)
class ImmutableFileSnapshot:
    entries: tuple[SnapshotEntry, ...]
    tracked_sources: Mapping[Path, FileIdentity | None]
    parsed_json: Mapping[str, Any]
    digest_ledger: Mapping[str, str]
    inventory: frozenset[str]
    label: str
    storage: tempfile.TemporaryDirectory[str]

    def entry(self, relative_path: str) -> SnapshotEntry | None:
        return next(
            (entry for entry in self.entries if entry.relative_path == relative_path),
            None,
        )

    def cleanup(self) -> None:
        """Delete the private backing tree, including read-only snapshot files."""

        root = Path(self.storage.name)
        if root.exists():
            root.chmod(0o755)
            for path in root.rglob("*"):
                try:
                    path.chmod(0o755 if path.is_dir() else 0o600)
                except OSError:
                    pass
        self.storage.cleanup()

    def stability_errors(self) -> list[str]:
        errors: list[str] = []
        for source, expected in self.tracked_sources.items():
            current = _identity(source)
            unsafe_new_entry = expected is None and (
                source.is_symlink() or source.exists()
            )
            if current != expected or unsafe_new_entry:
                errors.append(f"{self.label} changed after capture: {source}")
        return errors

    def materialized_stability_errors(self, root: Path) -> list[str]:
        expected = {entry.relative_path: entry for entry in self.entries}
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        errors: list[str] = []
        if actual != set(expected):
            errors.append(f"{self.label} materialized inventory changed")
        for relative, entry in expected.items():
            path = root.joinpath(*PurePosixPath(relative).parts)
            if path.is_symlink() or not path.is_file():
                errors.append(
                    f"{self.label} materialized file became unsafe: {relative}"
                )
                continue
            if _identity(path) == entry.snapshot_identity:
                continue
            try:
                digest = _sha256_path(path)
            except OSError:
                errors.append(f"{self.label} materialized file became unreadable")
                continue
            if digest != entry.sha256:
                errors.append(f"{self.label} materialized bytes changed: {relative}")
        return errors

    @contextmanager
    def materialized(self) -> Iterator[Path]:
        root = Path(self.storage.name)
        try:
            for relative_path, expected_digest in self.digest_ledger.items():
                entry = self.entry(relative_path)
                assert entry is not None
                snapshot_path = entry.snapshot_path
                if _identity(snapshot_path) == entry.snapshot_identity:
                    continue
                if snapshot_path.is_symlink() or not snapshot_path.is_file():
                    raise RuntimeError(
                        f"immutable snapshot file became unsafe: {relative_path}"
                    )
                if _sha256_path(snapshot_path) != expected_digest:
                    raise RuntimeError(
                        f"immutable snapshot digest changed: {relative_path}"
                    )
            for directory in sorted(
                (path for path in root.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                directory.chmod(0o555)
            root.chmod(0o555)
            yield root
        finally:
            if root.exists():
                root.chmod(0o755)
                for path in root.rglob("*"):
                    try:
                        path.chmod(0o755 if path.is_dir() else 0o600)
                    except OSError:
                        pass
                self.cleanup()


@dataclass(frozen=True)
class PackSnapshot:
    source_root: Path
    files: ImmutableFileSnapshot

    @classmethod
    def capture(
        cls,
        pack_dir: Path,
        *,
        validate_structural_json: bool = True,
    ) -> tuple[PackSnapshot | None, list[str]]:
        if pack_dir.is_symlink() or not pack_dir.is_dir():
            return None, [f"Pack directory not found or unsafe: {pack_dir}"]
        paths: list[Path] = []
        errors: list[str] = []
        for path in pack_dir.rglob("*"):
            try:
                mode = path.lstat().st_mode
            except OSError as exc:
                errors.append(
                    "unable to inspect evidence-pack entry: "
                    f"{path.relative_to(pack_dir)} ({exc})"
                )
                continue
            if stat.S_ISLNK(mode):
                errors.append(
                    "evidence packs must not contain symlinks: "
                    f"{path.relative_to(pack_dir)}"
                )
                continue
            if stat.S_ISREG(mode):
                paths.append(path)
                continue
            if not stat.S_ISDIR(mode):
                errors.append(
                    "evidence packs must contain only regular files and directories: "
                    f"{path.relative_to(pack_dir)}"
                )
        if errors:
            return None, errors
        entries: list[SnapshotEntry] = []
        tracked: dict[Path, FileIdentity | None] = {}
        storage = tempfile.TemporaryDirectory(prefix="invarlock-pack-snapshot-")
        snapshot_root = Path(storage.name)
        for path in sorted(
            paths, key=lambda item: item.relative_to(pack_dir).as_posix()
        ):
            relative = path.relative_to(pack_dir).as_posix()
            tracked[path] = _identity(path)
            entry, error = _capture_entry(
                path,
                relative_path=relative,
                snapshot_path=snapshot_root.joinpath(*PurePosixPath(relative).parts),
                parse_json=validate_structural_json and relative == "manifest.json",
            )
            if error is not None:
                errors.append(error)
            else:
                assert entry is not None
                entries.append(entry)
        if errors:
            storage.cleanup()
            return None, errors
        if validate_structural_json:
            entries, parsed, json_errors = _validate_pack_json_inputs(entries)
            if json_errors:
                storage.cleanup()
                return None, json_errors
        else:
            parsed = {}
        snapshot = cls(
            source_root=pack_dir,
            files=ImmutableFileSnapshot(
                entries=tuple(entries),
                tracked_sources=MappingProxyType(tracked),
                parsed_json=MappingProxyType(parsed),
                digest_ledger=MappingProxyType(
                    {entry.relative_path: entry.sha256 for entry in entries}
                ),
                inventory=frozenset(entry.relative_path for entry in entries),
                label="pack snapshot",
                storage=storage,
            ),
        )
        if stability_errors := snapshot.stability_errors():
            snapshot.files.cleanup()
            return None, stability_errors
        return snapshot, []

    def stability_errors(self) -> list[str]:
        errors = self.files.stability_errors()
        if self.source_root.is_symlink() or not self.source_root.is_dir():
            return [*errors, "pack snapshot root changed after capture"]
        current_inventory = {
            path.relative_to(self.source_root).as_posix()
            for path in self.source_root.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        if current_inventory != self.files.inventory:
            missing = sorted(self.files.inventory - current_inventory)
            extra = sorted(current_inventory - self.files.inventory)
            errors.append(
                "pack snapshot changed after capture; "
                f"missing={missing!r}; extra={extra!r}"
            )
        return errors


__all__ = ["PackSnapshot"]
