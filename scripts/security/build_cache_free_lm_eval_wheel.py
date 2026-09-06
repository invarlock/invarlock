#!/usr/bin/env python3
"""Derive the cache-free, exact-match LM Evaluation Harness image profile."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import re
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

UPSTREAM_VERSION = "0.4.12"
DERIVED_VERSION = "0.4.12+invarlock.exactmatch.1"
UPSTREAM_WHEEL_SHA256 = (
    "02971ff68284dd14cfa7fce9310a58452c4162e8d413ba96aa7988a0ff9352ef"
)
UPSTREAM_DIST_INFO = f"lm_eval-{UPSTREAM_VERSION}.dist-info"
DERIVED_DIST_INFO = f"lm_eval-{DERIVED_VERSION}.dist-info"
DERIVED_WHEEL_NAME = f"lm_eval-{DERIVED_VERSION}-py3-none-any.whl"
REMOVED_REQUIREMENTS = frozenset(("lm-eval", "sqlitedict", "rouge-score", "nltk"))


class DerivationError(ValueError):
    """The authenticated upstream wheel or compiled lock was unexpected."""


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _record_digest(payload: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode()
    return f"sha256={encoded.rstrip('=')}"


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(name)
        and "\\" not in name
        and not path.is_absolute()
        and ".." not in path.parts
    )


def validate_wheel_record(archive: zipfile.ZipFile) -> None:
    """Validate that every file in a wheel is covered by its RECORD."""

    members = archive.infolist()
    names = [member.filename for member in members if not member.is_dir()]
    if len(names) != len(set(names)) or not all(_safe_member(name) for name in names):
        raise DerivationError("wheel members are duplicated or unsafe")
    records = [name for name in names if name.endswith(".dist-info/RECORD")]
    if len(records) != 1:
        raise DerivationError("wheel must contain exactly one RECORD")
    record_name = records[0]
    try:
        rows = list(csv.reader(io.StringIO(archive.read(record_name).decode("utf-8"))))
    except (KeyError, UnicodeDecodeError, csv.Error) as exc:
        raise DerivationError("wheel RECORD is unreadable") from exc
    if any(len(row) != 3 for row in rows):
        raise DerivationError("wheel RECORD has an invalid row")
    entries = {row[0]: (row[1], row[2]) for row in rows}
    if len(entries) != len(rows) or set(entries) != set(names):
        raise DerivationError("wheel RECORD does not cover the wheel contents")
    for name in names:
        digest, size = entries[name]
        if name == record_name:
            if digest or size:
                raise DerivationError("wheel RECORD must not hash itself")
            continue
        payload = archive.read(name)
        if digest != _record_digest(payload) or size != str(len(payload)):
            raise DerivationError("wheel RECORD does not match the wheel contents")


def patch_metadata(
    payload: bytes,
    upstream_version: str,
    derived_version: str,
    removed: tuple[bytes, ...],
) -> bytes:
    """Replace the identity and remove only exact authenticated dependency lines."""

    version = f"Version: {upstream_version}\n".encode()
    if payload.count(version) != 1:
        raise DerivationError("upstream wheel metadata version changed")
    for dependency in removed:
        if payload.count(dependency) != 1:
            raise DerivationError("upstream wheel metadata dependency changed")
    patched = payload.replace(version, f"Version: {derived_version}\n".encode(), 1)
    for dependency in removed:
        patched = patched.replace(dependency, b"", 1)
    return patched


def _patch_metadata(payload: bytes) -> bytes:
    return patch_metadata(
        payload,
        UPSTREAM_VERSION,
        DERIVED_VERSION,
        (b"Requires-Dist: sqlitedict\n", b"Requires-Dist: rouge-score>=0.0.4\n"),
    )


def _patch_model(payload: bytes) -> bytes:
    type_import = (
        b"if TYPE_CHECKING:\n"
        b"    from sqlitedict import SqliteDict\n\n"
        b"    from lm_eval.api.instance import Instance\n"
    )
    replacement_import = (
        b"if TYPE_CHECKING:\n    from lm_eval.api.instance import Instance\n"
    )
    annotation = b"self.dbdict: SqliteDict | None"
    class_start = (
        b"class CachingLM:\n    def __init__(self, lm: LM, cache_db: str) -> None:\n"
    )
    class_end = b"        lm.set_cache_hook(self.get_cache_hook())\n"
    if payload.count(type_import) != 1 or payload.count(annotation) != 1:
        raise DerivationError("upstream cache type surface changed")
    start = payload.find(class_start)
    end = payload.find(class_end, start)
    if start < 0 or end < 0 or payload.find(class_start, start + 1) >= 0:
        raise DerivationError("upstream cache implementation changed")
    end += len(class_end)
    replacement_class = class_start + (
        b'        """Reject response caching in the cache-free integration image."""\n'
        b"        raise RuntimeError(\n"
        b'            "response caching is unavailable in the cache-free InvarLock integration"\n'
        b"        )\n"
    )
    patched = payload.replace(type_import, replacement_import, 1)
    patched = patched.replace(annotation, b"self.dbdict: Any | None", 1)
    adjusted_start = patched.find(class_start)
    adjusted_end = patched.find(class_end, adjusted_start) + len(class_end)
    patched = patched[:adjusted_start] + replacement_class + patched[adjusted_end:]
    if b"sqlitedict" in patched.lower():
        raise DerivationError("cache dependency remained in the derived model module")
    return patched


def _record(files: dict[str, bytes], record_name: str) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for name, payload in sorted(files.items()):
        writer.writerow((name, _record_digest(payload), len(payload)))
    writer.writerow((record_name, "", ""))
    return output.getvalue().encode("utf-8")


def build_derived_wheel(
    source: Path,
    output_directory: Path,
    *,
    upstream_sha256: str,
    upstream_dist_info: str,
    derived_dist_info: str,
    derived_wheel_name: str,
    patches: dict[str, Callable[[bytes], bytes]],
) -> Path:
    """Build one deterministic local-version wheel from the pinned upstream wheel."""

    try:
        source_payload = source.read_bytes()
    except OSError as exc:
        raise DerivationError("upstream wheel is unreadable") from exc
    if _digest(source_payload) != upstream_sha256:
        raise DerivationError("upstream evaluator wheel SHA-256 changed")
    try:
        with zipfile.ZipFile(io.BytesIO(source_payload)) as archive:
            validate_wheel_record(archive)
            source_names = [
                member.filename for member in archive.infolist() if not member.is_dir()
            ]
            files = {name: archive.read(name) for name in source_names}
    except zipfile.BadZipFile as exc:
        raise DerivationError("upstream wheel is not a readable ZIP archive") from exc

    record_name = f"{upstream_dist_info}/RECORD"
    if record_name not in files or not set(patches).issubset(files):
        raise DerivationError(
            "upstream wheel is missing the authenticated patch surface"
        )
    del files[record_name]
    for name, patch in patches.items():
        files[name] = patch(files[name])
    renamed = {
        (
            f"{derived_dist_info}/{name.removeprefix(f'{upstream_dist_info}/')}"
            if name.startswith(f"{upstream_dist_info}/")
            else name
        ): payload
        for name, payload in files.items()
    }
    derived_record = f"{derived_dist_info}/RECORD"
    renamed[derived_record] = _record(renamed, derived_record)

    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / derived_wheel_name
    if destination.exists() or destination.is_symlink():
        raise DerivationError("derived wheel output already exists")
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(renamed, key=lambda item: item == derived_record):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, renamed[name])
    with zipfile.ZipFile(destination) as archive:
        validate_wheel_record(archive)
    return destination


def build_wheel(source: Path, output_directory: Path) -> Path:
    """Preserve the selected scorer and HF execution code from the pinned wheel."""

    return build_derived_wheel(
        source,
        output_directory,
        upstream_sha256=UPSTREAM_WHEEL_SHA256,
        upstream_dist_info=UPSTREAM_DIST_INFO,
        derived_dist_info=DERIVED_DIST_INFO,
        derived_wheel_name=DERIVED_WHEEL_NAME,
        patches={
            f"{UPSTREAM_DIST_INFO}/METADATA": _patch_metadata,
            "lm_eval/api/model.py": _patch_model,
        },
    )


def _requirement_name(line: str) -> str | None:
    if not line or line[0].isspace() or line.startswith(("#", "-")):
        return None
    match = re.match(r"([A-Za-z0-9_.-]+)==", line)
    return match.group(1).lower().replace("_", "-") if match else None


def filter_lock(
    source: Path,
    destination: Path,
    *,
    removed_requirements: frozenset[str] = REMOVED_REQUIREMENTS,
) -> None:
    """Remove the replaced upstream wheel and unused dependency blocks."""

    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if _requirement_name(line) is not None and current:
            blocks.append(current)
            current = []
        current.append(line)
    if current:
        blocks.append(current)
    removed: dict[str, int] = dict.fromkeys(sorted(removed_requirements), 0)
    retained: list[str] = []
    for block in blocks:
        name = next(
            (_requirement_name(line) for line in block if _requirement_name(line)),
            None,
        )
        if name in removed_requirements:
            removed[name] += 1
        else:
            retained.extend(block)
    for name, count in removed.items():
        if count != 1:
            suffix = " exactly once" if count else ""
            raise DerivationError(f"compiled lock must contain {name}{suffix}")
    payload = "".join(retained)
    if "sqlitedict" in payload.lower() or any(
        _requirement_name(line) in removed_requirements for line in payload.splitlines()
    ):
        raise DerivationError("removed requirement remained in the dependency lock")
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise DerivationError("filtered lock output must be a regular file")
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise DerivationError("filtered lock temporary output already exists")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    wheel = subparsers.add_parser("build-wheel")
    wheel.add_argument("--input", type=Path, required=True)
    wheel.add_argument("--output-directory", type=Path, required=True)
    lock = subparsers.add_parser("filter-lock")
    lock.add_argument("--input", type=Path, required=True)
    lock.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build-wheel":
            print(build_wheel(args.input, args.output_directory))
        else:
            filter_lock(args.input, args.output)
    except (DerivationError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
