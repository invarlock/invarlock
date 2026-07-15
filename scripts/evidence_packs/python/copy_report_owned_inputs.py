#!/usr/bin/env python3
"""Copy only scenario-owned report inputs into a generic evidence pack."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import stat
from pathlib import Path, PurePosixPath

_RUNTIME_PRESET = re.compile(r"calibrated_preset_[A-Za-z0-9][A-Za-z0-9._-]*\.yaml")
_ERROR_RUN_FILE = re.compile(
    r"(?:[0-9]{6}/)?(?:report\.json|runtime\.manifest\.json|events\.jsonl)"
)


class OwnedInputError(ValueError):
    """A report-adjacent directory contains an unowned entry."""


def _absolute_lexical_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path, *, label: str) -> Path:
    """Return a lexical absolute path after rejecting every existing link."""

    absolute = _absolute_lexical_path(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            current_stat = current.lstat()
        except FileNotFoundError:
            break
        except OSError as exc:
            raise OwnedInputError(f"{label} is unavailable") from exc
        if stat.S_ISLNK(current_stat.st_mode):
            raise OwnedInputError(f"{label} must not contain a symlink component")
    return absolute


def _safe_report_relative_path(value: str) -> tuple[str, ...]:
    if not value or "\\" in value:
        raise OwnedInputError("report-relative path is unsafe")
    parts = tuple(value.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise OwnedInputError("report-relative path is unsafe")
    return parts


def _prepare_destination(destination: Path) -> Path:
    destination = _reject_symlink_components(
        destination, label="owned-input destination"
    )
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OwnedInputError("owned-input destination cannot be created") from exc
    destination = _reject_symlink_components(
        destination, label="owned-input destination"
    )
    try:
        destination_stat = destination.lstat()
    except OSError as exc:
        raise OwnedInputError("owned-input destination is unavailable") from exc
    if not stat.S_ISDIR(destination_stat.st_mode):
        raise OwnedInputError("owned-input destination must be a regular directory")
    return destination


def _require_regular_file(path: Path, *, label: str) -> None:
    _reject_symlink_components(path, label=label)
    try:
        path_stat = path.lstat()
    except OSError as exc:
        raise OwnedInputError(f"{label} is unavailable") from exc
    if not stat.S_ISREG(path_stat.st_mode):
        raise OwnedInputError(f"{label} must be a regular file")


def _allowed_runtime_input(relative: str) -> bool:
    return relative in {"baseline_report.json", "runtime_input.json"} or bool(
        _RUNTIME_PRESET.fullmatch(relative)
    )


def _allowed_error_input(relative: str) -> bool:
    return bool(_ERROR_RUN_FILE.fullmatch(relative))


def copy_owned_inputs(
    source: Path,
    destination: Path,
    *,
    kind: str,
    report_relative_path: str,
) -> None:
    report_parts = _safe_report_relative_path(report_relative_path)
    source = _reject_symlink_components(source, label=f"{kind} input")
    try:
        source_stat = source.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise OwnedInputError(f"{kind} input is unavailable") from exc
    if not stat.S_ISDIR(source_stat.st_mode):
        raise OwnedInputError(f"{kind} input must be a regular directory")

    error_owned = len(report_parts) >= 3 and report_parts[1] == "errors"
    if kind in {"source", "edited"} and not error_owned:
        raise OwnedInputError(f"{kind} input is only owned by error scenarios")

    destination = _prepare_destination(destination)
    files: list[tuple[Path, str]] = []
    for path in source.rglob("*"):
        mode = path.lstat().st_mode
        relative = path.relative_to(source).as_posix()
        if stat.S_ISLNK(mode):
            raise OwnedInputError(f"{kind} input contains a symlink: {relative}")
        if stat.S_ISDIR(mode):
            allowed_directory = (
                False
                if kind == "runtime_inputs"
                else bool(re.fullmatch(r"[0-9]{6}", relative))
            )
            if not allowed_directory:
                raise OwnedInputError(
                    f"{kind} input contains an unowned directory: {relative}"
                )
            continue
        if not stat.S_ISREG(mode):
            raise OwnedInputError(
                f"{kind} input contains a non-regular file: {relative}"
            )
        allowed = (
            _allowed_runtime_input(relative)
            if kind == "runtime_inputs"
            else _allowed_error_input(relative)
        )
        if not allowed:
            raise OwnedInputError(f"{kind} input is not scenario-owned: {relative}")
        files.append((path, relative))

    for source_path, relative in sorted(files, key=lambda item: item[1]):
        destination_path = destination.joinpath(*PurePosixPath(relative).parts)
        _require_regular_file(source_path, label=f"{kind} input {relative}")
        _reject_symlink_components(
            destination_path.parent, label="owned-input destination"
        )
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        _reject_symlink_components(
            destination_path.parent, label="owned-input destination"
        )
        if destination_path.is_symlink() or (
            destination_path.exists() and not destination_path.is_file()
        ):
            raise OwnedInputError(
                f"owned-input destination must be a regular file path: {relative}"
            )
        shutil.copyfile(source_path, destination_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("runtime_inputs", "source", "edited"))
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("report_relative_path")
    args = parser.parse_args()
    try:
        copy_owned_inputs(
            args.source,
            args.destination,
            kind=args.kind,
            report_relative_path=args.report_relative_path,
        )
    except (OSError, OwnedInputError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
