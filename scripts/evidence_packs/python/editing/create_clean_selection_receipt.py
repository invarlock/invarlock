"""Emit a strict clean-transformation selected-entry/receipt object.

This is a pure contract tool.  It consumes evaluated candidate records and does
not run models, tune runtime guards, or modify evidence-pack lifecycle state.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from pathlib import Path

try:
    from .clean_selection_contract import (
        CleanSelectionContractError,
        load_candidate_record,
        select_clean_transformation,
        verify_selected_entry,
    )
except ImportError:  # pragma: no cover - direct script-path loading
    from clean_selection_contract import (  # type: ignore[no-redef]
        CleanSelectionContractError,
        load_candidate_record,
        select_clean_transformation,
        verify_selected_entry,
    )


def _write_selected_entry(path: Path, payload: dict[str, object]) -> None:
    """Write atomically, allowing a rerun only when the prior result is exact."""

    if path.exists() or path.is_symlink():
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise CleanSelectionContractError(
                "selected-entry output is unavailable"
            ) from exc
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise CleanSelectionContractError(
                "selected-entry output must be a regular file"
            )
        existing = load_candidate_record(path)
        try:
            verified = verify_selected_entry(existing)
        except CleanSelectionContractError as exc:
            raise CleanSelectionContractError(
                "refusing to overwrite an invalid selected-entry output"
            ) from exc
        if verified != payload:
            raise CleanSelectionContractError(
                "refusing to overwrite a different selected-entry receipt"
            )
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    except OSError as exc:
        raise CleanSelectionContractError(
            f"could not atomically write selected-entry output: {exc}"
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-record", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    payload = select_clean_transformation(load_candidate_record(args.candidate_record))
    _write_selected_entry(args.out, payload)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
