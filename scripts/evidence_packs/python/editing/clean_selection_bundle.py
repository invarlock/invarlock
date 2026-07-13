# ruff: noqa: UP045  # Evidence-pack shell hosts still include Python 3.9.
"""Stage, verify, and resolve v1 clean-transformation selection bundles.

This command is intentionally the only shell-facing bridge for clean generated
transformation selections. It accepts only receipt-bound v1 bundles and
verifies the retained candidate report/replay/runtime JSON inventory before
returning parameters to a task worker.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Optional

if __package__ in {None, ""}:  # pragma: no cover - direct shell execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from invarlock.clean_selection.common import CleanSelectionEvidenceError
from invarlock.clean_selection.snapshot import (
    referenced_candidate_paths,
    selected_entry_for,
    snapshot_selection_bundle_file,
    verify_selection_bundle_file,
)
from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
    read_regular_file_bytes,
    sha256_prefixed,
)

try:
    from .clean_selection_contract import clean_edit_dir_name
except ImportError:  # pragma: no cover - direct shell execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from clean_selection_contract import clean_edit_dir_name

STAGED_SELECTION_BUNDLE = "selection_bundle.json"


def _atomic_json_write(path: Path, payload: object) -> None:
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError as exc:
        raise CleanSelectionEvidenceError(
            f"could not atomically write clean-selection output: {exc}"
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _destination_path(root: Path, relative: str) -> Path:
    current = root
    for part in relative.split("/"):
        if current.exists() or current.is_symlink():
            try:
                mode = current.lstat().st_mode
            except OSError as exc:
                raise CleanSelectionEvidenceError(
                    "clean-selection destination is unavailable"
                ) from exc
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise CleanSelectionEvidenceError(
                    "clean-selection destination traverses a non-directory"
                )
        current = current / part
    return current


def _write_snapshot_identical_or_fail(destination: Path, payload: bytes) -> None:
    """Publish only the exact bytes that the verifier already authenticated."""

    if destination.exists() or destination.is_symlink():
        try:
            existing = read_regular_file_bytes(
                destination, label="existing clean-selection destination"
            )
        except StrictJsonError as exc:
            raise CleanSelectionEvidenceError(str(exc)) from exc
        if existing != payload:
            raise CleanSelectionEvidenceError(
                "refusing to overwrite different staged clean-selection evidence"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as destination_handle:
        temporary = Path(destination_handle.name)
        destination_handle.write(payload)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())
    try:
        os.replace(temporary, destination)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def stage_selection_bundle(*, bundle_path: Path, destination: Path) -> Path:
    """Verify and stage exactly the bundle plus its referenced JSON sidecars."""

    snapshot = snapshot_selection_bundle_file(bundle_path)
    bundle = snapshot.bundle
    if destination.exists() or destination.is_symlink():
        try:
            mode = destination.lstat().st_mode
        except OSError as exc:
            raise CleanSelectionEvidenceError(
                "selection staging destination unavailable"
            ) from exc
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise CleanSelectionEvidenceError(
                "selection staging destination must be a regular directory"
            )
    else:
        destination.mkdir(parents=True, exist_ok=True)
    staged_bundle = destination / STAGED_SELECTION_BUNDLE
    _write_snapshot_identical_or_fail(staged_bundle, snapshot.bundle_bytes)
    for relative in referenced_candidate_paths(bundle):
        staged = _destination_path(destination, relative)
        _write_snapshot_identical_or_fail(staged, snapshot.sidecar_bytes[relative])
    expected_files = {STAGED_SELECTION_BUNDLE, *referenced_candidate_paths(bundle)}
    observed_files: set[str] = set()
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise CleanSelectionEvidenceError(
                "selection staging destination contains a symlink"
            )
        if path.is_file():
            observed_files.add(path.relative_to(destination).as_posix())
    if observed_files != expected_files:
        raise CleanSelectionEvidenceError(
            "selection staging destination has unbound or missing candidate evidence"
        )
    # Re-read the staged inventory, so success proves that the bytes used by
    # worker tasks—not just the host source—pass the same evidence checks.
    verify_selection_bundle_file(staged_bundle, evidence_root=destination)
    return staged_bundle


def _resolved_payload(
    *, bundle_path: Path, model_key: str, edit_type: str, requested_scope: str
) -> dict[str, object]:
    snapshot = snapshot_selection_bundle_file(bundle_path)
    bundle = snapshot.bundle
    selected = selected_entry_for(
        bundle,
        model_key=model_key,
        edit_type=edit_type,
        requested_scope=requested_scope,
    )
    entry = selected["selected_entry"]
    assert isinstance(entry, Mapping)
    parameters = entry["parameters"]
    assert isinstance(parameters, Mapping)
    scope = entry["scope"]
    assert isinstance(scope, str)
    if edit_type == "quant_rtn":
        param1, param2 = str(parameters["bits"]), str(parameters["group_size"])
    elif edit_type == "synthetic_lowrank_delta":
        param1, param2 = str(parameters["rank"]), str(parameters["scale"])
    elif edit_type == "synthetic_dense_update":
        param1, param2 = str(parameters["step_size"]), str(parameters["iterations"])
    else:  # selected_entry_for already excludes this, preserve a hard boundary.
        raise CleanSelectionEvidenceError("selected entry has an unsupported edit type")
    edit_dir = clean_edit_dir_name(selected)
    receipt = entry["selection_receipt"]
    assert isinstance(receipt, Mapping)
    return {
        "status": "selected",
        "reason": "receipt_bound_clean_selection",
        "edit_type": edit_type,
        "requested_edit_type": edit_type,
        "param1": param1,
        "param2": param2,
        "scope": scope,
        "version": "clean",
        "edit_dir_name": edit_dir,
        "selected_candidate_id": receipt["selected_candidate_id"],
        "candidate_set_sha256": receipt["candidate_set_sha256"],
        "selection_bundle_sha256": sha256_prefixed(snapshot.bundle_bytes),
    }


def resolve_clean_selection(
    *, bundle_path: Path, model_key: str, edit_type: str, requested_scope: str = ""
) -> dict[str, object]:
    """Return shell-safe resolved params from a fully verified v1 selection."""

    return _resolved_payload(
        bundle_path=bundle_path,
        model_key=model_key,
        edit_type=edit_type,
        requested_scope=requested_scope,
    )


def _rewrite_spec(edit_type: str, payload: Mapping[str, object]) -> str:
    scope = payload["scope"]
    param1 = payload["param1"]
    param2 = payload["param2"]
    if not all(isinstance(value, str) and value for value in (scope, param1, param2)):
        raise CleanSelectionEvidenceError("resolved v1 clean selection is incomplete")
    return f"{edit_type}:{param1}:{param2}:{scope}"


def rewrite_clean_batch_specs(
    *, bundle_path: Path, model_key: str, edit_specs: object
) -> list[object]:
    """Replace only ``type:clean`` specs with explicit v1-selected literals."""

    if not isinstance(edit_specs, list):
        raise CleanSelectionEvidenceError("batch edit specs must be a JSON list")
    rewritten: list[object] = []
    for raw in edit_specs:
        if not isinstance(raw, Mapping):
            raise CleanSelectionEvidenceError("batch edit spec must be an object")
        item = dict(raw)
        spec = item.get("spec")
        version = item.get("version", "clean")
        if not isinstance(spec, str) or not isinstance(version, str):
            raise CleanSelectionEvidenceError(
                "batch edit spec and version must be strings"
            )
        parts = spec.split(":")
        edit_type = parts[0] if parts else ""
        is_clean = version == "clean" and len(parts) >= 2 and parts[1] == "clean"
        if not is_clean:
            rewritten.append(item)
            continue
        if edit_type == "magnitude_prune":
            raise CleanSelectionEvidenceError(
                "clean magnitude-prune is handled only by the pruning-specific selection bridge"
            )
        requested_scope = parts[2] if len(parts) > 2 else ""
        resolved = _resolved_payload(
            bundle_path=bundle_path,
            model_key=model_key,
            edit_type=edit_type,
            requested_scope=requested_scope,
        )
        item["spec"] = _rewrite_spec(edit_type, resolved)
        item["version"] = "clean"
        item["selection_edit_dir_name"] = resolved["edit_dir_name"]
        rewritten.append(item)
    return rewritten


def _parse_json_argument(value: str, *, label: str) -> object:
    try:
        return parse_json_bytes(value.encode("utf-8"), label=label)
    except StrictJsonError as exc:
        raise CleanSelectionEvidenceError(f"{label} is not valid JSON") from exc


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--bundle", required=True, type=Path)
    verify.add_argument("--evidence-root", type=Path)
    stage = commands.add_parser("stage")
    stage.add_argument("--bundle", required=True, type=Path)
    stage.add_argument("--dest", required=True, type=Path)
    resolve = commands.add_parser("resolve")
    resolve.add_argument("--bundle", required=True, type=Path)
    resolve.add_argument("--model-key", required=True)
    resolve.add_argument("--edit-type", required=True)
    resolve.add_argument("--requested-scope", default="")
    rewrite = commands.add_parser("rewrite-batch")
    rewrite.add_argument("--bundle", required=True, type=Path)
    rewrite.add_argument("--model-key", required=True)
    rewrite.add_argument("--edit-specs-json", required=True)
    args = parser.parse_args(argv)
    if args.command == "verify":
        verify_selection_bundle_file(args.bundle, evidence_root=args.evidence_root)
        return 0
    if args.command == "stage":
        staged = stage_selection_bundle(bundle_path=args.bundle, destination=args.dest)
        print(staged)
        return 0
    if args.command == "resolve":
        print(
            json.dumps(
                resolve_clean_selection(
                    bundle_path=args.bundle,
                    model_key=args.model_key,
                    edit_type=args.edit_type,
                    requested_scope=args.requested_scope,
                ),
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "rewrite-batch":
        print(
            json.dumps(
                rewrite_clean_batch_specs(
                    bundle_path=args.bundle,
                    model_key=args.model_key,
                    edit_specs=_parse_json_argument(
                        args.edit_specs_json, label="--edit-specs-json"
                    ),
                ),
                allow_nan=False,
                separators=(",", ":"),
            )
        )
        return 0
    raise AssertionError(f"unexpected command {args.command}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "STAGED_SELECTION_BUNDLE",
    "resolve_clean_selection",
    "rewrite_clean_batch_specs",
    "stage_selection_bundle",
]
