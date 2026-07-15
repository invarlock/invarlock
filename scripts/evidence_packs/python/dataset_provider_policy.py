from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from scripts.evidence_packs.python.preset_generator import (
        _resolve_dataset_provider_spec,
        _yaml_safe_dump,
    )
except ImportError:  # pragma: no cover - direct script execution
    from preset_generator import _resolve_dataset_provider_spec, _yaml_safe_dump


DATASET_PROVIDER_SNAPSHOT_SCHEMA = "invarlock.dataset-provider-input.v1"
WIKITEXT2_CONFIG_NAME = "wikitext-2-raw-v1"
WIKITEXT2_DATASET_NAME = "Salesforce/wikitext"
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _host_local_path(value: str) -> bool:
    """Return whether a provider coordinate names a host-local filesystem path.

    Provider snapshots are copied into evidence packs and then into public
    manifests.  Keep those snapshots portable: they may identify a dataset,
    but must never disclose a machine-local location in POSIX, Windows, home
    expansion, UNC, or file-URI notation.
    """

    normalized = value.strip()
    return (
        normalized.startswith(("/", "~/", "~\\", "\\\\"))
        or bool(_WINDOWS_ABSOLUTE_PATH_RE.match(normalized))
        or normalized.lower().startswith("file:")
    )


def _reject_host_local_provider_paths(value: Any) -> None:
    """Reject host-local paths anywhere in publishable provider coordinates."""

    if isinstance(value, str):
        if _host_local_path(value):
            raise ValueError(
                "dataset provider coordinates must not contain host-local paths"
            )
        return
    if isinstance(value, dict):
        for item in value.values():
            _reject_host_local_provider_paths(item)
        return
    if isinstance(value, list):
        for item in value:
            _reject_host_local_provider_paths(item)


def _validate_portable_local_jsonl_reference(field: str, value: object) -> None:
    """Require declared local JSONL references to be portable relative paths."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"local_jsonl {field} must be a non-empty string")
    normalized = value.strip()
    if _host_local_path(normalized) or "\\" in normalized:
        raise ValueError(f"local_jsonl {field} must be a portable relative path")
    path = PurePosixPath(normalized)
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"local_jsonl {field} must be a portable relative path")


def _validate_public_provider_coordinates(provider: dict[str, Any]) -> None:
    """Apply publication-safe path rules to normalized provider coordinates."""

    _reject_host_local_provider_paths(provider)
    if provider.get("kind") != "local_jsonl":
        return
    for field in ("file", "path", "data_files"):
        if field in provider:
            _validate_portable_local_jsonl_reference(field, provider[field])


def resolve_dataset_provider_spec(kind: str | None = None) -> str | dict[str, Any]:
    """Return the effective provider specification."""

    effective_kind = kind
    if effective_kind is None or not effective_kind.strip():
        effective_kind = os.environ.get("INVARLOCK_DATASET", "")
    return _resolve_dataset_provider_spec(effective_kind)


def dataset_provider_manifest_parameters(
    kind: str | None = None,
) -> dict[str, Any]:
    """Return stable provider inputs suitable for evidence-pack provenance."""

    spec = resolve_dataset_provider_spec(kind)
    if isinstance(spec, str):
        return {"kind": spec}

    effective_kind = str(spec.get("kind") or kind or "wikitext2").strip()
    payload: dict[str, Any] = {"kind": effective_kind}
    for field in ("dataset_name", "config_name", "revision"):
        value = spec.get(field)
        if isinstance(value, str) and value.strip():
            payload[field] = value.strip()

    if effective_kind == "local_jsonl":
        for field in ("file", "path", "data_files"):
            value = spec.get(field)
            if isinstance(value, str) and value.strip():
                payload[field] = value.strip()
    _validate_public_provider_coordinates(payload)
    return payload


def _json_coordinates(value: Any) -> Any:
    """Normalize provider coordinates and omit operational cache placement."""

    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("dataset provider mapping keys must be strings")
            if key == "cache_dir":
                continue
            normalized[key] = _json_coordinates(item)
        return normalized
    if isinstance(value, list):
        return [_json_coordinates(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(
        f"dataset provider coordinates contain unsupported {type(value).__name__}"
    )


def effective_dataset_provider_coordinates(
    kind: str | None = None,
) -> dict[str, Any]:
    spec = resolve_dataset_provider_spec(kind)
    if isinstance(spec, str):
        coordinates: Any = {"kind": spec.strip()}
    else:
        coordinates = _json_coordinates(spec)
    if not isinstance(coordinates, dict):  # pragma: no cover - defensive contract
        raise ValueError("effective dataset provider must be a mapping")
    effective_kind = coordinates.get("kind")
    if not isinstance(effective_kind, str) or not effective_kind.strip():
        raise ValueError("effective dataset provider must declare a non-empty kind")
    coordinates["kind"] = effective_kind.strip()
    if coordinates["kind"] == "wikitext2":
        expected = {
            "dataset_name": WIKITEXT2_DATASET_NAME,
            "config_name": WIKITEXT2_CONFIG_NAME,
        }
        for field, expected_value in expected.items():
            supplied = coordinates.get(field)
            if supplied is not None and supplied != expected_value:
                raise ValueError(
                    f"wikitext2 {field} is fixed to {expected_value!r}, "
                    f"not {supplied!r}"
                )
            coordinates[field] = expected_value
    _validate_public_provider_coordinates(coordinates)
    return coordinates


def _canonical_json(payload: Any) -> bytes:
    try:
        rendered = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"dataset provider coordinates are not canonical JSON: {exc}"
        ) from exc
    return rendered.encode("utf-8")


def build_dataset_provider_snapshot(kind: str | None = None) -> dict[str, Any]:
    provider = effective_dataset_provider_coordinates(kind)
    digest = hashlib.sha256(_canonical_json(provider)).hexdigest()
    return {
        "schema": DATASET_PROVIDER_SNAPSHOT_SCHEMA,
        "provider": provider,
        "provider_sha256": f"sha256:{digest}",
    }


def validate_dataset_provider_snapshot(payload: object) -> dict[str, Any]:
    """Validate one already-snapshotted provider payload without reopening a path."""

    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "provider",
        "provider_sha256",
    }:
        raise ValueError(
            "dataset provider snapshot must contain only schema, provider, and provider_sha256"
        )
    if payload.get("schema") != DATASET_PROVIDER_SNAPSHOT_SCHEMA:
        raise ValueError("dataset provider snapshot has an unsupported schema")
    provider = payload.get("provider")
    if not isinstance(provider, dict):
        raise ValueError("dataset provider snapshot provider must be an object")
    normalized_provider = _json_coordinates(provider)
    if normalized_provider != provider:
        raise ValueError("dataset provider snapshot is not canonical")
    _validate_public_provider_coordinates(provider)
    if provider.get("kind") == "wikitext2" and (
        provider.get("dataset_name") != WIKITEXT2_DATASET_NAME
        or provider.get("config_name") != WIKITEXT2_CONFIG_NAME
    ):
        raise ValueError(
            "wikitext2 snapshot must record its fixed dataset_name and config_name"
        )
    expected = f"sha256:{hashlib.sha256(_canonical_json(provider)).hexdigest()}"
    actual = payload.get("provider_sha256")
    if not isinstance(actual, str) or not _SHA256_RE.fullmatch(actual):
        raise ValueError("dataset provider snapshot provider_sha256 is malformed")
    if actual != expected:
        raise ValueError(
            "dataset provider snapshot provider_sha256 does not match provider"
        )
    return payload


def load_dataset_provider_snapshot(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot read dataset provider snapshot {path}: {exc}"
        ) from exc
    return validate_dataset_provider_snapshot(payload)


def dataset_provider_parameters_from_snapshot(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required dataset provider snapshot is missing: {path}")
    payload = load_dataset_provider_snapshot(path)
    return dict(payload["provider"])


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_or_validate_dataset_provider_snapshot(
    path: Path,
    *,
    resume: bool,
    kind: str | None = None,
) -> None:
    current = build_dataset_provider_snapshot(kind)
    if resume:
        if not path.is_file():
            raise ValueError(
                "resume requires the original state/dataset_provider.json snapshot"
            )
        existing = load_dataset_provider_snapshot(path)
        if existing != current:
            raise ValueError(
                "resume dataset provider differs from the persisted run input; "
                "start a fresh output directory"
            )
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(
            "dataset provider snapshot already exists; use --resume only with "
            "matching provider inputs or choose a fresh output directory"
        )
    rendered = json.dumps(current, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temp_path: Path | None = None
    try:
        descriptor, raw_temp_path = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temp_path = Path(raw_temp_path)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp_path, path)
        _fsync_directory(path.parent)
    except FileExistsError as exc:
        raise ValueError(
            "dataset provider snapshot appeared during creation; refusing to overwrite it"
        ) from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def render_raw_json_provider_yaml(kind: str | None = None) -> str:
    """Render the effective raw JSON provider override as YAML mapping contents."""

    spec = resolve_dataset_provider_spec(kind)
    if not isinstance(spec, dict):
        raise ValueError("raw JSON provider override did not resolve to a mapping")
    return _yaml_safe_dump(spec, sort_keys=False).rstrip()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evidence-pack dataset policy helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render-raw-json-yaml")
    render.add_argument("--provider", default="")

    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--provider", default="")
    snapshot.add_argument("--out", type=Path, required=True)
    snapshot.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "render-raw-json-yaml":
            print(render_raw_json_provider_yaml(str(args.provider)))
            return 0
        if args.command == "snapshot":
            write_or_validate_dataset_provider_snapshot(
                args.out,
                resume=bool(args.resume),
                kind=str(args.provider),
            )
            return 0
    except (OSError, SystemExit, ValueError) as exc:
        message = str(exc)
        if message:
            print(f"ERROR: {message}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
