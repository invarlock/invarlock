from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _normalize_pack_path(pack_dir: Path, rel_path: str) -> Path | None:
    path = (pack_dir / rel_path).resolve()
    try:
        path.relative_to(pack_dir.resolve())
    except ValueError:
        return None
    return path


def _validate_reference(
    *,
    pack_dir: Path,
    label: str,
    payload: Any,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return errors

    rel_path = payload.get("path")
    digest = payload.get("digest")
    if rel_path is None and digest is None:
        return errors
    if not isinstance(rel_path, str) or not rel_path:
        return [
            f"{label} must include a non-empty path when digest verification is enabled"
        ]
    if (
        not isinstance(digest, str)
        or not digest.startswith("sha256:")
        or len(digest) != 71
    ):
        return [f"{label} digest must be a sha256:... string"]

    resolved = _normalize_pack_path(pack_dir, rel_path)
    if resolved is None:
        return [f"{label} path escapes the pack root: {rel_path}"]
    if not resolved.is_file():
        return [f"{label} path is missing: {rel_path}"]

    actual = _sha256(resolved)
    if actual != digest:
        return [
            f"{label} digest mismatch for {rel_path} (expected {digest}, got {actual})"
        ]
    return errors


def verify_manifest_attestation(pack_dir: Path) -> list[str]:
    manifest_path = pack_dir / "manifest.json"
    payload = _load_json(manifest_path)
    if not isinstance(payload, dict):
        return ["manifest must decode to a JSON object"]

    errors: list[str] = []
    errors.extend(
        _validate_reference(
            pack_dir=pack_dir, label="subject", payload=payload.get("subject")
        )
    )

    invocation = payload.get("invocation")
    if isinstance(invocation, dict):
        errors.extend(
            _validate_reference(
                pack_dir=pack_dir,
                label="invocation.config_source",
                payload=invocation.get("config_source"),
            )
        )

    errors.extend(
        _validate_reference(
            pack_dir=pack_dir, label="environment", payload=payload.get("environment")
        )
    )

    materials = payload.get("materials")
    if isinstance(materials, list):
        for index, material in enumerate(materials):
            errors.extend(
                _validate_reference(
                    pack_dir=pack_dir,
                    label=f"materials[{index}]",
                    payload=material,
                )
            )
    return errors


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify digest-backed proof-pack attestation fields."
    )
    parser.add_argument("pack_dir")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    errors = verify_manifest_attestation(Path(args.pack_dir))
    if errors:
        for error in errors:
            print(error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
