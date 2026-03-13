from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised in shell tests/integration
    import jsonschema
except Exception:  # pragma: no cover
    jsonschema = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_schema() -> dict[str, Any]:
    path = _repo_root() / "contracts" / "proof_pack_manifest.schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _manual_validate(payload: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["manifest must decode to a JSON object"]
    required = [
        "format",
        "checksums_sha256",
        "checksums_sha256_digest",
    ]
    for field in required:
        if field not in payload:
            errors.append(f"manifest missing required field: {field}")
    if payload.get("format") != "proof-pack-v1":
        errors.append(
            f"manifest format must be 'proof-pack-v1' (found {payload.get('format')!r})"
        )
    if payload.get("checksums_sha256") != "checksums.sha256":
        errors.append("manifest checksums_sha256 must point to 'checksums.sha256'")
    digest = payload.get("checksums_sha256_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        errors.append("manifest checksums_sha256_digest must be a 64-char sha256 hex")
    network_mode = payload.get("network_mode")
    if network_mode is not None and network_mode not in {"offline", "online"}:
        errors.append("manifest network_mode must be 'offline' or 'online'")
    artifacts = payload.get("artifacts")
    if artifacts is not None and not isinstance(artifacts, list):
        errors.append("manifest artifacts must be a list")
    return errors


def validate_manifest(path: Path) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"manifest is not valid JSON: {exc}"]

    if jsonschema is None:
        return _manual_validate(payload)

    schema = _load_schema()
    try:
        jsonschema.validate(instance=payload, schema=schema)
    except Exception as exc:
        return [f"manifest schema validation failed: {exc}"]
    return _manual_validate(payload)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a proof-pack manifest.")
    parser.add_argument("manifest", help="Path to manifest.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    errors = validate_manifest(Path(args.manifest))
    if errors:
        for error in errors:
            print(error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
