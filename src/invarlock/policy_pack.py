from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from invarlock.public_contracts import load_policy_pack_schema

try:  # pragma: no cover - exercised in integration/tests
    import jsonschema
except Exception:  # pragma: no cover
    jsonschema = None

POLICY_PACK_FORMAT = "policy-pack-v1"


def _load_structured_text(text: str, *, suffix: str) -> Any:
    if suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def _load_structured_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    return _load_structured_text(text, suffix=path.suffix)


def _normalize_overrides(overrides: Any) -> list[dict[str, Any]]:
    if overrides is None:
        return []
    if isinstance(overrides, list):
        normalized: list[dict[str, Any]] = []
        for item in overrides:
            if isinstance(item, dict):
                normalized.append(item)
            else:
                normalized.append({"value": item})
        return normalized
    if isinstance(overrides, dict):
        return [{"path": key, "value": value} for key, value in overrides.items()]
    return [{"value": overrides}]


def _compute_policy_pack_digest(policy: dict[str, Any]) -> str:
    canonical = json.dumps(policy, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def compute_policy_pack_digest(
    *, resolved_policy: dict[str, Any], overrides: list[dict[str, Any]]
) -> str:
    digest_payload = {
        "resolved_policy": resolved_policy,
        "overrides": overrides,
    }
    return _compute_policy_pack_digest(digest_payload)


def build_policy_pack(
    *,
    tier: str,
    resolved_policy: dict[str, Any],
    overrides: list[dict[str, Any]] | None = None,
    compatibility: dict[str, Any] | None = None,
    approval: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_overrides = _normalize_overrides(overrides)
    compatibility_obj = compatibility if isinstance(compatibility, dict) else {}
    compatibility_obj.setdefault("support_tiers", ["published_basis"])
    pack: dict[str, Any] = {
        "format": POLICY_PACK_FORMAT,
        "tier": str(tier),
        "resolved_policy": resolved_policy,
        "overrides": normalized_overrides,
        "policy_digest": compute_policy_pack_digest(
            resolved_policy=resolved_policy,
            overrides=normalized_overrides,
        ),
        "compatibility": compatibility_obj,
    }
    if isinstance(approval, dict) and approval:
        pack["approval"] = approval
    if isinstance(metadata, dict) and metadata:
        pack["metadata"] = metadata
    return pack


def load_policy_pack(path: Path) -> dict[str, Any]:
    payload = _load_structured_file(path)
    if not isinstance(payload, dict):
        raise ValueError("policy pack must decode to a JSON/YAML object")
    return payload


def write_policy_pack(path: Path, pack: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pack, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_policy_pack(pack: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(pack, dict):
        return ["policy pack must be a mapping"]
    if pack.get("format") != POLICY_PACK_FORMAT:
        errors.append(
            f"policy pack format must be {POLICY_PACK_FORMAT} (found {pack.get('format')!r})"
        )

    schema = load_policy_pack_schema()
    if schema and jsonschema is not None:
        try:
            jsonschema.validate(instance=pack, schema=schema)
        except Exception as exc:
            errors.append(f"schema validation failed: {exc}")

    resolved_policy = pack.get("resolved_policy")
    if not isinstance(resolved_policy, dict):
        errors.append("resolved_policy must be an object")
        return errors

    overrides = pack.get("overrides")
    if not isinstance(overrides, list):
        errors.append("overrides must be an ordered list")
        return errors

    expected_digest = compute_policy_pack_digest(
        resolved_policy=resolved_policy,
        overrides=_normalize_overrides(overrides),
    )
    observed_digest = pack.get("policy_digest")
    if observed_digest != expected_digest:
        errors.append(
            f"policy digest mismatch: observed={observed_digest!r} expected={expected_digest!r}"
        )
    return errors


__all__ = [
    "POLICY_PACK_FORMAT",
    "build_policy_pack",
    "compute_policy_pack_digest",
    "load_policy_pack",
    "verify_policy_pack",
    "write_policy_pack",
]
