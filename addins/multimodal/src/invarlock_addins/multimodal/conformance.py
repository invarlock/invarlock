"""Machine-readable ABI conformance check for the vision-text add-in."""

from __future__ import annotations

import json

from invarlock.core.runtime_provider import (
    INVARLOCK_RUNTIME_PROVIDER_ABI,
    RuntimeProvider,
)

from .provider import HFVisionTextProvider


def conformance_payload() -> dict[str, object]:
    """Return provider identity and ABI conformance without loading backends."""

    candidate: object = HFVisionTextProvider()
    errors: list[str] = []
    if not isinstance(candidate, RuntimeProvider):
        errors.append("provider does not implement RuntimeProvider")
        return {
            "abi_version": None,
            "errors": errors,
            "format_version": "invarlock/runtime-provider-conformance-v1",
            "ok": False,
            "provider": None,
        }
    provider = candidate
    if provider.name != "hf_vision_text":
        errors.append("provider name must be hf_vision_text")
    if provider.abi_version != INVARLOCK_RUNTIME_PROVIDER_ABI:
        errors.append("provider ABI does not match the installed core")
    capabilities = provider.capabilities()
    if capabilities.tasks != ("vision_text_generation",):
        errors.append("provider must expose only vision_text_generation")
    if capabilities.metrics != ("exact_match",):
        errors.append("provider must expose only exact_match")
    return {
        "abi_version": provider.abi_version,
        "errors": errors,
        "format_version": "invarlock/runtime-provider-conformance-v1",
        "ok": not errors,
        "provider": provider.name,
    }


def main() -> int:
    payload = conformance_payload()
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0 if payload["ok"] else 1


__all__ = ["conformance_payload", "main"]


if __name__ == "__main__":  # pragma: no cover - package smoke
    raise SystemExit(main())
