"""Local conformance command for the TensorRT-LLM provider add-in."""

from __future__ import annotations

import json

from invarlock.core.runtime_provider import (
    INVARLOCK_RUNTIME_PROVIDER_ABI,
    RuntimeProvider,
)

from .provider import TensorRTLLMProvider


def conformance_payload() -> dict[str, object]:
    """Return a machine-readable provider ABI conformance result."""
    candidate: object = TensorRTLLMProvider()
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
    if provider.name != "tensorrt_llm":
        errors.append("provider name must be tensorrt_llm")
    if provider.abi_version != INVARLOCK_RUNTIME_PROVIDER_ABI:
        errors.append("provider ABI does not match the installed core")
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


if __name__ == "__main__":  # pragma: no cover - exercised by image smoke
    raise SystemExit(main())
