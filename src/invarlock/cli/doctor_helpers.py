from __future__ import annotations

from typing import Any

from .backend_runtime import bitsandbytes_runtime_available


def get_adapter_rows() -> list[dict[str, Any]]:
    """Build adapter rows similar to doctor output for testing.

    Mirrors doctor adapter output without importing heavy optional backends.
    """
    from invarlock.core.registry import get_registry

    registry = get_registry()
    rows: list[dict[str, Any]] = []
    for name in registry.list_adapters():
        info = registry.get_plugin_info(name, "adapters")
        module = str(info.get("module") or "")
        support = (
            "auto"
            if module.startswith("invarlock.adapters") and name in {"hf_auto"}
            else ("core" if module.startswith("invarlock.adapters") else "optional")
        )
        backend, status, enable = None, "ready", ""

        if name in {
            "hf_causal",
            "hf_mlm",
            "hf_multimodal",
            "hf_seq2seq",
            "hf_auto",
        }:
            backend = "transformers"
        elif name == "hf_gptq":
            backend = "gptqmodel"
        elif name == "hf_awq":
            backend = "gptqmodel"
        elif name == "hf_bnb":
            backend = "bitsandbytes"
            if not bitsandbytes_runtime_available():
                status, enable = (
                    "unsupported",
                    "Requires CUDA or a compatible bitsandbytes runtime",
                )

        rows.append(
            {
                "name": name,
                "origin": "core" if support in {"core", "auto"} else "plugin",
                "mode": "auto-matcher" if support == "auto" else "adapter",
                "backend": backend,
                "version": None,
                "status": status,
                "enable": enable,
            }
        )

    return rows
