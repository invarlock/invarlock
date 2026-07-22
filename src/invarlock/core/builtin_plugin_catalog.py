"""Table-driven catalog for the single built-in runtime provider."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BuiltinPluginSpec:
    """Import metadata for a provider shipped by the core distribution."""

    name: str
    module: str
    class_name: str
    required_deps: tuple[str, ...] = ()


BUILTIN_RUNTIME_PROVIDERS = (
    BuiltinPluginSpec(
        name="hf_transformers",
        module="invarlock.runtime_providers.hf_transformers",
        class_name="HFTransformersProvider",
        # Identity and import-mode verification stay available in the base
        # distribution; execution imports the optional backend lazily.
        required_deps=(),
    ),
)


def builtin_plugin_specs(plugin_type: str) -> tuple[BuiltinPluginSpec, ...]:
    """Return the closed built-in catalog for the runtime-provider ABI."""

    if plugin_type != "runtime_providers":
        raise ValueError(f"Unknown plugin catalog type: {plugin_type}")
    return BUILTIN_RUNTIME_PROVIDERS


__all__ = [
    "BUILTIN_RUNTIME_PROVIDERS",
    "BuiltinPluginSpec",
    "builtin_plugin_specs",
]
