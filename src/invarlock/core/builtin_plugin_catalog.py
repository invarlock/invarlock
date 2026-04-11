"""Table-driven catalog for shipped plugin metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BuiltinPluginSpec:
    name: str
    module: str
    class_name: str
    required_deps: tuple[str, ...] = ()


BUILTIN_PLUGIN_CATALOG: dict[str, tuple[BuiltinPluginSpec, ...]] = {
    "adapters": (
        BuiltinPluginSpec(
            name="hf_causal",
            module="invarlock.adapters.hf_causal",
            class_name="HF_Causal_Adapter",
        ),
        BuiltinPluginSpec(
            name="hf_mlm",
            module="invarlock.adapters.hf_mlm",
            class_name="HF_MLM_Adapter",
        ),
        BuiltinPluginSpec(
            name="hf_multimodal",
            module="invarlock.adapters.hf_multimodal",
            class_name="HF_Multimodal_Adapter",
        ),
        BuiltinPluginSpec(
            name="hf_seq2seq",
            module="invarlock.adapters.hf_seq2seq",
            class_name="HF_Seq2Seq_Adapter",
        ),
        BuiltinPluginSpec(
            name="hf_auto",
            module="invarlock.adapters.auto",
            class_name="HF_Auto_Adapter",
        ),
        BuiltinPluginSpec(
            name="hf_gptq",
            module="invarlock.plugins.hf_gptq_adapter",
            class_name="HF_GPTQ_Adapter",
            required_deps=("auto_gptq",),
        ),
        BuiltinPluginSpec(
            name="hf_awq",
            module="invarlock.plugins.hf_awq_adapter",
            class_name="HF_AWQ_Adapter",
            required_deps=("awq",),
        ),
        BuiltinPluginSpec(
            name="hf_bnb",
            module="invarlock.plugins.hf_bnb_adapter",
            class_name="HF_BNB_Adapter",
            required_deps=("bitsandbytes",),
        ),
    ),
    "edits": (
        BuiltinPluginSpec(
            name="quant_rtn",
            module="invarlock.edits.quant_rtn",
            class_name="RTNQuantEdit",
        ),
        BuiltinPluginSpec(
            name="noop",
            module="invarlock.edits.noop",
            class_name="NoopEdit",
        ),
    ),
    "guards": (
        BuiltinPluginSpec(
            name="invariants",
            module="invarlock.guards.invariants",
            class_name="InvariantsGuard",
        ),
        BuiltinPluginSpec(
            name="spectral",
            module="invarlock.guards.spectral",
            class_name="SpectralGuard",
        ),
        BuiltinPluginSpec(
            name="variance",
            module="invarlock.guards.variance",
            class_name="VarianceGuard",
        ),
        BuiltinPluginSpec(
            name="rmt",
            module="invarlock.guards.rmt",
            class_name="RMTGuard",
        ),
        BuiltinPluginSpec(
            name="hello_guard",
            module="invarlock.plugins.hello_guard",
            class_name="HelloGuard",
        ),
    ),
}


def builtin_plugin_specs(plugin_type: str) -> tuple[BuiltinPluginSpec, ...]:
    try:
        return BUILTIN_PLUGIN_CATALOG[plugin_type]
    except KeyError as error:
        raise ValueError(f"Unknown plugin catalog type: {plugin_type}") from error


__all__ = [
    "BuiltinPluginSpec",
    "BUILTIN_PLUGIN_CATALOG",
    "builtin_plugin_specs",
]
