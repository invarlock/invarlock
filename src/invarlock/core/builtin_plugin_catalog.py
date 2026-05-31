"""Table-driven catalog for shipped plugin metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BuiltinPluginSpec:
    name: str
    module: str
    class_name: str
    required_deps: tuple[str, ...] = ()
    support_tier: str = "core_supported"
    strict_assurance_allowed: bool = True
    published_basis: bool = False
    deployment_claim: bool = False

    def support_metadata(self) -> dict[str, object]:
        return {
            "support_tier": self.support_tier,
            "strict_assurance_allowed": self.strict_assurance_allowed,
            "published_basis": self.published_basis,
            "deployment_claim": self.deployment_claim,
        }


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
            module="invarlock.plugins",
            class_name="HF_GPTQ_Adapter",
            required_deps=("gptqmodel",),
            support_tier="optional_backend_loader",
        ),
        BuiltinPluginSpec(
            name="hf_awq",
            module="invarlock.plugins",
            class_name="HF_AWQ_Adapter",
            required_deps=("gptqmodel",),
            support_tier="optional_backend_loader",
        ),
        BuiltinPluginSpec(
            name="hf_bnb",
            module="invarlock.plugins",
            class_name="HF_BNB_Adapter",
            required_deps=("bitsandbytes",),
            support_tier="optional_backend_loader",
        ),
    ),
    "edits": (
        BuiltinPluginSpec(
            name="quant_rtn",
            module="invarlock.edits.quant_rtn",
            class_name="RTNQuantEdit",
            support_tier="validation_simulation",
        ),
        BuiltinPluginSpec(
            name="noop",
            module="invarlock.edits",
            class_name="NoopEdit",
            support_tier="internal_baseline_edit",
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
            name="demo_hello_guard",
            module="invarlock.plugins",
            class_name="HelloGuard",
            support_tier="demo_only",
            strict_assurance_allowed=False,
        ),
    ),
}


def builtin_plugin_specs(plugin_type: str) -> tuple[BuiltinPluginSpec, ...]:
    try:
        return BUILTIN_PLUGIN_CATALOG[plugin_type]
    except KeyError as error:
        raise ValueError(f"Unknown plugin catalog type: {plugin_type}") from error


def builtin_plugin_support_metadata(
    plugin_type: str,
    name: str,
) -> dict[str, object]:
    for spec in builtin_plugin_specs(plugin_type):
        if spec.name == name:
            return spec.support_metadata()
    return {
        "support_tier": "third_party",
        "strict_assurance_allowed": False,
        "published_basis": False,
        "deployment_claim": False,
    }


__all__ = [
    "BuiltinPluginSpec",
    "BUILTIN_PLUGIN_CATALOG",
    "builtin_plugin_specs",
    "builtin_plugin_support_metadata",
]
