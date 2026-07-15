"""Capability-based claim selection for runtime-provider comparisons.

The existing weight-edit assurance claim is intentionally narrower than the new
deployment-runtime behavioral claim.  Keeping this decision in one exact gate
prevents an opaque runtime from acquiring weight-space authority by merely naming
the old claim in plugin metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..assurance_contract import ASSURANCE_CLAIM_SET
from .types import RuntimeProviderCapabilities

RUNTIME_BEHAVIORAL_CLAIM_SET = "invarlock-runtime-behavioral-regression-v1"

_SUPPORTED_RUNTIME_CLAIMS = frozenset(
    {ASSURANCE_CLAIM_SET, RUNTIME_BEHAVIORAL_CLAIM_SET}
)
_WEIGHT_EDIT_EVIDENCE = frozenset(
    {"behavior", "tokenizer", "weights", "modules", "activations"}
)
_BEHAVIORAL_EVIDENCE = frozenset({"behavior", "tokenizer"})
_REPLAYABLE_BEHAVIORAL_METRICS = frozenset({"exact_match"})


@dataclass(frozen=True)
class RuntimeClaimCompatibility:
    """Result of exact capability checks for one baseline/subject pair."""

    claim_set: str
    shared_metrics: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _missing_capability_errors(
    *,
    role: str,
    capabilities: RuntimeProviderCapabilities,
    claim_set: str,
    required_surfaces: frozenset[str],
) -> list[str]:
    errors: list[str] = []
    if claim_set not in capabilities.supported_claim_sets:
        errors.append(
            f"{role} provider '{capabilities.provider_name}' does not support "
            f"claim set '{claim_set}'."
        )
    if "text_causal" not in capabilities.tasks:
        errors.append(
            f"{role} provider '{capabilities.provider_name}' must support text_causal."
        )
    missing_surfaces = sorted(
        required_surfaces.difference(capabilities.evidence_surfaces)
    )
    if missing_surfaces:
        errors.append(
            f"{role} provider '{capabilities.provider_name}' lacks required "
            f"{', '.join(missing_surfaces)} evidence."
        )
    return errors


def evaluate_runtime_claim_compatibility(
    claim_set: str,
    *,
    baseline: RuntimeProviderCapabilities,
    subject: RuntimeProviderCapabilities,
) -> RuntimeClaimCompatibility:
    """Evaluate exact provider capabilities without loading either backend."""

    if claim_set not in _SUPPORTED_RUNTIME_CLAIMS:
        raise ValueError(f"Unsupported runtime claim set: {claim_set!r}")

    required_surfaces = (
        _WEIGHT_EDIT_EVIDENCE
        if claim_set == ASSURANCE_CLAIM_SET
        else _BEHAVIORAL_EVIDENCE
    )
    errors: list[str] = []
    for role, capabilities in (("baseline", baseline), ("subject", subject)):
        errors.extend(
            _missing_capability_errors(
                role=role,
                capabilities=capabilities,
                claim_set=claim_set,
                required_surfaces=required_surfaces,
            )
        )

    shared_metrics = tuple(sorted(set(baseline.metrics).intersection(subject.metrics)))

    if claim_set == ASSURANCE_CLAIM_SET:
        for role, capabilities in (("baseline", baseline), ("subject", subject)):
            if capabilities.provider_name != "hf_transformers":
                errors.append(
                    f"{role} provider must be hf_transformers for "
                    f"'{ASSURANCE_CLAIM_SET}'."
                )
            if "in_process" not in capabilities.execution_modes:
                errors.append(
                    f"{role} provider '{capabilities.provider_name}' must expose "
                    "in_process execution for the weight-edit claim."
                )
    else:
        shared_metrics = tuple(
            metric
            for metric in shared_metrics
            if metric in _REPLAYABLE_BEHAVIORAL_METRICS
        )
        if not shared_metrics:
            errors.append(
                "Runtime behavioral assurance requires the shared "
                "verifier-replayable metric exact_match."
            )

    return RuntimeClaimCompatibility(
        claim_set=claim_set,
        shared_metrics=shared_metrics,
        errors=tuple(errors),
    )


def require_runtime_claim_compatibility(
    claim_set: str,
    *,
    baseline: RuntimeProviderCapabilities,
    subject: RuntimeProviderCapabilities,
) -> RuntimeClaimCompatibility:
    """Return compatibility or fail closed with every capability error."""

    result = evaluate_runtime_claim_compatibility(
        claim_set,
        baseline=baseline,
        subject=subject,
    )
    if not result.ok:
        raise ValueError(" ".join(result.errors))
    return result


__all__ = [
    "RUNTIME_BEHAVIORAL_CLAIM_SET",
    "RuntimeClaimCompatibility",
    "evaluate_runtime_claim_compatibility",
    "require_runtime_claim_compatibility",
]
