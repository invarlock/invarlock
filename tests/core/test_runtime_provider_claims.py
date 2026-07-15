from __future__ import annotations

import dataclasses

import pytest

from invarlock.core.assurance_contract import ASSURANCE_CLAIM_SET
from invarlock.core.runtime_provider.claims import (
    RUNTIME_BEHAVIORAL_CLAIM_SET,
    RuntimeClaimCompatibility,
    evaluate_runtime_claim_compatibility,
    require_runtime_claim_compatibility,
)
from invarlock.core.runtime_provider.types import RuntimeProviderCapabilities


def _capabilities(
    name: str,
    *,
    formats: tuple[str, ...],
    modes: tuple[str, ...],
    surfaces: tuple[str, ...],
    claims: tuple[str, ...],
    metrics: tuple[str, ...] = ("exact_match", "multiple_choice_accuracy"),
) -> RuntimeProviderCapabilities:
    return RuntimeProviderCapabilities(
        provider_name=name,
        artifact_formats=formats,  # type: ignore[arg-type]
        tasks=("text_causal",),
        metrics=metrics,  # type: ignore[arg-type]
        execution_modes=modes,  # type: ignore[arg-type]
        required_extra=None,
        required_image=None,
        platform_constraints=(),
        evidence_surfaces=surfaces,  # type: ignore[arg-type]
        supported_claim_sets=claims,
    )


def _hf_capabilities() -> RuntimeProviderCapabilities:
    return _capabilities(
        "hf_transformers",
        formats=("hf_snapshot",),
        modes=("in_process",),
        surfaces=("behavior", "tokenizer", "weights", "modules", "activations"),
        claims=(ASSURANCE_CLAIM_SET, RUNTIME_BEHAVIORAL_CLAIM_SET),
    )


def _gguf_capabilities() -> RuntimeProviderCapabilities:
    return _capabilities(
        "llama_cpp",
        formats=("gguf",),
        modes=("local_process", "container"),
        surfaces=("behavior", "tokenizer", "build"),
        claims=(RUNTIME_BEHAVIORAL_CLAIM_SET,),
    )


def test_weight_edit_claim_is_explicitly_hf_in_process_and_full_evidence() -> None:
    result = evaluate_runtime_claim_compatibility(
        ASSURANCE_CLAIM_SET,
        baseline=_hf_capabilities(),
        subject=_hf_capabilities(),
    )

    assert result == RuntimeClaimCompatibility(
        claim_set=ASSURANCE_CLAIM_SET,
        shared_metrics=("exact_match", "multiple_choice_accuracy"),
        errors=(),
    )


def test_weight_edit_claim_rejects_opaque_provider_even_if_it_self_declares_claim() -> (
    None
):
    dishonest = dataclasses.replace(
        _gguf_capabilities(),
        supported_claim_sets=(ASSURANCE_CLAIM_SET,),
        evidence_surfaces=(
            "behavior",
            "tokenizer",
            "weights",
            "modules",
            "activations",
        ),
        execution_modes=("in_process",),
    )

    result = evaluate_runtime_claim_compatibility(
        ASSURANCE_CLAIM_SET,
        baseline=_hf_capabilities(),
        subject=dishonest,
    )

    assert result.ok is False
    assert "hf_transformers" in " ".join(result.errors)


def test_behavioral_claim_accepts_hf_to_gguf_without_weight_surfaces() -> None:
    result = require_runtime_claim_compatibility(
        RUNTIME_BEHAVIORAL_CLAIM_SET,
        baseline=_hf_capabilities(),
        subject=_gguf_capabilities(),
    )

    assert result.ok is True
    assert result.shared_metrics == ("exact_match", "multiple_choice_accuracy")


def test_behavioral_claim_fails_closed_without_shared_replayable_metric() -> None:
    gguf = dataclasses.replace(
        _gguf_capabilities(), metrics=("normalized_nll_per_utf8_byte",)
    )

    with pytest.raises(ValueError, match="shared verifier-replayable metric"):
        require_runtime_claim_compatibility(
            RUNTIME_BEHAVIORAL_CLAIM_SET,
            baseline=_hf_capabilities(),
            subject=gguf,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("evidence_surfaces", ("tokenizer", "build"), "behavior evidence"),
        ("supported_claim_sets", (ASSURANCE_CLAIM_SET,), "does not support"),
    ],
)
def test_behavioral_claim_rejects_missing_exact_capabilities(
    field: str, value: tuple[str, ...], message: str
) -> None:
    subject = dataclasses.replace(_gguf_capabilities(), **{field: value})

    result = evaluate_runtime_claim_compatibility(
        RUNTIME_BEHAVIORAL_CLAIM_SET,
        baseline=_hf_capabilities(),
        subject=subject,
    )

    assert result.ok is False
    assert message in " ".join(result.errors)


def test_unknown_runtime_claim_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported runtime claim set"):
        require_runtime_claim_compatibility(
            "future-claim-v9",
            baseline=_hf_capabilities(),
            subject=_hf_capabilities(),
        )
