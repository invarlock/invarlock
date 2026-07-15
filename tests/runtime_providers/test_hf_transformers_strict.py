from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.core.runtime_provider import (
    EvaluationBatch,
    EvaluationRecord,
    ModelRuntimeSpec,
    RuntimeExecutionContext,
    RuntimeExecutionSettings,
)
from invarlock.core.runtime_provider.types import JSONScalar
from invarlock.runtime_providers import hf_transformers
from invarlock.runtime_providers.hf_transformers import (
    HFTransformersCausalScorer,
    HFTransformersProvider,
    hf_tokenizer_contract_sha256,
)
from tests.runtime_providers._hf_transformers_helpers import (
    _IMAGE_DIGEST,
    _REAL_STRICT_EXECUTION_BINDING,
    _artifact_sha256,
    _authenticated_test_runtime,  # noqa: F401
    _batch,
    _BindingTokenizer,
    _observation,
    _spec,
)


def test_hf_strict_open_rejects_arbitrary_scorer_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hf_transformers,
        "_require_strict_execution_binding",
        _REAL_STRICT_EXECUTION_BINDING,
    )
    spec = _spec()

    with pytest.raises(RuntimeError, match="provider-owned.*arbitrary scorer"):
        HFTransformersProvider().open(
            spec,
            RuntimeExecutionContext(
                strict=True,
                allow_network=False,
                container_image_digest=_IMAGE_DIGEST,
                device_kind="cpu",
                artifact_identity_sha256=_artifact_sha256(spec),
                model_adapter=object(),
                native_model=object(),
                scorer=_observation,
            ),
        )


def test_hf_strict_open_rejects_scorer_bound_to_different_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hf_transformers,
        "_require_strict_execution_binding",
        _REAL_STRICT_EXECUTION_BINDING,
    )
    spec = _spec()
    native_model = object()
    scorer = HFTransformersCausalScorer(
        model=object(),
        tokenizer=_BindingTokenizer(),
        artifact_identity_sha256=_artifact_sha256(spec),
    )

    with pytest.raises(ValueError, match="exact native model"):
        HFTransformersProvider().open(
            spec,
            RuntimeExecutionContext(
                strict=True,
                allow_network=False,
                container_image_digest=_IMAGE_DIGEST,
                device_kind="cpu",
                artifact_identity_sha256=_artifact_sha256(spec),
                model_adapter=object(),
                native_model=native_model,
                scorer=scorer,
            ),
        )


def test_hf_strict_open_rejects_remote_identity_without_local_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hf_transformers,
        "_require_strict_execution_binding",
        _REAL_STRICT_EXECUTION_BINDING,
    )
    spec = _spec()
    model = object()
    scorer = HFTransformersCausalScorer(
        model=model,
        tokenizer=_BindingTokenizer(),
        artifact_identity_sha256=_artifact_sha256(spec),
    )

    with pytest.raises(RuntimeError, match="materialized local checkpoint"):
        HFTransformersProvider().open(
            spec,
            RuntimeExecutionContext(
                strict=True,
                allow_network=False,
                container_image_digest=_IMAGE_DIGEST,
                device_kind="cpu",
                artifact_identity_sha256=_artifact_sha256(spec),
                model_adapter=object(),
                native_model=model,
                scorer=scorer,
            ),
        )


def test_hf_strict_open_rejects_live_tokenizer_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        hf_transformers,
        "_require_strict_execution_binding",
        _REAL_STRICT_EXECUTION_BINDING,
    )
    checkpoint = tmp_path / "authenticated-hf"
    checkpoint.mkdir()
    checkpoint.joinpath("model.safetensors").write_bytes(b"not-read-before-tokenizer")
    model = object()
    spec = ModelRuntimeSpec(
        provider_name="hf_transformers",
        model_id=str(checkpoint),
        adapter_name="hf_causal",
        settings={
            "batch_size": 1,
            "checkpoint_tree_sha256": checkpoint_tree_sha256(checkpoint),
            "context_length": 8,
            "max_output_tokens": 1,
            "offline": True,
            "seed": 17,
            "timeout_seconds": 30,
            "tokenizer_metadata_sha256": "9" * 64,
        },
    )
    identity_sha256 = _artifact_sha256(spec)
    scorer = HFTransformersCausalScorer(
        model=model,
        tokenizer=_BindingTokenizer(),
        artifact_identity_sha256=identity_sha256,
    )

    with pytest.raises(ValueError, match="live tokenizer"):
        HFTransformersProvider().open(
            spec,
            RuntimeExecutionContext(
                strict=True,
                allow_network=False,
                container_image_digest=_IMAGE_DIGEST,
                device_kind="cpu",
                artifact_identity_sha256=identity_sha256,
                model_adapter=object(),
                native_model=model,
                scorer=scorer,
            ),
        )


def test_hf_strict_open_rejects_training_mode_submodule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        hf_transformers,
        "_require_strict_execution_binding",
        _REAL_STRICT_EXECUTION_BINDING,
    )
    checkpoint = tmp_path / "authenticated-hf"
    checkpoint.mkdir()
    checkpoint.joinpath("model.safetensors").write_bytes(b"not-read-before-eval")
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5))
    model.eval()
    model[1].train()
    tokenizer = _BindingTokenizer()
    spec = ModelRuntimeSpec(
        provider_name="hf_transformers",
        model_id=str(checkpoint),
        adapter_name="hf_causal",
        settings={
            "batch_size": 1,
            "checkpoint_tree_sha256": checkpoint_tree_sha256(checkpoint),
            "context_length": 8,
            "max_output_tokens": 1,
            "offline": True,
            "seed": 17,
            "timeout_seconds": 30,
            "tokenizer_metadata_sha256": hf_tokenizer_contract_sha256(tokenizer),
        },
    )
    identity_sha256 = _artifact_sha256(spec)
    scorer = HFTransformersCausalScorer(
        model=model,
        tokenizer=tokenizer,
        artifact_identity_sha256=identity_sha256,
    )

    with pytest.raises(RuntimeError, match=r"model\.eval\(\).*every submodule"):
        HFTransformersProvider().open(
            spec,
            RuntimeExecutionContext(
                strict=True,
                allow_network=False,
                container_image_digest=_IMAGE_DIGEST,
                device_kind="cpu",
                artifact_identity_sha256=identity_sha256,
                model_adapter=object(),
                native_model=model,
                scorer=scorer,
            ),
        )


def test_hf_owned_scorer_enables_and_restores_deterministic_algorithms() -> None:
    torch = pytest.importorskip("torch")
    observed: list[bool] = []

    class TinyCausal(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, *, input_ids, **_kwargs):
            observed.append(bool(torch.are_deterministic_algorithms_enabled()))
            logits = torch.zeros((1, input_ids.shape[1], 4), device=input_ids.device)
            logits[:, -1, 2] = self.weight
            return SimpleNamespace(logits=logits)

    class TinyTokenizer:
        eos_token_id = None

        def __call__(self, _text: str, **_kwargs):
            return {"input_ids": torch.tensor([[1]], dtype=torch.long)}

        def decode(self, token_ids, **_kwargs) -> str:
            return " ".join(str(token_id) for token_id in token_ids)

    model = TinyCausal().eval()
    scorer = HFTransformersCausalScorer(
        model=model,
        tokenizer=TinyTokenizer(),
        artifact_identity_sha256="a" * 64,
    )
    batch = EvaluationBatch(
        schedule_sha256="b" * 64,
        records=(
            EvaluationRecord(
                record_id="sample",
                input_text="hello",
                input_sha256=hashlib.sha256(b"hello").hexdigest(),
            ),
        ),
    )
    settings = RuntimeExecutionSettings(
        seed=7,
        context_length=8,
        batch_size=1,
        max_output_tokens=1,
        timeout_seconds=30,
    )
    prior_enabled = bool(torch.are_deterministic_algorithms_enabled())
    prior_warn_only = bool(torch.is_deterministic_algorithms_warn_only_enabled())
    torch.use_deterministic_algorithms(False, warn_only=False)
    try:
        observation = scorer(batch, settings)
        assert observation.records[0].output_text == "2"
        assert observed == [True]
        assert torch.are_deterministic_algorithms_enabled() is False
    finally:
        torch.use_deterministic_algorithms(
            prior_enabled,
            warn_only=prior_warn_only,
        )


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"offline": False}, "offline=true"),
        ({"seed": None}, "seed must be a non-negative integer"),
        ({"batch_size": None}, "batch_size must be a positive integer"),
    ],
)
def test_hf_strict_receipt_requires_exact_offline_execution_settings(
    settings: dict[str, JSONScalar], message: str
) -> None:
    spec = _spec(**settings)
    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest=_IMAGE_DIGEST,
        device_kind="cpu",
        artifact_identity_sha256=_artifact_sha256(spec),
        model_adapter=object(),
        native_model=object(),
        scorer=_observation,
    )

    with pytest.raises(ValueError, match=message):
        HFTransformersProvider().open(spec, context)


def test_hf_strict_receipt_rejects_missing_execution_setting() -> None:
    settings = dict(_spec().settings)
    del settings["timeout_seconds"]
    spec = ModelRuntimeSpec(
        provider_name="hf_transformers",
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        adapter_name="hf_causal",
        settings=settings,
    )
    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest=_IMAGE_DIGEST,
        device_kind="cpu",
        artifact_identity_sha256=_artifact_sha256(spec),
        model_adapter=object(),
        native_model=object(),
        scorer=_observation,
    )

    with pytest.raises(ValueError, match="timeout_seconds"):
        HFTransformersProvider().open(spec, context)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("strict_container_boundary_present", False, "container boundary"),
        ("network_allowed", True, "offline"),
        ("remote_code_allowed", True, "remote code"),
        ("third_party_plugins_allowed", True, "third-party plugins"),
    ],
)
def test_hf_strict_receipt_rejects_untrusted_runtime_allowances(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    value: bool,
    message: str,
) -> None:
    monkeypatch.setattr(hf_transformers, attribute, lambda: value)
    spec = _spec()
    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest=_IMAGE_DIGEST,
        device_kind="cpu",
        artifact_identity_sha256=_artifact_sha256(spec),
        model_adapter=object(),
        native_model=object(),
        scorer=_observation,
    )

    with pytest.raises(ValueError, match=message):
        HFTransformersProvider().open(spec, context)


@pytest.mark.parametrize("image_binding", ["digest", "reference"])
def test_hf_strict_receipt_rejects_runtime_image_drift(
    monkeypatch: pytest.MonkeyPatch, image_binding: str
) -> None:
    if image_binding == "digest":
        monkeypatch.setattr(
            hf_transformers,
            "resolve_runtime_image_digest",
            lambda: "sha256:" + "9" * 64,
        )
        message = "image digest"
    else:
        monkeypatch.setattr(
            hf_transformers,
            "resolve_runtime_image",
            lambda: "registry.invalid/invarlock:mutable",
        )
        message = "image reference"
    spec = _spec()
    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest=_IMAGE_DIGEST,
        device_kind="cpu",
        artifact_identity_sha256=_artifact_sha256(spec),
        model_adapter=object(),
        native_model=object(),
        scorer=_observation,
    )

    with pytest.raises(ValueError, match=message):
        HFTransformersProvider().open(spec, context)


def test_hf_nonstrict_session_never_emits_a_strict_receipt() -> None:
    spec = _spec()
    batch = _batch()
    session = HFTransformersProvider().open(
        spec,
        RuntimeExecutionContext(
            strict=False,
            allow_network=True,
            container_image_digest=None,
            device_kind="cpu",
            artifact_identity_sha256=_artifact_sha256(spec),
            model_adapter=object(),
            native_model=object(),
            scorer=lambda candidate: _observation(
                candidate, artifact_sha256=_artifact_sha256(spec)
            ),
        ),
    )

    session.score(batch)
    with pytest.raises(RuntimeError, match="strict authenticated execution"):
        session.runtime_receipt()
