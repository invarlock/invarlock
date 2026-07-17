from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.core.runtime_provider import (
    EvaluationBatch,
    EvaluationInputPart,
    EvaluationRecord,
    ModelRuntimeSpec,
    RuntimeExecutionContext,
    RuntimeExecutionSettings,
    evaluation_input_parts_sha256,
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


def test_hf_strict_binding_rejects_loader_initialized_live_parameters(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    authenticated_weight = torch.tensor([1.0, 2.0])
    safetensors_torch.save_file(
        {"transformer.wte.weight": authenticated_weight},
        tmp_path / "model.safetensors",
    )

    # Transformers accepts partial checkpoints by initializing missing model
    # parameters. Strict binding must reject that live-only execution state.
    model = SimpleNamespace(
        base_model_prefix="transformer",
        state_dict=lambda: {
            "transformer.wte.weight": authenticated_weight.clone(),
            "transformer.wpe.weight": torch.tensor([3.0, 4.0]),
        },
    )

    with pytest.raises(ValueError, match="unauthenticated live model tensors"):
        hf_transformers._require_safetensors_match(tmp_path, model=model)


def test_hf_strict_binding_rejects_ambiguous_exact_and_nested_candidates(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    authenticated_weight = torch.tensor([1.0, 2.0])
    safetensors_torch.save_file(
        {"model.weight": authenticated_weight},
        tmp_path / "model.safetensors",
    )
    model = SimpleNamespace(
        base_model_prefix="model",
        state_dict=lambda: {
            "model.weight": authenticated_weight.clone(),
            "model.language_model.weight": torch.tensor([2.0, 1.0]),
        },
    )

    with pytest.raises(
        ValueError,
        match="ambiguous authenticated checkpoint tensor mapping",
    ):
        hf_transformers._require_safetensors_match(tmp_path, model=model)


def test_hf_strict_binding_allows_proven_tied_live_parameter_alias(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    tied_weight = torch.tensor([1.0, 2.0])
    safetensors_torch.save_file(
        {"transformer.wte.weight": tied_weight},
        tmp_path / "model.safetensors",
    )
    model = SimpleNamespace(
        base_model_prefix="transformer",
        state_dict=lambda: {
            "transformer.wte.weight": tied_weight,
            "lm_head.weight": tied_weight.view_as(tied_weight),
        },
    )

    hf_transformers._require_safetensors_match(tmp_path, model=model)


def test_hf_strict_binding_rejects_shifted_live_parameter_view(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    shared_storage = torch.tensor([1.0, 2.0, 3.0])
    authenticated_weight = shared_storage[:2]
    safetensors_torch.save_file(
        {"transformer.wte.weight": authenticated_weight},
        tmp_path / "model.safetensors",
    )
    model = SimpleNamespace(
        base_model_prefix="transformer",
        state_dict=lambda: {
            "transformer.wte.weight": authenticated_weight,
            "lm_head.weight": shared_storage[1:],
        },
    )

    with pytest.raises(
        ValueError,
        match="unauthenticated live model tensors",
    ):
        hf_transformers._require_safetensors_match(tmp_path, model=model)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("missing_keys", {"transformer.wpe.weight"}),
        ("unexpected_keys", {"unbound.weight"}),
        ("mismatched_keys", {"transformer.wte.weight"}),
        ("error_msgs", ["loader failure"]),
    ],
)
def test_hf_strict_loader_rejects_incomplete_loading_info(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    model = SimpleNamespace(base_model_prefix="transformer")
    loading_info: dict[str, object] = {
        "missing_keys": set(),
        "unexpected_keys": set(),
        "mismatched_keys": set(),
        "error_msgs": [],
    }
    loading_info[field] = value

    with pytest.raises(ValueError, match="loading reported missing"):
        hf_transformers.load_hf_model_with_strict_loading_info(
            lambda *_args, **_kwargs: (model, loading_info),
            tmp_path,
        )


def test_hf_strict_loader_requires_and_returns_complete_loading_info(
    tmp_path: Path,
) -> None:
    model = SimpleNamespace(base_model_prefix="transformer")
    observed: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def loader(*args: object, **kwargs: object) -> object:
        observed.append((args, kwargs))
        return model, {
            "missing_keys": set(),
            "unexpected_keys": set(),
            "mismatched_keys": set(),
            "error_msgs": [],
        }

    loaded = hf_transformers.load_hf_model_with_strict_loading_info(
        loader,
        tmp_path,
    )

    assert loaded is model
    assert observed == [
        (
            (str(tmp_path),),
            {
                "local_files_only": True,
                "trust_remote_code": False,
                "use_safetensors": True,
                "output_loading_info": True,
            },
        )
    ]


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
                provider_state=object(),
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
    provider_state = object()
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
                provider_state=provider_state,
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
                provider_state=model,
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
                provider_state=model,
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
                provider_state=model,
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


def test_hf_owned_scorer_does_not_execute_future_task_contracts() -> None:
    torch = pytest.importorskip("torch")

    class TinyCausal(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    class NeverCalledTokenizer:
        eos_token_id = None

        def __call__(self, _text: str, **_kwargs):
            raise AssertionError("future task reached text tokenization")

        def decode(self, _token_ids, **_kwargs) -> str:
            raise AssertionError("future task reached text decoding")

    scorer = HFTransformersCausalScorer(
        model=TinyCausal().eval(),
        tokenizer=NeverCalledTokenizer(),
        artifact_identity_sha256="a" * 64,
    )
    batch = EvaluationBatch(
        schedule_sha256="b" * 64,
        records=(
            EvaluationRecord(
                record_id="future-1",
                input_text="Transcribe the authenticated audio.",
                input_sha256=hashlib.sha256(
                    b"Transcribe the authenticated audio."
                ).hexdigest(),
            ),
        ),
        task="audio_text_generation",
    )

    with pytest.raises(ValueError, match="supports only text_causal"):
        scorer(
            batch,
            RuntimeExecutionSettings(
                seed=7,
                context_length=8,
                batch_size=1,
                max_output_tokens=1,
                timeout_seconds=30,
            ),
        )


def test_hf_owned_scorer_rejects_structured_content_on_text_task() -> None:
    torch = pytest.importorskip("torch")

    class TinyCausal(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    class NeverCalledTokenizer:
        eos_token_id = None

        def __call__(self, _text: str, **_kwargs):
            raise AssertionError("structured content reached text tokenization")

        def decode(self, _token_ids, **_kwargs) -> str:
            raise AssertionError("structured content reached text decoding")

    prompt = "What is shown?"
    parts = (
        EvaluationInputPart(
            kind="content",
            role="image",
            content_id="image_1",
            media_type="image/png",
            byte_length=1,
            sha256="c" * 64,
        ),
        EvaluationInputPart(
            kind="text",
            role="prompt",
            text=prompt,
            sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        ),
    )
    scorer = HFTransformersCausalScorer(
        model=TinyCausal().eval(),
        tokenizer=NeverCalledTokenizer(),
        artifact_identity_sha256="a" * 64,
    )
    batch = EvaluationBatch(
        schedule_sha256="b" * 64,
        records=(
            EvaluationRecord(
                record_id="vision-1",
                input_text=prompt,
                input_sha256=evaluation_input_parts_sha256(parts),
                input_parts=parts,
            ),
        ),
    )

    with pytest.raises(ValueError, match="requires one prompt text input part"):
        scorer(
            batch,
            RuntimeExecutionSettings(
                seed=7,
                context_length=8,
                batch_size=1,
                max_output_tokens=1,
                timeout_seconds=30,
            ),
        )


def test_hf_exact_match_stops_at_eos_and_excludes_special_tokens() -> None:
    torch = pytest.importorskip("torch")
    decode_calls: list[dict[str, object]] = []

    class EosCausal(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, *, input_ids, **_kwargs):
            logits = torch.zeros((1, input_ids.shape[1], 4), device=input_ids.device)
            selected = 2 if input_ids.shape[1] == 1 else 3
            logits[:, -1, selected] = self.weight
            return SimpleNamespace(logits=logits)

    class EosTokenizer:
        eos_token_id = 3

        def __call__(self, _text: str, **_kwargs):
            return {"input_ids": torch.tensor([[1]], dtype=torch.long)}

        def decode(self, token_ids, **kwargs) -> str:
            decode_calls.append(dict(kwargs))
            values = [int(token_id) for token_id in token_ids]
            visible = ["visible" for token_id in values if token_id == 2]
            if kwargs["skip_special_tokens"] is not True and 3 in values:
                visible.append("<eos>")
            return "".join(visible)

    scorer = HFTransformersCausalScorer(
        model=EosCausal().eval(),
        tokenizer=EosTokenizer(),
        artifact_identity_sha256="a" * 64,
    )
    observation = scorer(
        EvaluationBatch(
            schedule_sha256="b" * 64,
            records=(
                EvaluationRecord(
                    record_id="sample",
                    input_text="prompt",
                    input_sha256=hashlib.sha256(b"prompt").hexdigest(),
                ),
            ),
        ),
        RuntimeExecutionSettings(
            seed=7,
            context_length=8,
            batch_size=1,
            max_output_tokens=4,
            timeout_seconds=30,
        ),
    )

    assert observation.records[0].output_text == "visible"
    assert decode_calls == [
        {
            "clean_up_tokenization_spaces": False,
            "skip_special_tokens": True,
        }
    ]


def test_hf_owned_scorer_emits_precomputed_teacher_forced_nll_facts() -> None:
    torch = pytest.importorskip("torch")

    class UniformCausal(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, *, input_ids, **_kwargs):
            logits = torch.zeros((1, input_ids.shape[1], 4), device=input_ids.device)
            return SimpleNamespace(logits=logits * self.weight)

    class FixedTokenizer:
        eos_token_id = None

        def __call__(self, text: str, *, add_special_tokens: bool, **_kwargs):
            if add_special_tokens:
                assert text == "prompt"
                token_ids = [1]
            else:
                assert text == "é"
                token_ids = [2, 3]
            return {"input_ids": torch.tensor([token_ids], dtype=torch.long)}

        def decode(self, token_ids, **_kwargs) -> str:
            values = [int(token_id) for token_id in token_ids]
            if values == [1]:
                return "prompt"
            if values == [1, 2, 3]:
                return "prompté"
            return "".join(str(token_id) for token_id in values)

    scorer = HFTransformersCausalScorer(
        model=UniformCausal().eval(),
        tokenizer=FixedTokenizer(),
        artifact_identity_sha256="a" * 64,
    )
    batch = EvaluationBatch(
        schedule_sha256="b" * 64,
        records=(
            EvaluationRecord(
                record_id="sample",
                input_text="prompt",
                input_sha256=hashlib.sha256(b"prompt").hexdigest(),
                expected_output="é",
            ),
        ),
        metric="normalized_nll_per_utf8_byte",
    )

    observation = scorer(
        batch,
        RuntimeExecutionSettings(
            seed=7,
            context_length=8,
            batch_size=1,
            max_output_tokens=2,
            timeout_seconds=30,
        ),
    )

    record = observation.records[0]
    # The four-token uniform distribution assigns log(1/4) to each of the two
    # fixed target tokens. This expected value is defined independently of the
    # scorer output and catches accidental generation-score substitution.
    assert record.logprob_sum == pytest.approx(-2.0 * math.log(4.0))
    assert record.token_count == 2
    assert record.utf8_byte_count == 2
    assert -record.logprob_sum / record.utf8_byte_count == pytest.approx(math.log(4.0))


def test_hf_owned_scorer_rejects_boundary_unstable_target_tokenization() -> None:
    torch = pytest.importorskip("torch")

    class NonUniformCausal(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, *, input_ids, **_kwargs):
            logits = torch.arange(8, dtype=torch.float32, device=input_ids.device)
            return SimpleNamespace(
                logits=logits.repeat(1, input_ids.shape[1], 1) * self.weight
            )

    class BoundaryChangingTokenizer:
        eos_token_id = None

        def __call__(self, text: str, *, add_special_tokens: bool, **_kwargs):
            token_ids = [1] if add_special_tokens else [2]
            return {"input_ids": torch.tensor([token_ids], dtype=torch.long)}

        def decode(self, token_ids, **_kwargs) -> str:
            values = [int(token_id) for token_id in token_ids]
            return {(1,): "A", (1, 2): "A é"}.get(tuple(values), "é")

    scorer = HFTransformersCausalScorer(
        model=NonUniformCausal().eval(),
        tokenizer=BoundaryChangingTokenizer(),
        artifact_identity_sha256="a" * 64,
    )
    batch = EvaluationBatch(
        schedule_sha256="b" * 64,
        records=(
            EvaluationRecord(
                record_id="sample",
                input_text="A",
                input_sha256=hashlib.sha256(b"A").hexdigest(),
                expected_output="é",
            ),
        ),
        metric="normalized_nll_per_utf8_byte",
    )

    with pytest.raises(ValueError, match="exact tokenizer continuation"):
        scorer(
            batch,
            RuntimeExecutionSettings(
                seed=7,
                context_length=8,
                batch_size=1,
                max_output_tokens=1,
                timeout_seconds=30,
            ),
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
        provider_state=object(),
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
        settings=settings,
    )
    context = RuntimeExecutionContext(
        strict=True,
        allow_network=False,
        container_image_digest=_IMAGE_DIGEST,
        device_kind="cpu",
        artifact_identity_sha256=_artifact_sha256(spec),
        provider_state=object(),
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
        provider_state=object(),
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
        provider_state=object(),
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
            provider_state=object(),
            scorer=lambda candidate: _observation(
                candidate, artifact_sha256=_artifact_sha256(spec)
            ),
        ),
    )

    session.score(batch)
    with pytest.raises(RuntimeError, match="strict authenticated execution"):
        session.runtime_receipt()


def test_hf_strict_rescore_reuses_authenticated_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "authenticated-checkpoint"
    checkpoint.mkdir()
    observed: list[Path | None] = []

    def require_binding(**values: object) -> None:
        candidate = values.get("checkpoint")
        observed.append(candidate if isinstance(candidate, Path) else None)

    monkeypatch.setattr(
        hf_transformers, "require_loaded_hf_checkpoint_binding", require_binding
    )
    monkeypatch.setattr(
        HFTransformersCausalScorer,
        "__call__",
        lambda self, batch, settings: _observation(
            batch, settings, artifact_sha256=self.artifact_identity_sha256
        ),
    )
    spec = _spec()
    model = object()
    scorer = HFTransformersCausalScorer(
        model=model,
        tokenizer=_BindingTokenizer(),
        artifact_identity_sha256=_artifact_sha256(spec),
        checkpoint_path=checkpoint,
    )
    session = HFTransformersProvider().open(
        spec,
        RuntimeExecutionContext(
            strict=True,
            allow_network=False,
            container_image_digest=_IMAGE_DIGEST,
            device_kind="cpu",
            artifact_identity_sha256=_artifact_sha256(spec),
            provider_state=model,
            scorer=scorer,
        ),
    )

    session.score(_batch())

    assert observed == [checkpoint, checkpoint]
