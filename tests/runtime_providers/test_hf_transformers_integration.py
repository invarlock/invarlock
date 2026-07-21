from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import pytest

from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.core.runtime_provider import (
    EvaluationBatch,
    EvaluationRecord,
    HFSnapshotArtifactIdentity,
    ModelRuntimeSpec,
    RuntimeArtifactResources,
    RuntimeExecutionContext,
    artifact_identity_sha256,
)
from invarlock.core.runtime_provider.behavioral_observation import (
    verify_runtime_behavioral_observation,
)
from invarlock.runtime_provider_evidence import encode_scoring_observation
from invarlock.runtime_providers import hf_transformers
from invarlock.runtime_providers.hf_transformers import (
    HFTransformersCausalScorer,
    HFTransformersProvider,
    hf_tokenizer_contract_sha256,
)
from tests.runtime_providers._hf_transformers_helpers import (
    _IMAGE_DIGEST,
    _REAL_BACKEND_IDENTITY,
    _REAL_DEVICE_FACTS,
    _REAL_STRICT_EXECUTION_BINDING,
    _artifact_sha256,
    _authenticated_test_runtime,  # noqa: F401
    _BindingTokenizer,
)


def test_hf_provider_receipts_a_real_tiny_local_transformers_journey(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    monkeypatch.setattr(
        hf_transformers, "_installed_backend_identity", _REAL_BACKEND_IDENTITY
    )
    monkeypatch.setattr(hf_transformers, "_observed_device_facts", _REAL_DEVICE_FACTS)
    monkeypatch.setattr(
        hf_transformers,
        "_require_strict_execution_binding",
        _REAL_STRICT_EXECUTION_BINDING,
    )
    torch.manual_seed(17)
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=32,
            n_positions=8,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    model.eval()
    checkpoint = tmp_path / "tiny-local-hf"
    model.save_pretrained(checkpoint, safe_serialization=True)
    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        **{f"token-{index}": index + 4 for index in range(27)},
        "target": 31,
    }
    tokenizer_backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab, unk_token="<unk>")
    )
    tokenizer_backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(checkpoint)
    tree_sha256 = checkpoint_tree_sha256(checkpoint)
    tokenizer_sha256 = hf_tokenizer_contract_sha256(tokenizer)
    spec = ModelRuntimeSpec(
        provider_name="hf_transformers",
        model_id=str(checkpoint),
        settings={
            "batch_size": 1,
            "checkpoint_tree_sha256": tree_sha256,
            "context_length": 8,
            "max_output_tokens": 1,
            "offline": True,
            "seed": 17,
            "timeout_seconds": 30,
            "tokenizer_metadata_sha256": tokenizer_sha256,
        },
    )
    provider = HFTransformersProvider()
    identity = provider.identify_artifact(spec)
    input_text = "token-1 token-2 token-3"
    batch = EvaluationBatch(
        schedule_sha256=hashlib.sha256(b"tiny-local-hf-schedule").hexdigest(),
        records=(
            EvaluationRecord(
                record_id="tiny-1",
                input_text=input_text,
                input_sha256=hashlib.sha256(input_text.encode("utf-8")).hexdigest(),
                # This expectation is fixed from the deterministic seed/model
                # contract before the provider runs. Do not derive it from the
                # observation under test.
                expected_output="token-13",
            ),
        ),
    )

    context = provider.prepare_execution(
        spec,
        RuntimeArtifactResources(
            root=tmp_path,
            primary_artifact=checkpoint.name,
            support_resources={},
            device_kind="cpu",
            container_image_digest=_IMAGE_DIGEST,
        ),
    )
    session = provider.open(spec, context)
    observation = session.score(batch)
    receipt = session.runtime_receipt()
    verified = verify_runtime_behavioral_observation(
        json.loads(encode_scoring_observation(observation)),
        expected_provider_name="hf_transformers",
        expected_artifact_identity_sha256=artifact_identity_sha256(identity),
        expected_batch=batch,
        metric="exact_match",
    )

    assert observation.records[0].status == "ok"
    assert observation.records[0].output_text == "token-13"
    assert verified.total_records == 1
    assert verified.value == 1.0
    assert receipt.backend.name == "transformers+torch"
    assert receipt.backend.version == (
        f"transformers={importlib.metadata.version('transformers')};"
        f"torch={importlib.metadata.version('torch')}"
    )
    assert receipt.backend.source_sha256 is not None
    assert receipt.backend.binary_sha256 is not None
    assert receipt.backend.build_sha256 is not None
    assert receipt.device.device_kind == "cpu"
    assert isinstance(receipt.artifact_identity, HFSnapshotArtifactIdentity)
    assert receipt.artifact_identity.tokenizer_metadata_sha256 == tokenizer_sha256
    assert receipt.artifact_identity.checkpoint_tree_sha256 == tree_sha256.removeprefix(
        "sha256:"
    )
    assert str(tmp_path) not in repr(receipt)

    nll_batch = EvaluationBatch(
        schedule_sha256=hashlib.sha256(b"tiny-local-hf-nll-schedule").hexdigest(),
        records=(
            EvaluationRecord(
                record_id="tiny-nll-1",
                input_text=input_text,
                input_sha256=hashlib.sha256(input_text.encode("utf-8")).hexdigest(),
                expected_output=" target",
            ),
        ),
        metric="normalized_nll_per_utf8_byte",
    )
    nll_observation = session.score(nll_batch)
    nll_receipt = session.runtime_receipt()
    nll_verified = verify_runtime_behavioral_observation(
        json.loads(encode_scoring_observation(nll_observation)),
        expected_provider_name="hf_transformers",
        expected_artifact_identity_sha256=artifact_identity_sha256(identity),
        expected_batch=nll_batch,
        metric="normalized_nll_per_utf8_byte",
    )
    nll_record = nll_observation.records[0]
    assert nll_record.output_text is None
    assert nll_record.logprob_sum is not None and nll_record.logprob_sum < 0
    assert nll_record.token_count == 1
    assert nll_record.utf8_byte_count == len(b" target")
    assert nll_verified.value > 0
    assert (
        nll_receipt.scoring_observation_sha256
        == hashlib.sha256(encode_scoring_observation(nll_observation)).hexdigest()
    )

    prepared_model = context.provider_state
    assert prepared_model is not None
    with torch.no_grad():
        next(prepared_model.parameters()).add_(1)
    with pytest.raises(ValueError, match="tensors do not match"):
        session.score(batch)
    with pytest.raises(RuntimeError, match="unavailable before scoring"):
        session.runtime_receipt()


def test_hf_strict_open_rejects_loaded_weights_from_another_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    monkeypatch.setattr(
        hf_transformers,
        "_require_strict_execution_binding",
        _REAL_STRICT_EXECUTION_BINDING,
    )
    monkeypatch.setattr(
        hf_transformers,
        "_require_model_config_match",
        lambda checkpoint, *, model: None,
    )
    torch.manual_seed(11)
    authenticated_model = torch.nn.Linear(3, 2)
    torch.manual_seed(29)
    unrelated_model = torch.nn.Linear(3, 2)
    authenticated_model.eval()
    unrelated_model.eval()
    checkpoint = tmp_path / "authenticated-hf"
    checkpoint.mkdir()
    safetensors_torch.save_file(
        {
            key: value.detach().cpu().contiguous()
            for key, value in authenticated_model.state_dict().items()
        },
        checkpoint / "model.safetensors",
    )
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
        model=unrelated_model,
        tokenizer=tokenizer,
        artifact_identity_sha256=identity_sha256,
    )

    with pytest.raises(ValueError, match="tensors do not match"):
        HFTransformersProvider().open(
            spec,
            RuntimeExecutionContext(
                strict=True,
                allow_network=False,
                container_image_digest=_IMAGE_DIGEST,
                device_kind="cpu",
                artifact_identity_sha256=identity_sha256,
                provider_state=unrelated_model,
                scorer=scorer,
            ),
        )


def test_hf_strict_loader_rejects_real_partial_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    config = transformers.GPT2Config(
        vocab_size=32,
        n_positions=8,
        n_embd=8,
        n_layer=1,
        n_head=1,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = transformers.GPT2LMHeadModel(config)
    checkpoint = tmp_path / "partial-hf"
    checkpoint.mkdir()
    config.save_pretrained(checkpoint)
    safetensors_torch.save_file(
        {"transformer.wte.weight": model.state_dict()["transformer.wte.weight"]},
        checkpoint / "model.safetensors",
        metadata={"format": "pt"},
    )

    with pytest.raises(ValueError, match="loading reported missing"):
        hf_transformers.load_hf_model_with_strict_loading_info(
            transformers.AutoModelForCausalLM.from_pretrained,
            checkpoint,
        )
