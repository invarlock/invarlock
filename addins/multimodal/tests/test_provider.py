from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import invarlock_addins.multimodal.provider as provider_module
import pytest
from invarlock_addins.multimodal.provider import (
    HFVisionTextProvider,
    HFVisionTextScorer,
    _read_content_bytes,
)

from invarlock.core.runtime_provider import (
    EvaluationBatch,
    EvaluationInputPart,
    EvaluationRecord,
    ModelRuntimeSpec,
    RuntimeExecutionSettings,
    evaluation_input_parts_sha256,
)
from invarlock.core.runtime_provider.behavioral_observation import (
    verify_runtime_behavioral_observation,
)
from invarlock.runtime_provider_evidence import encode_scoring_observation

_DIGEST = "a" * 64


def _spec(**overrides: object) -> ModelRuntimeSpec:
    settings: dict[str, object] = {
        "batch_size": 1,
        "checkpoint_tree_sha256": "b" * 64,
        "context_length": 512,
        "max_output_tokens": 32,
        "offline": True,
        "processor_metadata_sha256": "c" * 64,
        "seed": 7,
        "timeout_seconds": 30,
        "tokenizer_metadata_sha256": "d" * 64,
    }
    settings.update(overrides)
    return ModelRuntimeSpec(
        provider_name="hf_vision_text",
        model_id="org/vision-text-model",
        settings=settings,  # type: ignore[arg-type]
    )


def test_provider_declares_only_implemented_behavior() -> None:
    provider = HFVisionTextProvider()

    assert provider.capabilities().tasks == ("vision_text_generation",)
    assert provider.capabilities().metrics == ("exact_match",)
    assert provider.capabilities().execution_modes == ("container",)
    assert provider.identify_artifact(_spec()).checkpoint_tree_sha256 == "b" * 64


def test_cuda_device_facts_allow_torch_without_private_driver_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = SimpleNamespace(type="cuda", index=0)
    model = SimpleNamespace(
        parameters=lambda: iter((SimpleNamespace(device=device),)),
        buffers=lambda: iter(()),
    )
    torch = SimpleNamespace(
        _C=SimpleNamespace(),
        cuda=SimpleNamespace(
            get_device_name=lambda _index: "Test GPU",
            get_device_capability=lambda _index: (9, 0),
        ),
        version=SimpleNamespace(cuda="12.8"),
    )
    real_import = provider_module.importlib.import_module
    monkeypatch.setattr(
        provider_module.importlib,
        "import_module",
        lambda name: torch if name == "torch" else real_import(name),
    )

    facts = provider_module._device_facts(model, expected_kind="cuda")

    assert facts.device_name == "Test GPU"
    assert facts.compute_capability == "9.0"
    assert facts.driver_version is None
    assert facts.cuda_runtime_version == "12.8"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"batch_size": 2}, "batch_size=1"),
        ({"offline": False}, "offline=true"),
        ({"processor_metadata_sha256": "bad"}, "sha256 digest"),
        ({"unexpected": True}, "unsupported"),
    ],
)
def test_config_rejects_unimplemented_or_ambiguous_behavior(
    change: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        HFVisionTextProvider().validate_config(_spec(**change))


def test_content_store_rechecks_length_and_digest(tmp_path: Path) -> None:
    payload = b"authenticated image bytes"
    tmp_path.joinpath("image_001").write_bytes(payload)

    assert (
        _read_content_bytes(
            tmp_path,
            content_id="image_001",
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_byte_length=len(payload),
        )
        == payload
    )
    with pytest.raises(ValueError, match="identity does not match"):
        _read_content_bytes(
            tmp_path,
            content_id="image_001",
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_byte_length=len(payload) + 1,
        )
    with pytest.raises(ValueError, match="digest does not match"):
        _read_content_bytes(
            tmp_path,
            content_id="image_001",
            expected_sha256=_DIGEST,
            expected_byte_length=len(payload),
        )


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_content_store_rejects_links_and_path_syntax(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-image"
    outside.write_bytes(b"outside")
    tmp_path.joinpath("linked_image").symlink_to(outside)

    with pytest.raises(ValueError, match="unavailable"):
        _read_content_bytes(
            tmp_path,
            content_id="linked_image",
            expected_sha256=hashlib.sha256(b"outside").hexdigest(),
            expected_byte_length=len(b"outside"),
        )


def test_scorer_executes_authenticated_vision_text_exact_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    image_bytes = b"test image bytes"
    tmp_path.joinpath("image_001").write_bytes(image_bytes)
    prompt = "What animal is shown?"
    parts = (
        EvaluationInputPart(
            kind="content",
            role="image",
            content_id="image_001",
            media_type="image/png",
            byte_length=len(image_bytes),
            sha256=hashlib.sha256(image_bytes).hexdigest(),
        ),
        EvaluationInputPart(
            kind="text",
            role="prompt",
            text=prompt,
            sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        ),
    )
    record = EvaluationRecord(
        record_id="record_001",
        input_text=prompt,
        input_sha256=evaluation_input_parts_sha256(parts),
        expected_output="cat",
        input_parts=parts,
    )

    class _Image:
        closed = False

        def close(self) -> None:
            self.closed = True

    image = _Image()
    monkeypatch.setattr(
        provider_module, "_decode_image", lambda *_args, **_kwargs: image
    )

    class _Processor:
        def apply_chat_template(self, messages, **kwargs):  # noqa: ANN001
            assert messages[0]["content"][1]["text"] == prompt
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return "<image>What animal is shown?"

        def __call__(self, **kwargs):  # noqa: ANN003
            assert kwargs["images"] is image
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "pixel_values": torch.ones((1, 3, 2, 2)),
            }

        def batch_decode(self, sequences, **kwargs):  # noqa: ANN001
            assert sequences.tolist() == [[9]]
            assert kwargs["clean_up_tokenization_spaces"] is False
            return ["cat"]

    class _Module:
        training = False

    class _Model:
        def parameters(self):
            return iter((torch.ones(1),))

        def buffers(self):
            return iter(())

        def modules(self):
            return iter((_Module(),))

        def generate(self, **kwargs):  # noqa: ANN003
            assert kwargs["do_sample"] is False
            assert kwargs["max_new_tokens"] == 4
            return torch.tensor([[1, 2, 3, 9]])

    scorer = HFVisionTextScorer(
        model=_Model(),
        processor=_Processor(),
        content_store=tmp_path,
        artifact_identity_sha256="e" * 64,
    )
    batch = EvaluationBatch(
        schedule_sha256="f" * 64,
        records=(record,),
        metric="exact_match",
        task="vision_text_generation",
    )
    settings = RuntimeExecutionSettings(
        seed=1,
        context_length=128,
        batch_size=1,
        max_output_tokens=4,
        timeout_seconds=10,
    )

    observation = scorer(batch, settings)

    assert image.closed is True
    assert observation.records[0].output_text == "cat"
    result = verify_runtime_behavioral_observation(
        json.loads(encode_scoring_observation(observation)),
        expected_provider_name="hf_vision_text",
        expected_artifact_identity_sha256="e" * 64,
        expected_batch=batch,
        metric="exact_match",
    )
    assert result.metric == "exact_match"
    assert result.value == 1.0
    with pytest.raises(ValueError, match="safe basename"):
        _read_content_bytes(
            tmp_path,
            content_id="../outside-image",
            expected_sha256=hashlib.sha256(b"outside").hexdigest(),
            expected_byte_length=len(b"outside"),
        )
