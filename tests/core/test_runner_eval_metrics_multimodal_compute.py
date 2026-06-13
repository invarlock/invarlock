from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import invarlock.core.runner_eval_metrics as rem
from invarlock.adapters.hf_multimodal import HF_Multimodal_Adapter
from tests.core._support_runner_eval_metrics import (
    _FakeModel,
    _FakeRunner,
    _write_ppm,
)


def test_compute_real_metrics_supports_vision_text_classification() -> None:
    class _VisionModel(_FakeModel):
        def __call__(self, **kwargs):
            labels = kwargs["labels"]
            loss = torch.tensor(0.25 if int((labels != -100).sum().item()) > 0 else 0.0)
            return SimpleNamespace(loss=loss)

        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3]], dtype=torch.long)

    class _VisionAdapter:
        def prepare_model_inputs(self, batch, device, include_labels):  # noqa: ANN001
            payload = {
                "input_ids": torch.tensor([[1, 2]], device=device, dtype=torch.long),
                "attention_mask": torch.tensor(
                    [[1, 1]], device=device, dtype=torch.long
                ),
                "_example_id": batch["id"],
                "_reference_answers": list(batch["answers"]),
                "_processor_sha256": "proc-123",
                "_prediction": batch["prediction"],
                "_answer_token_count": 1,
            }
            if include_labels:
                payload["labels"] = torch.tensor(
                    [[-100, 7]], device=device, dtype=torch.long
                )
            return payload

        def prepare_generation_inputs(self, batch, device):  # noqa: ANN001
            payload = self.prepare_model_inputs(batch, device, include_labels=False)
            payload["_max_new_tokens"] = 8
            return payload

        def decode_generated(self, generated_ids, prepared_batch):  # noqa: ANN001
            return [prepared_batch["_prediction"]]

    runner = _FakeRunner()
    config = SimpleNamespace(
        context={
            "eval": {
                "loss": {"type": "classification", "resolved_type": "classification"},
                "metric": {"kind": "accuracy"},
            }
        }
    )

    metrics, eval_windows = rem.compute_real_metrics(
        runner,
        _VisionModel(),
        calibration_data=[
            {
                "id": "ex-1",
                "image_path": "/tmp/a.png",
                "answers": ["cat"],
                "prediction": "cat",
            },
            {
                "id": "ex-2",
                "image_path": "/tmp/b.png",
                "answers": ["dog"],
                "prediction": "cat",
            },
        ],
        adapter=_VisionAdapter(),
        preview_n=1,
        final_n=1,
        config=config,
    )

    assert metrics["primary_metric"]["kind"] == "accuracy"
    assert metrics["classification"]["preview"]["correct_total"] == 1
    assert metrics["classification"]["final"]["correct_total"] == 0
    assert metrics["classification"]["counts_source"] == "measured"
    assert metrics["paired_windows"] == 1
    assert metrics["window_match_fraction"] == 1.0
    assert metrics["window_pairing_reason"] is None
    assert metrics["window_pairing_preview"] == {
        "matched": 1,
        "expected": 1,
        "reason": None,
    }
    assert eval_windows["preview"]["example_ids"] == ["ex-1"]
    assert eval_windows["final"]["records"][0]["correct"] is False
    assert eval_windows["preview"]["input_records"][0]["image_path"] == "/tmp/a.png"
    assert eval_windows["preview"]["input_records"][0]["answers"] == ["cat"]
    assert "prediction" not in eval_windows["preview"]["input_records"][0]
    assert metrics["primary_metric"]["preview"] == 1.0
    assert metrics["primary_metric"]["final"] == 0.0


def test_multimodal_metric_kind_rejects_unknown_config_value() -> None:
    config = SimpleNamespace(context={"eval": {"metric": {"kind": "vqa_accuracy"}}})

    with pytest.raises(ValueError, match="Unsupported metric kind"):
        rem._resolve_metric_kind(config, fallback="accuracy")


def test_compute_real_metrics_retries_multimodal_processor_without_truncation(
    tmp_path: Path,
) -> None:
    class _RetryingProcessor:
        def __init__(self) -> None:
            self.name_or_path = "retrying-processor"
            self.tokenizer = SimpleNamespace(
                name_or_path="retrying-tokenizer",
                vocab_size=32,
                eos_token="</s>",
                pad_token="<pad>",
            )
            self.image_processor = SimpleNamespace(
                size={"height": 1, "width": 1},
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.25, 0.25, 0.25],
            )
            self.calls: list[tuple[str, bool, int | None]] = []

        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=False
        ):  # noqa: ANN001
            del tokenize
            prompt = messages[0]["content"][1]["text"]
            if len(messages) > 1:
                answer = messages[1]["content"][0]["text"]
                return f"USER:{prompt}\nASSISTANT:{answer}"
            suffix = "\nASSISTANT:" if add_generation_prompt else ""
            return f"USER:{prompt}{suffix}"

        def __call__(self, *, text, images, return_tensors, truncation, max_length):  # noqa: ANN001
            del images, return_tensors
            self.calls.append((text, bool(truncation), max_length))
            if truncation:
                raise ValueError(
                    "Mismatch in `image` token count between text and `input_ids`. "
                    "Likely due to `truncation='max_length'`."
                )
            if "ASSISTANT:cat" in text:
                input_ids = torch.tensor([[11, 12, 13, 14]], dtype=torch.long)
            else:
                input_ids = torch.tensor([[11, 12]], dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }

        def batch_decode(self, ids, skip_special_tokens=True):  # noqa: ANN001
            del skip_special_tokens
            rows = ids.tolist()
            if rows == [[101]]:
                return ["cat"]
            if rows == [[102]]:
                return ["not-cat"]
            return [" ".join(str(token) for token in row) for row in rows]

    class _VisionModel(_FakeModel):
        def __init__(self) -> None:
            super().__init__()
            self.generate_calls = 0

        def __call__(self, **kwargs):
            del kwargs
            return SimpleNamespace(loss=torch.tensor(0.25))

        def generate(self, **kwargs):
            del kwargs
            self.generate_calls += 1
            token = 101 if self.generate_calls == 1 else 102
            return torch.tensor([[1, 2, token]], dtype=torch.long)

    image_path = tmp_path / "demo.ppm"
    _write_ppm(image_path)

    adapter = HF_Multimodal_Adapter()
    processor = _RetryingProcessor()
    adapter._processor = processor
    adapter._processor_digest = adapter._compute_processor_digest(processor)
    adapter._last_model_id = "fake/model"

    metrics, eval_windows = rem.compute_real_metrics(
        _FakeRunner(),
        _VisionModel(),
        calibration_data=[
            {
                "id": "ex-1",
                "example_id": "ex-1",
                "image_path": str(image_path),
                "prompt": "what is shown?",
                "answers": ["cat"],
                "seq_len": 8,
            },
            {
                "id": "ex-2",
                "example_id": "ex-2",
                "image_path": str(image_path),
                "prompt": "what is shown?",
                "answers": ["cat"],
                "seq_len": 8,
            },
        ],
        adapter=adapter,
        preview_n=1,
        final_n=1,
        config=SimpleNamespace(
            context={
                "eval": {
                    "loss": {
                        "type": "classification",
                        "resolved_type": "classification",
                    },
                    "metric": {"kind": "accuracy"},
                }
            }
        ),
    )

    assert metrics["classification"]["counts_source"] == "measured"
    assert metrics["classification"]["n_correct"] == 0
    assert metrics["classification"]["n_total"] == 1
    assert metrics["classification"]["estimated"] is False
    assert eval_windows["preview"]["records"][0]["correct"] is True
    assert eval_windows["final"]["records"][0]["correct"] is False
    assert eval_windows["preview"]["input_records"][0]["image_path"] == str(image_path)
    assert eval_windows["preview"]["input_records"][0]["prompt"] == "what is shown?"
    assert eval_windows["preview"]["input_records"][0]["answers"] == ["cat"]
    assert any(
        truncation is False and max_length is None
        for _text, truncation, max_length in processor.calls
    )


def test_evaluate_vision_text_arm_requires_real_hook_surface() -> None:
    with pytest.raises(RuntimeError, match="prepare_model_inputs"):
        rem._evaluate_vision_text_arm(
            _FakeModel(),
            [{"id": "ex-1", "image_path": "/tmp/a.png", "answers": ["cat"]}],
            adapter=object(),
            device="cpu",
        )

    class _MissingDecode:
        def prepare_model_inputs(self, batch, device, include_labels):  # noqa: ANN001
            return {
                "input_ids": torch.tensor([[1]], device=device),
                "labels": torch.tensor([[1]], device=device),
            }

        def prepare_generation_inputs(self, batch, device):  # noqa: ANN001
            return {"input_ids": torch.tensor([[1]], device=device)}

    with pytest.raises(RuntimeError, match="decode_generated"):
        rem._evaluate_vision_text_arm(
            _FakeModel(),
            [{"id": "ex-1", "image_path": "/tmp/a.png", "answers": ["cat"]}],
            adapter=_MissingDecode(),
            device="cpu",
        )

    class _MissingGeneration:
        def prepare_model_inputs(self, batch, device, include_labels):  # noqa: ANN001
            return {
                "input_ids": torch.tensor([[1]], device=device),
                "labels": torch.tensor([[1]], device=device),
            }

        def decode_generated(self, generated_ids, generation_inputs):  # noqa: ANN001
            return ["cat"]

    with pytest.raises(RuntimeError, match="prepare_generation_inputs"):
        rem._evaluate_vision_text_arm(
            _FakeModel(),
            [{"id": "ex-1", "image_path": "/tmp/a.png", "answers": ["cat"]}],
            adapter=_MissingGeneration(),
            device="cpu",
        )


def test_evaluate_vision_text_arm_skips_zero_answer_tokens_and_blank_processor_sha() -> (
    None
):
    class _VisionModel:
        def __call__(self, **_kwargs):
            return SimpleNamespace(loss=torch.tensor(0.5))

        def generate(self, **_kwargs):
            return torch.tensor([[1, 2]])

    class _Adapter:
        def prepare_model_inputs(self, batch, device, include_labels):  # noqa: ANN001
            return {
                "input_ids": torch.tensor([[1]], device=device),
                "labels": torch.tensor([[1]], device=device),
                "_answer_token_count": 0,
            }

        def prepare_generation_inputs(self, batch, device):  # noqa: ANN001
            return {
                "input_ids": torch.tensor([[1]], device=device),
                "_reference_answers": ["   "],
                "_processor_sha256": 123,
            }

        def decode_generated(self, generated_ids, generation_inputs):  # noqa: ANN001
            return ["cat"]

    payload, latency_ms = rem._evaluate_vision_text_arm(
        _VisionModel(),
        [
            {
                "id": "batch-id",
                "image_path": "/tmp/demo.png",
                "answers": ["cat"],
                "image_sha256": "img",
                "prompt_sha256": "prompt",
                "answer_sha256": "answer",
            }
        ],
        adapter=_Adapter(),
        device="cpu",
    )

    assert latency_ms >= 0.0
    assert payload["total_tokens"] == 0
    assert payload["logloss"] == []
    assert payload["token_counts"] == []
    assert payload["records"][0]["id"] == "batch-id"
    assert payload["records"][0]["references"] == []
    assert payload["records"][0]["correct"] is False
    assert "processor_sha256" not in payload


def test_evaluate_vision_text_arm_empty_batches_omit_input_records() -> None:
    class _Adapter:
        def prepare_model_inputs(self, batch, device, include_labels):  # noqa: ANN001
            raise AssertionError("empty arm should not prepare inputs")

        def prepare_generation_inputs(self, batch, device):  # noqa: ANN001
            raise AssertionError("empty arm should not prepare generation")

        def decode_generated(self, generated_ids, generation_inputs):  # noqa: ANN001
            raise AssertionError("empty arm should not decode")

    payload, latency_ms = rem._evaluate_vision_text_arm(
        _FakeModel(),
        [],
        adapter=_Adapter(),
        device="cpu",
    )

    assert latency_ms == 0.0
    assert payload["total"] == 0
    assert math.isnan(payload["accuracy"])
    assert math.isnan(payload["mean_logloss"])
    assert "input_records" not in payload


def test_evaluate_vision_text_arm_keeps_first_processor_sha_across_multiple_batches() -> (
    None
):
    class _VisionModel:
        def __call__(self, **_kwargs):
            return SimpleNamespace(loss=torch.tensor(0.25))

        def generate(self, **_kwargs):
            return torch.tensor([[1, 2]])

    class _Adapter:
        def prepare_model_inputs(self, batch, device, include_labels):  # noqa: ANN001
            del include_labels
            return {
                "input_ids": torch.tensor([[1]], device=device),
                "labels": torch.tensor([[1]], device=device),
                "_answer_token_count": 1,
            }

        def prepare_generation_inputs(self, batch, device):  # noqa: ANN001
            return {
                "input_ids": torch.tensor([[1]], device=device),
                "_reference_answers": [batch["answer"]],
                "_processor_sha256": batch["processor_sha"],
                "_example_id": batch["id"],
            }

        def decode_generated(self, generated_ids, generation_inputs):  # noqa: ANN001
            del generated_ids, generation_inputs
            return ["cat"]

    payload, _latency_ms = rem._evaluate_vision_text_arm(
        _VisionModel(),
        [
            {"id": "ex-1", "answer": "cat", "processor_sha": "sha-a"},
            {"id": "ex-2", "answer": "cat", "processor_sha": "sha-b"},
        ],
        adapter=_Adapter(),
        device="cpu",
    )

    assert payload["processor_sha256"] == "sha-a"
    assert payload["example_ids"] == ["ex-1", "ex-2"]


def test_evaluate_vision_text_arm_treats_string_outputs_and_references_as_single_values() -> (
    None
):
    class _VisionModel:
        def __call__(self, **_kwargs):
            return SimpleNamespace(loss=torch.tensor(0.5))

        def generate(self, **_kwargs):
            return torch.tensor([[1, 2]])

    class _Adapter:
        def prepare_model_inputs(self, batch, device, include_labels):  # noqa: ANN001
            return {
                "input_ids": torch.tensor([[1]], device=device),
                "labels": torch.tensor([[1]], device=device),
                "_answer_token_count": 1,
            }

        def prepare_generation_inputs(self, batch, device):  # noqa: ANN001
            return {
                "input_ids": torch.tensor([[1]], device=device),
                "_reference_answers": " Cat ",
            }

        def decode_generated(self, generated_ids, generation_inputs):  # noqa: ANN001
            return "Cat"

    payload, _latency_ms = rem._evaluate_vision_text_arm(
        _VisionModel(),
        [{"id": "example-1", "image_path": "/tmp/demo.png", "answers": ["cat"]}],
        adapter=_Adapter(),
        device="cpu",
    )

    assert payload["records"][0]["prediction"] == "Cat"
    assert payload["records"][0]["references"] == ["Cat"]
    assert payload["records"][0]["correct"] is True
