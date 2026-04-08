from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

import invarlock.core.runner_eval_metrics as rem
import invarlock.core.runner_eval_metrics_multimodal as remm
from invarlock.adapters.hf_multimodal import HF_Multimodal_Adapter


class _FakeRunner:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, str, dict[str, Any]]] = []

    def _resolve_policy_flags(self, _config: Any) -> dict[str, bool]:
        return {"allow_calibration_materialize": True}

    def _log_event(
        self,
        component: str,
        operation: str,
        level: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.events.append((component, operation, level, data or {}))


class _FakeModel:
    def __init__(self) -> None:
        self._device = "cpu"
        self._param = SimpleNamespace(device="cpu")

    def eval(self) -> None:
        return None

    def parameters(self):
        yield self._param

    def to(self, device) -> _FakeModel:
        self._device = str(device)
        self._param = SimpleNamespace(device=device)
        return self


def _write_ppm(path: Path) -> None:
    path.write_text("P3\n1 1\n255\n255 0 0\n", encoding="utf-8")


def _summary(
    *,
    ppl: float,
    total_tokens: int,
    weighted_log_loss: float,
    num_batches: int,
    log_losses: list[float],
    window_ids: list[int],
    token_counts: list[int] | None = None,
    actual_token_counts: list[int] | None = None,
) -> dict[str, Any]:
    token_counts = token_counts or []
    actual_token_counts = actual_token_counts or token_counts or [1] * num_batches
    return {
        "ppl": ppl,
        "total_tokens": total_tokens,
        "actual_total_tokens": sum(actual_token_counts),
        "num_batches": num_batches,
        "log_losses": log_losses,
        "window_ids": window_ids,
        "tokens": [[wid + 1] for wid in window_ids],
        "attention_masks": [[1] for _ in window_ids],
        "weighted_log_loss": weighted_log_loss,
        "window_token_counts": token_counts,
        "masked_token_counts": token_counts or [1] * num_batches,
        "actual_token_counts": actual_token_counts,
        "labels": [[wid + 1] for wid in window_ids],
    }


def test_compute_real_metrics_uses_logloss_lists_final_weights_and_seq2seq(
    monkeypatch,
) -> None:
    preview_summary = _summary(
        ppl=2.0,
        total_tokens=0,
        weighted_log_loss=0.0,
        num_batches=1,
        log_losses=[math.log(2.0)],
        window_ids=[0],
        token_counts=[],
        actual_token_counts=[1],
    )
    final_summary = _summary(
        ppl=4.0,
        total_tokens=0,
        weighted_log_loss=0.0,
        num_batches=1,
        log_losses=[math.log(4.0)],
        window_ids=[1],
        token_counts=[3],
        actual_token_counts=[3],
    )

    def fake_slice_summary(*_args, start_idx: int, **_kwargs):
        return (preview_summary, None) if start_idx == 0 else (final_summary, None)

    monkeypatch.setattr(rem, "compute_slice_summary", fake_slice_summary)
    monkeypatch.setattr(rem, "measure_latency", lambda *_args, **_kwargs: 0.25)
    monkeypatch.setattr(
        rem,
        "compute_window_pairing_metrics",
        lambda **_kwargs: {
            "preview": {"matched": 1, "expected": 1, "reason": None},
            "final": {"matched": 1, "expected": 1, "reason": None},
            "match_fraction": 1.0,
            "overlap_fraction": 0.0,
            "duplicate_fraction": 0.0,
            "count_mismatch": False,
            "reason": None,
        },
    )
    monkeypatch.setattr(
        rem,
        "assess_bootstrap_coverage",
        lambda **_kwargs: {
            "preview_required": 4,
            "final_required": 4,
            "replicates_required": 1000,
            "preview_ok": False,
            "final_ok": True,
            "replicates_ok": True,
            "coverage": {
                "preview": {"used": 1, "required": 4, "ok": False},
                "final": {"used": 1, "required": 4, "ok": True},
                "replicates": {"used": 0, "required": 1000, "ok": True},
            },
        },
    )

    runner = _FakeRunner()
    config = SimpleNamespace(
        context={
            "eval": {
                "loss": {"type": "seq2seq"},
                "bootstrap": {"enabled": False},
            },
            "auto": {"tier": "balanced"},
        }
    )

    metrics, _eval_windows = rem.compute_real_metrics(
        runner,
        _FakeModel(),
        calibration_data=[{"input_ids": [1]}, {"input_ids": [2]}],
        adapter=object(),
        preview_n=1,
        final_n=1,
        config=config,
    )

    assert metrics["primary_metric"]["kind"] == "ppl_seq2seq"
    assert (
        metrics["paired_delta_summary"]["mean_preview_weighted"]
        == metrics["paired_delta_summary"]["mean"]
    )
    assert any(
        operation == "bootstrap_coverage_warning"
        for _, operation, _, _ in runner.events
    )


def test_compute_real_metrics_marks_primary_metric_invalid_on_ratio_ci_inconsistency(
    monkeypatch,
) -> None:
    preview_summary = _summary(
        ppl=2.0,
        total_tokens=2,
        weighted_log_loss=2 * math.log(2.0),
        num_batches=2,
        log_losses=[math.log(2.0), math.log(2.2)],
        window_ids=[0, 1],
        token_counts=[1, 1],
        actual_token_counts=[1, 1],
    )
    final_summary = _summary(
        ppl=3.0,
        total_tokens=2,
        weighted_log_loss=2 * math.log(3.0),
        num_batches=2,
        log_losses=[math.log(3.0), math.log(3.5)],
        window_ids=[2, 3],
        token_counts=[1, 1],
        actual_token_counts=[1, 1],
    )

    def fake_slice_summary(*_args, start_idx: int, **_kwargs):
        return (preview_summary, None) if start_idx == 0 else (final_summary, None)

    monkeypatch.setattr(rem, "compute_slice_summary", fake_slice_summary)
    monkeypatch.setattr(rem, "measure_latency", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(
        rem,
        "compute_window_pairing_metrics",
        lambda **_kwargs: {
            "preview": {"matched": 2, "expected": 2, "reason": None},
            "final": {"matched": 2, "expected": 2, "reason": None},
            "match_fraction": 1.0,
            "overlap_fraction": 0.0,
            "duplicate_fraction": 0.0,
            "count_mismatch": False,
            "reason": None,
        },
    )
    monkeypatch.setattr(
        rem,
        "assess_bootstrap_coverage",
        lambda **_kwargs: {
            "preview_required": 1,
            "final_required": 1,
            "replicates_required": 1,
            "preview_ok": True,
            "final_ok": True,
            "replicates_ok": True,
            "coverage": {},
        },
    )

    metrics, _eval_windows = rem.compute_real_metrics(
        _FakeRunner(),
        _FakeModel(),
        calibration_data=[{"input_ids": [1]}] * 4,
        adapter=object(),
        preview_n=2,
        final_n=2,
        config=SimpleNamespace(
            context={"eval": {"bootstrap": {"enabled": True, "replicates": 4}}}
        ),
        compute_paired_delta_log_ci_fn=lambda *_args, **_kwargs: (0.1, 0.2),
        logspace_to_ratio_ci_fn=lambda _delta_ci: (999.0, 999.0),
    )

    assert metrics["primary_metric"]["invalid"] is True
    assert metrics["primary_metric"]["degraded_reason"] == "primary_metric_invalid"


def test_compute_real_metrics_preserves_invalid_non_finite_pm_and_delta(
    monkeypatch,
) -> None:
    preview_summary = _summary(
        ppl=float("nan"),
        total_tokens=1,
        weighted_log_loss=float("inf"),
        num_batches=1,
        log_losses=[0.1],
        window_ids=[0],
        token_counts=[1],
        actual_token_counts=[1],
    )
    final_summary = _summary(
        ppl=float("nan"),
        total_tokens=1,
        weighted_log_loss=float("inf"),
        num_batches=1,
        log_losses=[0.1],
        window_ids=[1],
        token_counts=[1],
        actual_token_counts=[1],
    )

    def fake_slice_summary(*_args, start_idx: int, **_kwargs):
        return (preview_summary, None) if start_idx == 0 else (final_summary, None)

    monkeypatch.setattr(rem, "compute_slice_summary", fake_slice_summary)
    monkeypatch.setattr(rem, "measure_latency", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(
        rem,
        "compute_window_pairing_metrics",
        lambda **_kwargs: {
            "preview": {"matched": 1, "expected": 1, "reason": None},
            "final": {"matched": 1, "expected": 1, "reason": None},
            "match_fraction": 1.0,
            "overlap_fraction": 0.0,
            "duplicate_fraction": 0.0,
            "count_mismatch": False,
            "reason": None,
        },
    )
    monkeypatch.setattr(
        rem,
        "assess_bootstrap_coverage",
        lambda **_kwargs: {
            "preview_required": 1,
            "final_required": 1,
            "replicates_required": 1,
            "preview_ok": True,
            "final_ok": True,
            "replicates_ok": True,
            "coverage": {},
        },
    )

    metrics, _eval_windows = rem.compute_real_metrics(
        _FakeRunner(),
        _FakeModel(),
        calibration_data=[{"input_ids": [1]}, {"input_ids": [2]}],
        adapter=object(),
        preview_n=1,
        final_n=1,
        config=SimpleNamespace(context={"eval": {"bootstrap": {"enabled": False}}}),
    )

    assert metrics["primary_metric"]["invalid"] is True
    assert metrics["primary_metric"]["preview"] is None
    assert metrics["primary_metric"]["final"] is None
    assert metrics["primary_metric"]["degraded_reason"] == "non_finite_pm"
    assert not math.isfinite(metrics["logloss_delta"])


def test_compute_real_metrics_marks_empty_slice_metrics_invalid(monkeypatch) -> None:
    preview_summary = _summary(
        ppl=float("nan"),
        total_tokens=0,
        weighted_log_loss=0.0,
        num_batches=0,
        log_losses=[],
        window_ids=[],
        token_counts=[],
        actual_token_counts=[],
    )
    final_summary = _summary(
        ppl=float("nan"),
        total_tokens=0,
        weighted_log_loss=0.0,
        num_batches=0,
        log_losses=[],
        window_ids=[],
        token_counts=[],
        actual_token_counts=[],
    )

    def fake_slice_summary(*_args, start_idx: int, **_kwargs):
        return (preview_summary, None) if start_idx == 0 else (final_summary, None)

    monkeypatch.setattr(rem, "compute_slice_summary", fake_slice_summary)
    monkeypatch.setattr(rem, "measure_latency", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(
        rem,
        "compute_window_pairing_metrics",
        lambda **_kwargs: {
            "preview": {"matched": 0, "expected": 0, "reason": None},
            "final": {"matched": 0, "expected": 0, "reason": None},
            "match_fraction": 1.0,
            "overlap_fraction": 0.0,
            "duplicate_fraction": 0.0,
            "count_mismatch": False,
            "reason": None,
        },
    )
    monkeypatch.setattr(
        rem,
        "assess_bootstrap_coverage",
        lambda **_kwargs: {
            "preview_required": 1,
            "final_required": 1,
            "replicates_required": 1,
            "preview_ok": True,
            "final_ok": True,
            "replicates_ok": True,
            "coverage": {},
        },
    )

    metrics, _eval_windows = rem.compute_real_metrics(
        _FakeRunner(),
        _FakeModel(),
        calibration_data=[{"input_ids": [1]}, {"input_ids": [2]}],
        adapter=object(),
        preview_n=1,
        final_n=1,
        config=SimpleNamespace(context={"eval": {"bootstrap": {"enabled": False}}}),
    )

    assert metrics["primary_metric"]["invalid"] is True
    assert metrics["primary_metric"]["degraded_reason"] == "non_finite_pm"


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
                "metric": {"kind": "vqa_accuracy"},
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

    assert metrics["primary_metric"]["kind"] == "vqa_accuracy"
    assert metrics["classification"]["preview"]["correct_total"] == 1
    assert metrics["classification"]["final"]["correct_total"] == 0
    assert metrics["classification"]["counts_source"] == "measured"
    assert eval_windows["preview"]["example_ids"] == ["ex-1"]
    assert eval_windows["final"]["records"][0]["correct"] is False
    assert metrics["primary_metric"]["preview"] == 1.0
    assert metrics["primary_metric"]["final"] == 0.0


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
                    "metric": {"kind": "vqa_accuracy"},
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
    assert any(
        truncation is False and max_length is None
        for _text, truncation, max_length in processor.calls
    )


def test_runner_eval_metrics_hook_and_metric_resolution_helpers() -> None:
    class _Adapter:
        def prepare_model_inputs(self):  # noqa: ANN201
            return None

    assert rem._resolve_adapter_hook(None, "prepare_model_inputs") is None
    assert rem._resolve_adapter_hook(Mock(), "prepare_model_inputs") is None
    assert callable(rem._resolve_adapter_hook(_Adapter(), "prepare_model_inputs"))

    fallback = "fallback_metric"
    assert rem._resolve_metric_kind(None, fallback=fallback) == fallback
    assert rem._resolve_metric_kind(SimpleNamespace(context=[]), fallback=fallback) == (
        fallback
    )
    assert (
        rem._resolve_metric_kind(
            SimpleNamespace(context={"eval": []}),
            fallback=fallback,
        )
        == fallback
    )
    assert (
        rem._resolve_metric_kind(
            SimpleNamespace(context={"eval": {"metric": {"kind": "auto"}}}),
            fallback=fallback,
        )
        == fallback
    )


def test_runner_eval_metrics_small_helpers_filter_and_normalize_inputs() -> None:
    prepared = {
        "input_ids": [1, 2, 3],
        "labels": [1, 2, 3],
        "_private": "drop",
        7: "drop",
    }

    assert rem._model_kwargs(prepared) == {
        "input_ids": [1, 2, 3],
        "labels": [1, 2, 3],
    }
    assert remm._decode_prediction_text(None) == ""
    assert remm._decode_prediction_text(7) == "7"
    assert rem._normalize_answer_text("  Cat   Dog\n") == "cat dog"
    assert rem._normalize_answer_text(None) == ""
    assert rem._normalize_answer_text(0) == "0"
    assert remm._normalize_reference_answers(123) == []
    assert (
        rem._resolve_metric_kind(
            SimpleNamespace(context={"eval": {"metric": {}}}),
            fallback="ppl_causal",
        )
        == "ppl_causal"
    )
    assert rem._is_multimodal_batch({"example_id": "ex-1"}) is True
    assert rem._is_multimodal_batch({"input_ids": [1, 2, 3]}) is False


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


def test_compute_real_metrics_rejects_invalid_device_override(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(
            Process=lambda: SimpleNamespace(
                memory_info=lambda: SimpleNamespace(rss=0),
            )
        ),
    )

    with pytest.raises((RuntimeError, ValueError)):
        rem.compute_real_metrics(
            _FakeRunner(),
            _FakeModel(),
            calibration_data=[{"input_ids": [1]}],
            adapter=object(),
            preview_n=1,
            final_n=1,
            config=SimpleNamespace(
                context={"eval": {"device_override": "not-a-real-device"}}
            ),
        )


def test_compute_real_metrics_propagates_pairing_metric_failures(monkeypatch) -> None:
    preview_summary = _summary(
        ppl=2.0,
        total_tokens=1,
        weighted_log_loss=math.log(2.0),
        num_batches=1,
        log_losses=[math.log(2.0)],
        window_ids=[0],
        token_counts=[1],
        actual_token_counts=[1],
    )
    final_summary = _summary(
        ppl=3.0,
        total_tokens=1,
        weighted_log_loss=math.log(3.0),
        num_batches=1,
        log_losses=[math.log(3.0)],
        window_ids=[1],
        token_counts=[1],
        actual_token_counts=[1],
    )

    def fake_slice_summary(*_args, start_idx: int, **_kwargs):
        return (preview_summary, None) if start_idx == 0 else (final_summary, None)

    monkeypatch.setattr(rem, "compute_slice_summary", fake_slice_summary)
    monkeypatch.setattr(rem, "measure_latency", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(
        rem,
        "compute_window_pairing_metrics",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("pairing boom")),
    )

    with pytest.raises(RuntimeError, match="pairing boom"):
        rem.compute_real_metrics(
            _FakeRunner(),
            _FakeModel(),
            calibration_data=[{"input_ids": [1]}, {"input_ids": [2]}],
            adapter=object(),
            preview_n=1,
            final_n=1,
            config=SimpleNamespace(context={"eval": {"bootstrap": {"enabled": False}}}),
        )
