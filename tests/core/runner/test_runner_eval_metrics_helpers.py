from __future__ import annotations

import math
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import invarlock.core.runner_runtime.eval_metrics as rem
import invarlock.core.runner_runtime.eval_metrics_multimodal as remm
from tests.core._support_runner_eval_metrics import _FakeModel, _FakeRunner, _summary


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
    assert remm._decode_prediction_text(["cat", "dog"]) == "cat"
    assert remm._decode_prediction_text(()) == ""
    assert rem._normalize_answer_text("  Cat   Dog\n") == "cat dog"
    assert rem._normalize_answer_text(None) == ""
    assert rem._normalize_answer_text(0) == "0"
    assert remm._normalize_reference_answers(123) == []
    assert remm._normalize_reference_answers(" cat ") == ["cat"]
    assert sorted(remm._normalize_reference_answers({" dog ", ""})) == ["dog"]
    assert (
        rem._resolve_metric_kind(
            SimpleNamespace(context={"eval": {"metric": {}}}),
            fallback="ppl_causal",
        )
        == "ppl_causal"
    )
    assert (
        remm._resolve_metric_kind(
            SimpleNamespace(context={"eval": {"metric": {"kind": "Accuracy"}}}),
            fallback="ppl_causal",
        )
        == "accuracy"
    )
    assert (
        remm._resolve_metric_kind(
            SimpleNamespace(context={"eval": {"metric": []}}),
            fallback="ppl_causal",
        )
        == "ppl_causal"
    )
    assert rem._is_multimodal_batch({"example_id": "ex-1"}) is True
    assert rem._is_multimodal_batch({"input_ids": [1, 2, 3]}) is False
    assert rem._has_multimodal_batches([], [{"answers": ["cat"]}]) is True
    assert rem._has_multimodal_batches([], [{"input_ids": [1, 2, 3]}]) is False


def test_runner_eval_metrics_multimodal_hook_resolution_rejects_noncallable_attr() -> (
    None
):
    class _Adapter:
        prepare_model_inputs = "not-callable"

    assert remm._resolve_adapter_hook(_Adapter(), "prepare_model_inputs") is None


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
