from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from invarlock.core.runner_runtime.eval_windows import (
    compute_slice_summary,
    resolve_limit,
    slice_calibration,
)


class _RunnerRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, str, dict[str, object]]] = []

    def _log_event(
        self,
        component: str,
        operation: str,
        level: str,
        data: dict[str, object] | None = None,
    ) -> None:
        self.events.append((component, operation, level, data or {}))


class _LossModel(torch.nn.Module):
    def __init__(self, loss: float) -> None:
        super().__init__()
        self.loss = loss
        self.param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, labels=None):  # noqa: D401
        return SimpleNamespace(loss=torch.tensor(self.loss, device=input_ids.device))


class _MissingLossModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, labels=None):  # noqa: D401
        return SimpleNamespace()


class _RandomAccessOnly:
    def __init__(self, values: list[int]) -> None:
        self._values = values

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise TypeError("slice disabled")
        return self._values[index]

    def __len__(self) -> int:
        return len(self._values)


class _IterableOnly:
    def __init__(self, values: list[int]) -> None:
        self._values = values

    def __iter__(self):
        return iter(self._values)


def test_slice_calibration_supports_random_access_and_materialization() -> None:
    random_access = _RandomAccessOnly([0, 1, 2, 3, 4])
    sliced, source = slice_calibration(
        random_access,
        start=1,
        count=3,
        allow_materialize=False,
    )
    assert sliced == [1, 2, 3]
    assert source is random_access

    iterable_only = _IterableOnly([10, 11, 12, 13])
    sliced_iter, materialized = slice_calibration(
        iterable_only,
        start=1,
        count=2,
        allow_materialize=True,
    )
    assert sliced_iter == [11, 12]
    assert materialized == [10, 11, 12, 13]


def test_slice_calibration_raises_without_supported_access() -> None:
    with pytest.raises(TypeError):
        slice_calibration(
            _IterableOnly([1, 2, 3]),
            start=0,
            count=2,
            allow_materialize=False,
        )


def test_slice_calibration_reraises_unexpected_access_errors() -> None:
    class _ExplodingSlice:
        def __getitem__(self, index):
            if isinstance(index, slice):
                raise AssertionError("explode")
            return 0

        def __len__(self) -> int:
            return 1

    with pytest.raises(AssertionError, match="explode"):
        slice_calibration(
            _ExplodingSlice(),
            start=0,
            count=1,
            allow_materialize=False,
        )


def test_resolve_limit_uses_all_batches_for_non_positive_request() -> None:
    assert resolve_limit([], requested=5) == 0
    assert resolve_limit([1, 2, 3], requested=0) == 3
    assert resolve_limit([1, 2, 3], requested=-1) == 3
    assert resolve_limit([1, 2, 3], requested=2) == 2


def test_resolve_limit_caps_large_request_to_batch_count() -> None:
    assert resolve_limit([1, 2, 3], requested=99) == 3


def test_compute_slice_summary_handles_tensor_masks_and_labels_without_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_STORE_EVAL_WINDOWS", "0")
    runner = _RunnerRecorder()
    model = _LossModel(0.2)
    batch = {
        "input_ids": torch.tensor([[5, 6, 7]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
        "labels": torch.tensor([[5, 6, -100]], dtype=torch.long),
    }

    summary, error = compute_slice_summary(
        runner,
        model,
        [batch],
        max_batches=1,
        start_idx=0,
        device=next(model.parameters()).device,
        resolved_loss_mode="causal",
    )

    assert error is None
    assert summary["num_batches"] == 1
    assert summary["total_tokens"] == 2
    assert summary["actual_total_tokens"] == 2
    assert summary["masked_token_counts"] == [2]
    assert summary["actual_token_counts"] == [2]
    assert summary["tokens"] == []
    assert summary["attention_masks"] == []
    assert summary["labels"] == []
    assert any(operation == "label_alignment" for _, operation, _, _ in runner.events)


def test_compute_slice_summary_treats_whitespace_false_store_flag_as_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_STORE_EVAL_WINDOWS", " false ")
    runner = _RunnerRecorder()
    model = _LossModel(0.1)
    batch = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}

    summary, error = compute_slice_summary(
        runner,
        model,
        [batch],
        max_batches=1,
        start_idx=0,
        device=next(model.parameters()).device,
        resolved_loss_mode="causal",
    )

    assert error is None
    assert summary["tokens"] == []
    assert summary["attention_masks"] == []
    assert summary["labels"] == []


def test_compute_slice_summary_zero_mask_batches_fall_back_without_debug_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INVARLOCK_DEBUG_TRACE", raising=False)
    runner = _RunnerRecorder()
    model = _LossModel(0.1)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "labels": torch.tensor([[-100, -100, -100]], dtype=torch.long),
    }

    summary, error = compute_slice_summary(
        runner,
        model,
        [batch],
        max_batches=1,
        start_idx=0,
        device=next(model.parameters()).device,
        resolved_loss_mode="causal",
    )

    assert error is None
    assert summary["masked_token_counts"] == [3]
    assert not any(
        operation == "zero_mask_batch" for _, operation, _, _ in runner.events
    )


def test_compute_slice_summary_strips_whitespace_when_storage_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_STORE_EVAL_WINDOWS", " false ")
    runner = _RunnerRecorder()
    model = _LossModel(0.2)
    batch = {
        "input_ids": torch.tensor([[5, 6, 7]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
    }

    summary, error = compute_slice_summary(
        runner,
        model,
        [batch],
        max_batches=1,
        start_idx=1,
        device=next(model.parameters()).device,
        resolved_loss_mode="causal",
    )

    assert error is None
    assert summary["tokens"] == []
    assert summary["attention_masks"] == []
    assert summary["labels"] == []


def test_compute_slice_summary_reports_mlm_missing_masks_for_unusable_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_DEBUG_TRACE", "1")
    runner = _RunnerRecorder()
    model = _MissingLossModel()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "labels": torch.tensor([[-100, -100, -100]], dtype=torch.long),
    }

    summary, error = compute_slice_summary(
        runner,
        model,
        [batch],
        max_batches=1,
        start_idx=4,
        device=next(model.parameters()).device,
        resolved_loss_mode="mlm",
    )

    assert summary["num_batches"] == 0
    assert error == {
        "error": "mlm_missing_masks",
        "detail": "MLM evaluation saw labels but zero masked tokens were accumulated; check calibration data integrity.",
    }
    operations = [operation for _, operation, _, _ in runner.events]
    assert "missing_loss" in operations
    assert "mlm_missing_masks" in operations


def test_compute_slice_summary_logs_zero_mask_batches_and_uses_token_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INVARLOCK_DEBUG_TRACE", "1")
    runner = _RunnerRecorder()
    model = _LossModel(0.3)
    batch = {
        "input_ids": torch.tensor([[4, 5, 6]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "labels": torch.tensor([[-100, -100, -100]], dtype=torch.long),
    }

    summary, error = compute_slice_summary(
        runner,
        model,
        [batch],
        max_batches=1,
        start_idx=7,
        device=next(model.parameters()).device,
        resolved_loss_mode="mlm",
    )

    assert error is None
    assert summary["num_batches"] == 1
    assert summary["total_tokens"] == 3
    assert summary["actual_total_tokens"] == 3
    assert summary["masked_token_counts"] == [3]
    zero_mask_events = [
        data
        for _, operation, _, data in runner.events
        if operation == "zero_mask_batch"
    ]
    assert zero_mask_events == [
        {
            "window_index": 7,
            "tokens_in_batch": 3,
            "masked_tokens": 0,
            "labels_sample": [-100, -100, -100],
            "fallback_weight": 3,
        }
    ]


def test_compute_slice_summary_skips_non_finite_losses_and_reports_error() -> None:
    runner = _RunnerRecorder()
    model = _LossModel(float("nan"))
    batch = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}

    summary, error = compute_slice_summary(
        runner,
        model,
        [batch],
        max_batches=1,
        start_idx=2,
        device=next(model.parameters()).device,
        resolved_loss_mode="causal",
    )

    assert summary["num_batches"] == 0
    assert summary["log_losses"] == []
    assert error == {
        "error": "non_finite_loss",
        "detail": "Evaluation produced only non-finite loss values; primary metric evidence is unavailable.",
    }
    operations = [operation for _, operation, _, _ in runner.events]
    assert "non_finite_loss" in operations
    assert "non_finite_loss_total" in operations
