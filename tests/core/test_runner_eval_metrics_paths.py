from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import invarlock.core.runner_eval_metrics as rem


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


def test_compute_real_metrics_falls_back_for_non_finite_pm_and_delta(
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

    assert metrics["primary_metric"]["preview"] == 1.0
    assert metrics["primary_metric"]["final"] == 1.0
    assert metrics["primary_metric"]["degraded_reason"] == "non_finite_pm"
    assert metrics["logloss_delta"] == 0.0
