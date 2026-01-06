from __future__ import annotations

import torch
import torch.nn as nn

from invarlock.guards import rmt as R


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.c_attn = nn.Linear(2, 2, bias=False)
        self.attn.c_proj = nn.Linear(2, 2, bias=False)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _TinyBlock()


def test_capture_baseline_mp_stats_allowed_module_names_filters() -> None:
    model = _TinyModel()
    allowed = ["block.attn.c_attn"]
    stats = R.capture_baseline_mp_stats(model, allowed_module_names=allowed)
    assert list(stats.keys()) == allowed


def test_capture_baseline_mp_stats_svd_failure_is_skipped(monkeypatch) -> None:
    monkeypatch.setattr(
        torch.linalg,
        "svdvals",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    stats = R.capture_baseline_mp_stats(_TinyModel())
    assert stats == {}


def test_layer_svd_stats_zero_matrix_quantile_branch_ratio_one() -> None:
    layer = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        layer.weight.zero_()
    stats = R.layer_svd_stats(layer)
    assert stats["worst_ratio"] == 1.0


def test_collect_calibration_batches_indices_policy_last_and_fallback() -> None:
    guard = R.RMTGuard()
    guard.activation_sampling["windows"]["indices_policy"] = "last"
    assert guard._collect_calibration_batches([1, 2, 3, 4], 2) == [3, 4]

    guard.activation_sampling["windows"]["indices_policy"] = "unknown"
    assert guard._collect_calibration_batches([1, 2, 3, 4], 2) == [1, 2]

    guard.activation_sampling["windows"]["indices_policy"] = "evenly_spaced"
    assert guard._collect_calibration_batches([1, 2, 3], 1) == [1]
    assert guard._collect_calibration_batches([], 1) == []
