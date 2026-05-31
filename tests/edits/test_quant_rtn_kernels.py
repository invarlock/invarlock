from __future__ import annotations

import pytest
import torch

from invarlock.edits.quant_rtn import RTNQuantEdit
from tests.edits._support_quant_rtn import target as _target


def test_quant_rtn_gpt_conv1d_uses_output_feature_axis() -> None:
    transformers = pytest.importorskip("transformers.pytorch_utils")
    conv = transformers.Conv1D(nf=3, nx=2)
    with torch.no_grad():
        conv.weight.copy_(
            torch.tensor(
                [
                    [0.1, 0.2, 0.3],
                    [1.1, 1.2, 1.3],
                ],
                dtype=conv.weight.dtype,
            )
        )
    edit = RTNQuantEdit(scope="all")

    result = edit._apply_rtn_quantization(conv, bitwidth=8, clamp_ratio=0.0)

    assert list(conv.weight.shape) == [2, 3]
    assert result["scale_stats"]["channel_count"] == 3
    assert result["error_metrics"]["rmse"] >= 0.0


def test_quant_rtn_supports_one_dimensional_weight_helpers() -> None:
    edit = RTNQuantEdit(scope="all")
    module = torch.nn.BatchNorm1d(4, affine=True)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([0.0, 0.1, -0.2, 0.3]))

    matrix, restore = edit._weight_to_channel_matrix(module, module.weight)
    restored = restore(matrix)

    assert list(matrix.shape) == [1, 4]
    assert torch.equal(restored, module.weight.detach())
    result = edit._apply_rtn_quantization(module, bitwidth=8, clamp_ratio=0.0)
    assert result["params_quantized"] == 4


def test_quant_rtn_kernel_wrapper_edges() -> None:
    edit = RTNQuantEdit(scope="all")
    module = torch.nn.Linear(2, 2, bias=False)

    assert edit._is_transformers_conv1d(module) is False
    dequantized, scales, stats = edit._quantize_per_channel(
        module.weight.detach(),
        qmin=-128,
        qmax=127,
    )

    assert dequantized.shape == module.weight.shape
    assert scales.numel() == 2
    assert stats["channel_count"] == 2


def test_quant_rtn_compute_stats_skips_channel_stats_for_one_dimensional_weight() -> (
    None
):
    edit = RTNQuantEdit(scope="all")
    module = torch.nn.BatchNorm1d(4, affine=True)
    stats = edit._compute_quantization_stats([_target("norm", module)])

    assert stats["module_stats"][0]["name"] == "norm"
    assert "channel_stats" not in stats["module_stats"][0]


def test_quant_rtn_stats_are_finite_for_single_value_channels() -> None:
    edit = RTNQuantEdit(scope="all")
    module = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(0.25)

    stats = edit._compute_quantization_stats([_target("linear", module)])
    module_stats = stats["module_stats"][0]
    quantized = edit._apply_rtn_quantization(module, bitwidth=8, clamp_ratio=0.0)

    assert torch.isfinite(torch.tensor(module_stats["weight_std"]))
    assert torch.isfinite(torch.tensor(module_stats["channel_stats"][0]["std"]))
    assert torch.isfinite(torch.tensor(quantized["scale_stats"]["scale_std"]))


def test_quant_rtn_population_std_empty_tensor_returns_zero() -> None:
    assert RTNQuantEdit._population_std(torch.empty(0)) == 0.0


def test_quant_rtn_outlier_clipping_and_error_metric_edges() -> None:
    edit = RTNQuantEdit(scope="all")
    weight = torch.tensor([[0.0, 1.0, 100.0], [0.0, -1.0, -100.0]])

    assert torch.equal(edit._apply_outlier_clipping(weight, 0.0), weight)
    clipped = edit._apply_outlier_clipping(weight, 0.2)
    assert clipped.abs().max() < weight.abs().max()

    module = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        module.weight.copy_(weight)
    quantized = edit._apply_rtn_quantization(module, bitwidth=8, clamp_ratio=0.2)
    assert quantized["clamp_applied"] is True
    assert quantized["error_metrics"]["clipped_fraction"] > 0.0

    both_zero = RTNQuantEdit._quantization_error_metrics(
        torch.zeros(4),
        torch.zeros(4),
        clipped_fraction=0.0,
        quant_code_edge_fraction=0.0,
    )
    one_zero = RTNQuantEdit._quantization_error_metrics(
        torch.ones(4),
        torch.zeros(4),
        clipped_fraction=0.0,
        quant_code_edge_fraction=0.0,
    )
    empty = RTNQuantEdit._quantization_error_metrics(
        torch.empty(0),
        torch.empty(0),
        clipped_fraction=0.0,
        quant_code_edge_fraction=0.0,
    )

    assert both_zero["cosine_similarity"] == 1.0
    assert one_zero["cosine_similarity"] == 0.0
    assert empty["mean_abs_error"] == 0.0
    assert both_zero["quant_code_edge_fraction"] == 0.0


def test_quant_rtn_aggregate_error_metric_edges() -> None:
    assert RTNQuantEdit._aggregate_error_metrics([]) == {}
    aggregate = RTNQuantEdit._aggregate_error_metrics(
        [
            {
                "params_quantized": 0,
                "error_metrics": {
                    "mean_abs_error": 0.1,
                    "max_abs_error": 0.2,
                    "rmse": 0.3,
                    "relative_rmse": 0.4,
                    "cosine_similarity": 0.5,
                    "quant_code_edge_fraction": 0.6,
                    "saturation_fraction": 0.6,
                    "clipped_fraction": 0.7,
                },
            }
        ]
    )

    assert aggregate["mean_abs_error"] == 0.1
    assert aggregate["max_abs_error"] == 0.2
    assert aggregate["quant_code_edge_fraction"] == 0.6
