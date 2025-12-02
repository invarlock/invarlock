import pytest
import torch

from invarlock.edits.quant_rtn import RTNQuantEdit


def test_percentile_clamp_reduces_outliers() -> None:
    edit = RTNQuantEdit(clamp_ratio=0.01)
    weight = torch.tensor(
        [
            [1.0, 2.0, 100.0, -50.0],
            [0.5, -0.2, 0.1, 0.2],
        ],
        dtype=torch.float32,
    )

    clamped = edit._apply_outlier_clipping(weight.clone(), edit.clamp_ratio)

    assert torch.all(clamped.abs() <= weight.abs() + 1e-6)
    assert not torch.equal(clamped, weight)


def test_quant_rtn_rejects_non_int8_bitwidth() -> None:
    """quant_rtn is a minimal INT8 demo edit; 4-bit is not supported."""
    with pytest.raises(ValueError):
        RTNQuantEdit(bitwidth=4)
