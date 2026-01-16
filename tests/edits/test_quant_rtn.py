import io

import pytest
import torch
from rich.console import Console

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


def test_quant_rtn_output_format() -> None:
    model = torch.nn.Linear(4, 4, bias=False)
    adapter = type("Adapter", (), {"describe": lambda _self, _m: {"n_layer": 1}})()
    edit = RTNQuantEdit(scope="all", max_modules=1)
    out = io.StringIO()
    console = Console(file=out, force_terminal=False)

    edit.apply(
        model,
        adapter,
        scope="all",
        max_modules=1,
        console=console,
    )

    text = out.getvalue()
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines
    assert all(line.startswith("[EDIT]") for line in lines)
    assert all(ord(ch) < 128 for ch in text)


def test_quant_rtn_emit_flag_suppresses_output() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    adapter = type("Adapter", (), {"describe": lambda _self, _m: {"n_layer": 1}})()
    edit = RTNQuantEdit(scope="all", max_modules=1)
    out = io.StringIO()
    console = Console(file=out, force_terminal=False)

    edit.apply(
        model,
        adapter,
        scope="all",
        max_modules=1,
        console=console,
        emit=False,
    )

    assert out.getvalue() == ""
