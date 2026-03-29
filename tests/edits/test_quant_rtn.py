import pytest
import torch

import invarlock.edits.quant_rtn as quant_rtn_mod
from invarlock.core.api import EditRuntime
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


def test_percentile_clamp_supports_fp16_inputs() -> None:
    # Regression test: torch.quantile() is not implemented for fp16/bf16 on some
    # backends, so clipping must compute thresholds in float32.
    edit = RTNQuantEdit(clamp_ratio=0.1)
    weight = torch.linspace(-1, 1, steps=100, dtype=torch.float16).view(10, 10)

    clamped = edit._apply_outlier_clipping(weight.clone(), edit.clamp_ratio)

    assert clamped.dtype == weight.dtype
    assert clamped.shape == weight.shape
    assert torch.isfinite(clamped).all()

    lower = edit.clamp_ratio / 2
    upper = 1 - lower
    q = torch.quantile(
        weight.float(),
        torch.tensor([lower, upper], dtype=torch.float32),
        dim=1,
        keepdim=True,
    )
    q_low, q_high = q[0], q[1]
    # Allow for fp16 rounding when comparing against float32 quantiles.
    assert (clamped.float() >= q_low - 1e-3).all()
    assert (clamped.float() <= q_high + 1e-3).all()


def test_quant_rtn_rejects_non_int8_bitwidth() -> None:
    """quant_rtn is a minimal INT8 demo edit; 4-bit is not supported."""
    with pytest.raises(ValueError):
        RTNQuantEdit(bitwidth=4)


def test_quant_rtn_module_has_no_functional_apply_shim() -> None:
    assert "apply" not in quant_rtn_mod.__dict__


def test_quant_rtn_output_format(caplog: pytest.LogCaptureFixture) -> None:
    model = torch.nn.Linear(4, 4, bias=False)
    adapter = type("Adapter", (), {"describe": lambda _self, _m: {"n_layer": 1}})()
    edit = RTNQuantEdit(scope="all", max_modules=1)
    with caplog.at_level("INFO", logger="invarlock.edits.quant_rtn"):
        edit.apply(
            model,
            adapter,
            plan={"scope": "all", "max_modules": 1},
            runtime=EditRuntime(emit=True),
        )

    text = "\n".join(record.message for record in caplog.records)
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines
    assert all(line.startswith("[EDIT]") for line in lines)
    assert all(ord(ch) < 128 for ch in text)


def test_quant_rtn_emit_flag_suppresses_output(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    adapter = type("Adapter", (), {"describe": lambda _self, _m: {"n_layer": 1}})()
    edit = RTNQuantEdit(scope="all", max_modules=1)
    with caplog.at_level("INFO", logger="invarlock.edits.quant_rtn"):
        edit.apply(
            model,
            adapter,
            plan={"scope": "all", "max_modules": 1},
            runtime=EditRuntime(emit=False),
        )

    assert not caplog.records


def test_quant_rtn_logs_when_console_missing(caplog: pytest.LogCaptureFixture) -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    adapter = type("Adapter", (), {"describe": lambda _self, _m: {"n_layer": 1}})()
    edit = RTNQuantEdit(scope="all", max_modules=1)

    with caplog.at_level("INFO", logger="invarlock.edits.quant_rtn"):
        edit.apply(
            model,
            adapter,
            plan={"scope": "all", "max_modules": 1},
            runtime=EditRuntime(emit=True),
        )

    assert caplog.records
    assert any(record.message.startswith("[EDIT]") for record in caplog.records)


def test_quant_rtn_apply_rejects_unsupported_plan_fields() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    adapter = type("Adapter", (), {"describe": lambda _self, _m: {"n_layer": 1}})()
    edit = RTNQuantEdit(scope="all", max_modules=1)

    result = edit.apply(model, adapter, plan={"bits": 8})

    assert result["error"] == "Unsupported RTN plan fields: bits"


def test_quant_rtn_apply_rejects_non_int8_bitwidth_override() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    adapter = type("Adapter", (), {"describe": lambda _self, _m: {"n_layer": 1}})()
    edit = RTNQuantEdit(scope="all", max_modules=1)

    result = edit.apply(model, adapter, plan={"bitwidth": 4})

    assert "only supports 8-bit quantization" in result["error"]
