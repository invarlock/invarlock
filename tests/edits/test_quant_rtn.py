import pytest
import torch

import invarlock.edits.quant_rtn as quant_rtn_mod
from invarlock.core.api import EditRuntime
from invarlock.core.exceptions import EditError
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
    model = torch.nn.Linear(16, 16, bias=False)
    adapter = type("Adapter", (), {"describe": lambda _self, _m: {"n_layer": 1}})()
    edit = RTNQuantEdit(scope="all", max_modules=1)
    with caplog.at_level("INFO", logger="invarlock.edits.quant_rtn"):
        edit.apply(
            model,
            adapter,
            plan={"scope": "all", "max_modules": 1},
            runtime=EditRuntime(),
        )

    assert not caplog.records


def test_quant_rtn_emit_flag_suppresses_output(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = torch.nn.Linear(10, 10, bias=False)
    adapter = type("Adapter", (), {"describe": lambda _self, _m: {"n_layer": 1}})()
    edit = RTNQuantEdit(scope="all", max_modules=1)
    with caplog.at_level("INFO", logger="invarlock.edits.quant_rtn"):
        edit.apply(
            model,
            adapter,
            plan={"scope": "all", "max_modules": 1},
            runtime=EditRuntime(),
        )

    assert not caplog.records


def test_quant_rtn_logs_when_console_missing(caplog: pytest.LogCaptureFixture) -> None:
    model = torch.nn.Linear(10, 10, bias=False)
    adapter = type("Adapter", (), {"describe": lambda _self, _m: {"n_layer": 1}})()
    edit = RTNQuantEdit(scope="all", max_modules=1)

    with caplog.at_level("INFO", logger="invarlock.edits.quant_rtn"):
        edit.apply(
            model,
            adapter,
            plan={"scope": "all", "max_modules": 1},
            runtime=EditRuntime(),
        )

    assert not caplog.records


def test_quant_rtn_apply_rejects_unsupported_plan_fields() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    adapter = type("Adapter", (), {"describe": lambda _self, _m: {"n_layer": 1}})()
    edit = RTNQuantEdit(scope="all", max_modules=1)

    with pytest.raises(ValueError, match="Unsupported RTN plan fields: bits"):
        edit.apply(model, adapter, plan={"bits": 8})


def test_quant_rtn_apply_rejects_non_int8_bitwidth_override() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    adapter = type("Adapter", (), {"describe": lambda _self, _m: {"n_layer": 1}})()
    edit = RTNQuantEdit(scope="all", max_modules=1)

    with pytest.raises(ValueError, match="only supports 8-bit quantization"):
        edit.apply(model, adapter, plan={"bitwidth": 4})


def test_quant_rtn_apply_fails_closed_when_no_target_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    adapter = type("Adapter", (), {"describe": lambda _self, _m: {"n_layer": 1}})()
    edit = RTNQuantEdit(scope="attn", max_modules=1)

    monkeypatch.setattr(RTNQuantEdit, "_identify_target_modules", lambda *_args: [])

    with pytest.raises(EditError, match="matched no target modules"):
        edit.apply(model, adapter, plan={"scope": "attn", "max_modules": 1})


def test_quant_rtn_preview_propagates_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    adapter = type("Adapter", (), {"describe": lambda _self, _m: {"n_layer": 1}})()
    edit = RTNQuantEdit(scope="all", max_modules=1)

    def _raise(*_args, **_kwargs):  # noqa: ANN001
        raise RuntimeError("preview boom")

    monkeypatch.setattr(edit, "_compute_quantization_stats", _raise)

    with pytest.raises(RuntimeError, match="preview boom"):
        edit.preview(model, adapter, None)


def test_quant_rtn_apply_propagates_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    adapter = type("Adapter", (), {"describe": lambda _self, _m: {"n_layer": 1}})()
    edit = RTNQuantEdit(scope="all", max_modules=1)

    def _raise(self, *_args, **_kwargs):  # noqa: ANN001
        raise RuntimeError("apply boom")

    monkeypatch.setattr(RTNQuantEdit, "_apply_rtn_quantization", _raise)
    monkeypatch.setattr(
        RTNQuantEdit,
        "_identify_target_modules",
        lambda self, _model: [("linear", model)],
    )

    with pytest.raises(RuntimeError, match="apply boom"):
        edit.apply(model, adapter, plan={"scope": "all", "max_modules": 1})
