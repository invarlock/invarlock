import pytest
import torch

import invarlock.edits.quant_rtn as quant_rtn_mod
from invarlock.core.api import EditRuntime
from invarlock.core.exceptions import EditError
from invarlock.edits.quant_rtn import (
    QuantTargetSelector,
    RTNQuantEdit,
    RTNQuantPlan,
    TargetModule,
)


def _target(name: str, module: torch.nn.Module) -> TargetModule:
    return TargetModule(
        name=name,
        module=module,
        selection_reason="test",
        matched_pattern="test",
        parameter_id=id(module.weight),
        module_type=f"{module.__class__.__module__}.{module.__class__.__name__}",
    )


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
    """quant_rtn is a minimal INT8 simulation edit; 4-bit is not supported."""
    with pytest.raises(ValueError):
        RTNQuantEdit(bitwidth=4)


def test_quant_rtn_rejects_group_size() -> None:
    with pytest.raises(ValueError, match="group_size is unsupported"):
        RTNQuantEdit(group_size=128)

    model = torch.nn.Linear(16, 16, bias=False)
    adapter = type("Adapter", (), {"describe": lambda _self, _m: {"n_layer": 1}})()
    edit = RTNQuantEdit(scope="all", max_modules=1)

    with pytest.raises(ValueError, match="group_size"):
        edit.apply(model, adapter, plan={"group_size": 128})


def test_quant_rtn_rejects_non_positive_max_modules() -> None:
    with pytest.raises(ValueError, match="max_modules"):
        RTNQuantEdit(max_modules=0)


def test_quant_rtn_plan_rejects_invalid_options() -> None:
    with pytest.raises(ValueError, match="max_modules"):
        RTNQuantPlan.from_payload({"max_modules": "many"})
    with pytest.raises(ValueError, match="Clamp ratio"):
        RTNQuantPlan(clamp_ratio=0.75).validate()
    with pytest.raises(ValueError, match="Scope"):
        RTNQuantPlan(scope="embed").validate()  # type: ignore[arg-type]


def test_quant_rtn_normalizers_cover_string_and_invalid_inputs() -> None:
    assert RTNQuantEdit._normalize_per_channel_option(None, default=False) is False
    assert RTNQuantEdit._normalize_per_channel_option("yes", default=False) is True
    assert RTNQuantEdit._normalize_per_channel_option("off", default=True) is False
    with pytest.raises(ValueError, match="per_channel"):
        RTNQuantEdit._normalize_per_channel_option("sometimes")

    selectors = RTNQuantEdit._normalize_module_selectors(
        {
            "attention": "attn.c_attn",
            "ffn": ["mlp.c_fc", "", 7],
            12: ["ignored"],
            "empty": [],
            "bad": object(),
        }
    )
    assert selectors == {"attention": ["attn.c_attn"], "ffn": ["mlp.c_fc"]}


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


def test_quant_rtn_apply_accepts_per_channel_and_module_selectors() -> None:
    model = torch.nn.Linear(16, 16, bias=False)
    adapter = type("Adapter", (), {"describe": lambda _self, _m: {"n_layer": 1}})()
    edit = RTNQuantEdit(scope="all", max_modules=1)

    result = edit.apply(
        model,
        adapter,
        plan={
            "scope": "all",
            "max_modules": 1,
            "per_channel": True,
            "module_selectors": {"attention": ["attn.c_attn"], "ffn": ["mlp.c_fc"]},
        },
    )

    assert result["deltas"]["params_changed"] > 0
    assert result["plan"]["per_channel"] is True
    assert result["plan"]["quantization_mode"] == "rtn_dequantized_weight_edit"
    assert result["plan"]["storage_format"] == "float_dequantized"
    assert result["plan"]["packed_quantized_storage"] is False
    assert result["plan"]["runtime_memory_reduction"] is False
    assert result["plan"]["module_selectors"] == {
        "attention": ["attn.c_attn"],
        "ffn": ["mlp.c_fc"],
    }
    assert result["plan"]["selected_modules"] == result["plan"]["target_modules"]
    assert (
        result["plan"]["physically_quantized_modules"]
        == result["plan"]["modules_quantized"]
    )
    assert result["plan_digest"].startswith("sha256:")
    assert "estimated_memory_saved_bytes" not in result["plan"]


def test_quant_rtn_counts_layers_modified_for_qwen_style_module_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_0 = torch.nn.Linear(2, 2, bias=False)
    module_1 = torch.nn.Linear(2, 2, bias=False)
    model = torch.nn.Sequential(module_0, module_1)
    adapter = type("Adapter", (), {"describe": lambda _self, _m: {"n_layer": 2}})()
    edit = RTNQuantEdit(scope="attn", max_modules=2)

    monkeypatch.setattr(
        RTNQuantEdit,
        "_select_target_modules",
        lambda self, _model: [
            _target("model.layers.0.self_attn.q_proj", module_0),
            _target("model.layers.1.self_attn.k_proj", module_1),
        ],
    )
    monkeypatch.setattr(
        RTNQuantEdit,
        "_apply_rtn_quantization",
        lambda self, *_args, **_kwargs: {
            "params_quantized": 16,
            "scale_stats": {},
            "error_metrics": {},
        },
    )

    result = edit.apply(model, adapter, plan={"scope": "attn", "max_modules": 2})

    assert result["deltas"]["params_changed"] == 32
    assert result["deltas"]["layers_modified"] == 2


def test_quant_rtn_apply_rejects_per_channel_false_override() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    adapter = type("Adapter", (), {"describe": lambda _self, _m: {"n_layer": 1}})()
    edit = RTNQuantEdit(scope="all", max_modules=1)

    with pytest.raises(ValueError, match="only supports per_channel=True"):
        edit.apply(model, adapter, plan={"scope": "all", "per_channel": False})


def test_quant_rtn_apply_fails_closed_when_no_target_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    adapter = type("Adapter", (), {"describe": lambda _self, _m: {"n_layer": 1}})()
    edit = RTNQuantEdit(scope="attn", max_modules=1)

    monkeypatch.setattr(RTNQuantEdit, "_select_target_modules", lambda *_args: [])

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
        "_select_target_modules",
        lambda self, _model: [_target("linear", model)],
    )

    with pytest.raises(RuntimeError, match="apply boom"):
        edit.apply(model, adapter, plan={"scope": "all", "max_modules": 1})


def test_quant_rtn_preview_reports_simulation_memory_fields() -> None:
    model = torch.nn.Linear(16, 16, bias=False)
    adapter = type(
        "Adapter",
        (),
        {"describe": lambda _self, _m: {"n_layer": 1, "total_params": 256}},
    )()
    edit = RTNQuantEdit(scope="all", max_modules=1)

    preview = edit.preview(model, adapter, None)
    metrics = preview["preview_metrics"]
    plan = preview["plan"]

    assert metrics["theoretical_packed_memory_saved_bytes"] > 0
    assert metrics["theoretical_packed_bits_per_param"] == 8
    assert metrics["actual_storage_format"] == "float_dequantized"
    assert metrics["packed_quantized_storage"] is False
    assert metrics["runtime_memory_reduction"] is False
    assert "estimated_memory_saved_bytes" not in metrics
    assert plan["quantization_mode"] == "rtn_dequantized_weight_edit"
    assert plan["plan_digest"].startswith("sha256:")
    assert plan["selected_modules"] == plan["physically_quantized_modules"]
    assert "parameter_id" not in plan["target_selection"][0]
    assert "parameter_id" not in plan["quantization_stats"]["module_stats"][0]
    assert plan["runtime_debug"]["target_parameter_ids"][0]["parameter_id"]


def test_quant_rtn_can_edit_and_limit_targets() -> None:
    edit = RTNQuantEdit()
    assert edit.can_edit({"n_layer": 1, "total_params": 1001}) is True
    assert edit.can_edit({"n_layer": 1, "total_params": 100}) is False
    assert edit.can_edit({"total_params": 1001}) is False
    assert edit.can_edit(
        {
            "n_layer": 1,
            "total_params": 1001,
            "module_names": ["transformer.h.0.mlp.c_fc"],
        }
    )
    assert not edit.can_edit(
        {
            "n_layer": 1,
            "total_params": 1001,
            "module_names": ["transformer.wte"],
        }
    )

    targets = [
        _target(str(index), torch.nn.Linear(2, 2, bias=False)) for index in range(3)
    ]
    limited, total = RTNQuantEdit._limit_targets(targets, 2)
    assert [target.name for target in limited] == ["0", "1"]
    assert total == 3
    assert RTNQuantEdit._limit_targets(targets, None)[0] == targets


def test_quant_rtn_target_selector_explains_user_and_default_matches() -> None:
    class MiniModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = torch.nn.Module()
            self.attn.c_attn = torch.nn.Linear(16, 16, bias=False)
            self.mlp = torch.nn.Module()
            self.mlp.c_fc = torch.nn.Linear(16, 16, bias=False)
            self.small = torch.nn.Linear(2, 2, bias=False)
            self.relu = torch.nn.ReLU()

    model = MiniModel()
    user_targets = QuantTargetSelector(
        scope="attn",
        module_selectors={"attention": ["attn.c_attn"]},
    ).select(model)
    assert user_targets[0].name == "attn.c_attn"
    assert user_targets[0].selection_reason == "model_profile_selector"

    default_targets = QuantTargetSelector(scope="ffn").select(model)
    assert [target.name for target in default_targets] == ["mlp.c_fc"]
    assert default_targets[0].selection_reason == "name_heuristic"

    all_targets = QuantTargetSelector(scope="all", min_params=100).select(model)
    assert {target.name for target in all_targets} == {"attn.c_attn", "mlp.c_fc"}


def test_quant_rtn_target_selector_covers_empty_weight_and_duplicate_patterns() -> None:
    class MiniModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = torch.nn.Module()
            self.attn.c_attn = torch.nn.Linear(16, 16, bias=False)
            self.weightless = torch.nn.Linear(16, 16, bias=False)
            self.weightless._parameters["weight"] = None

    model = MiniModel()
    selector = QuantTargetSelector(
        scope="ffn",
        module_selectors={"ffn": ["attn.c_attn", "attn.c_attn", " "]},
    )

    targets = selector.select(model)

    assert [target.name for target in targets] == ["attn.c_attn"]
    assert selector._selector_patterns_for_scope() == ("attn.c_attn",)


def test_quant_rtn_plan_digest_changes_with_meaningful_fields() -> None:
    base = RTNQuantPlan(scope="attn", clamp_ratio=0.0)
    changed = RTNQuantPlan(scope="attn", clamp_ratio=0.01)

    assert base.digest(target_modules=["a"]) != changed.digest(target_modules=["a"])
    assert base.digest(target_modules=["a"]) != base.digest(target_modules=["b"])


def test_quant_rtn_plan_digest_excludes_runtime_parameter_ids() -> None:
    plan = RTNQuantPlan(scope="all")
    target_a = {
        "module_name": "mlp.c_fc",
        "module_type": "torch.nn.modules.linear.Linear",
        "weight_shape": [16, 16],
        "params": 256,
        "selection_reason": "scope_all_min_params",
        "matched_pattern": None,
        "parameter_id": "1234",
    }
    target_b = dict(target_a)
    target_b["parameter_id"] = "9876"

    assert plan.digest(target_selection=[target_a]) == plan.digest(
        target_selection=[target_b]
    )


def test_quant_rtn_plan_digest_reproducible_for_same_model_structure() -> None:
    adapter = type("Adapter", (), {"describe": lambda _self, _m: {"n_layer": 1}})()
    first = torch.nn.Linear(16, 16, bias=False)
    second = torch.nn.Linear(16, 16, bias=False)
    edit = RTNQuantEdit(scope="all", max_modules=1)

    first_result = edit.apply(first, adapter, plan={"scope": "all", "max_modules": 1})
    second_result = edit.apply(second, adapter, plan={"scope": "all", "max_modules": 1})

    assert first_result["plan_digest"] == second_result["plan_digest"]


def test_quant_rtn_apply_emits_error_metrics_and_target_metadata() -> None:
    model = torch.nn.Linear(16, 16, bias=False)
    adapter = type("Adapter", (), {"describe": lambda _self, _m: {"n_layer": 1}})()
    edit = RTNQuantEdit(scope="all", max_modules=1)

    result = edit.apply(model, adapter, plan={"scope": "all", "max_modules": 1})
    module_name = result["plan"]["modules_quantized"][0]
    module_entry = result["deltas"]["bitwidth_map"][module_name]

    assert result["plan"]["aggregate_error_metrics"]["rmse"] >= 0.0
    assert module_entry["error_metrics"]["relative_rmse"] >= 0.0
    assert module_entry["actual_storage_format"] == "float_dequantized"
    assert module_entry["packed_quantized_storage"] is False
    assert module_entry["selection_reason"] == "scope_all_min_params"
    assert "parameter_id" not in module_entry
    assert "parameter_id" not in result["plan"]["target_selection"][0]
    assert result["plan"]["runtime_debug"]["target_parameter_ids"][0]["parameter_id"]
    assert result["plan"]["target_selection"][0]["weight_shape"] == [16, 16]


def test_quant_rtn_deduplicates_tied_weights() -> None:
    class TiedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Linear(16, 16, bias=False)
            self.b = torch.nn.Linear(16, 16, bias=False)
            self.b.weight = self.a.weight

    model = TiedModel()
    adapter = type("Adapter", (), {"describe": lambda _self, _m: {"n_layer": 1}})()
    edit = RTNQuantEdit(scope="all")

    result = edit.apply(model, adapter, plan={"scope": "all"})

    assert result["plan"]["total_modules_quantized"] == 1
    assert result["plan"]["total_modules_selected"] == 2
    assert result["plan"]["selected_modules"] == ["a", "b"]
    assert result["plan"]["physically_quantized_modules"] == ["a"]
    assert result["plan"]["deduplicated_modules"] == ["b"]
    assert "deduplicated_parameter_ids" not in result["plan"]
    assert len(result["plan"]["runtime_debug"]["deduplicated_parameter_ids"]) == 1
    assert result["plan"]["tied_parameter_groups"] == [["a", "b"]]


def test_quant_rtn_apply_runs_guard_chain_hooks() -> None:
    class GuardChain:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def prepare_all(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
            self.calls.append("prepare")
            return {"ok": True}

        def before_edit_all(self, *_args):  # noqa: ANN002
            self.calls.append("before")

        def after_edit_all(self, *_args):  # noqa: ANN002
            self.calls.append("after")

        def finalize_all(self, *_args):  # noqa: ANN002
            self.calls.append("finalize")
            return {"ok": True}

        def all_passed(self, *_args):  # noqa: ANN002
            self.calls.append("all_passed")
            return True

    model = torch.nn.Linear(16, 16, bias=False)
    adapter = type("Adapter", (), {"describe": lambda _self, _m: {"n_layer": 1}})()
    guard_chain = GuardChain()
    edit = RTNQuantEdit(scope="all", max_modules=1, guard_chain=guard_chain)  # type: ignore[arg-type]

    edit.apply(model, adapter)

    assert guard_chain.calls == ["prepare", "before", "after", "finalize", "all_passed"]


def test_quant_rtn_apply_rejects_zero_param_quantization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(16, 16, bias=False)
    adapter = type("Adapter", (), {"describe": lambda _self, _m: {"n_layer": 1}})()
    edit = RTNQuantEdit(scope="all", max_modules=1)

    monkeypatch.setattr(
        RTNQuantEdit,
        "_apply_rtn_quantization",
        lambda *_args, **_kwargs: {"params_quantized": 0, "error_metrics": {}},
    )

    with pytest.raises(EditError, match="without changing any parameters"):
        edit.apply(model, adapter)


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


def test_quant_rtn_compute_stats_skips_channel_stats_for_one_dimensional_weight() -> (
    None
):
    edit = RTNQuantEdit(scope="all")
    module = torch.nn.BatchNorm1d(4, affine=True)
    stats = edit._compute_quantization_stats([_target("norm", module)])

    assert stats["module_stats"][0]["name"] == "norm"
    assert "channel_stats" not in stats["module_stats"][0]


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
