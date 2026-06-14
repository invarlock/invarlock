import pytest
import torch
import torch.nn as nn

from invarlock.guards.variance import VarianceGuard
from invarlock.guards.variance_scaling import (
    _emit_progress,
    _t5_dense_relu_dense,
    equalise_residual_variance,
)


def test_variance_guard_tap_alias_matches_model_layers_down_proj():
    g = VarianceGuard(
        policy={
            "tap": "model.layers.*.mlp.down_proj",
        }
    )
    # Internal canonical naming is transformer.h.<idx>.mlp.c_proj.
    assert g._matches_tap("transformer.h.0.mlp.c_proj")  # noqa: SLF001


def test_variance_guard_normalize_module_name_covers_empty_and_suffix_cases():
    g = VarianceGuard()

    assert g._normalize_module_name("   ") == ""  # noqa: SLF001
    assert (
        g._normalize_module_name("transformer.h.0.mlp") == "transformer.h.0.mlp.c_proj"
    )  # noqa: SLF001,E501
    assert (
        g._normalize_module_name("transformer.h.0.attn")
        == "transformer.h.0.attn.c_proj"
    )  # noqa: SLF001,E501
    assert g._normalize_module_name("block0.attn") == "transformer.h.0.attn.c_proj"  # noqa: SLF001,E501


def test_variance_guard_set_run_context_noop_forces_monitor_only():
    guard = VarianceGuard()
    report = type(
        "Report",
        (),
        {"edit": {"name": "noop"}, "meta": {}, "context": {}},
    )()

    assert guard._monitor_only is False  # noqa: SLF001
    guard.set_run_context(report)
    assert guard._monitor_only is True  # noqa: SLF001


def test_equalise_residual_variance_apply_false_does_not_mutate_weights():
    class ToyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.c_proj = nn.Linear(4, 4, bias=False)
            with torch.no_grad():
                self.mlp.c_proj.weight.copy_(torch.eye(4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Scale output so alpha != 1 and we get a non-empty scale dict.
            return self.mlp.c_proj(x) * 2.0

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList([ToyBlock()])

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            x = input_ids.float()
            if x.dim() == 2:
                x = x.unsqueeze(-1).repeat(1, 1, 4)
            for blk in self.transformer.h:
                x = blk(x)
            return x

    model = ToyModel()
    original = model.transformer.h[0].mlp.c_proj.weight.detach().clone()

    dataloader = [{"input_ids": torch.ones(2, 3, dtype=torch.long)}]
    out = equalise_residual_variance(
        model,
        dataloader,
        windows=1,
        tol=0.0,
        clamp_range=None,
        apply=False,
        allow_empty=False,
    )

    assert "block0.mlp" in out
    assert torch.equal(model.transformer.h[0].mlp.c_proj.weight, original)


def test_variance_scaling_emit_progress_invokes_callback() -> None:
    events = []
    _emit_progress(
        lambda payload: events.append(
            (payload.phase, payload.completed, payload.total)
        ),
        completed=2,
        total=5,
    )
    assert events == [("calibration", 2, 5)]


def test_equalise_residual_variance_raises_on_non_iterable_dataloader():
    class ToyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.c_proj = nn.Linear(4, 4, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.mlp.c_proj(x)

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList([ToyBlock()])

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            x = input_ids.float()
            if x.dim() == 2:
                x = x.unsqueeze(-1).repeat(1, 1, 4)
            for blk in self.transformer.h:
                x = blk(x)
            return x

    model = ToyModel()

    with pytest.raises(
        ValueError, match="Empty dataloader provided and allow_empty=False"
    ):
        equalise_residual_variance(
            model,
            dataloader=123,
            windows=1,
            tol=0.0,
            clamp_range=None,
            apply=False,
            allow_empty=False,
        )


def test_t5_dense_relu_dense_falls_back_when_layer_iteration_fails() -> None:
    dense = nn.Linear(2, 2)

    class _BrokenLayers:
        def __iter__(self):
            raise TypeError("broken")

    block = nn.Module()
    block.layer = _BrokenLayers()
    block.DenseReluDense = dense

    assert _t5_dense_relu_dense(block) is dense


def test_t5_dense_relu_dense_finds_dense_inside_layer_iterable() -> None:
    dense = nn.Linear(2, 2)
    block = nn.Module()
    block.layer = [nn.Module()]
    block.layer[0].DenseReluDense = dense

    assert _t5_dense_relu_dense(block) is dense


def test_t5_dense_relu_dense_uses_fallback_after_empty_or_nonmatching_layers() -> None:
    dense = nn.Linear(2, 2)

    empty_block = nn.Module()
    empty_block.layer = []
    empty_block.DenseReluDense = dense
    assert _t5_dense_relu_dense(empty_block) is dense

    nonmatching_block = nn.Module()
    nonmatching_block.layer = [nn.Module()]
    nonmatching_block.DenseReluDense = dense
    assert _t5_dense_relu_dense(nonmatching_block) is dense


def test_equalise_residual_variance_normalizes_dict_batches_with_labels_and_masks() -> (
    None
):
    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.proj.weight.fill_(2.0)
            self.calls: list[
                tuple[torch.Size, torch.Size | None, torch.Size | None]
            ] = []

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
        ) -> torch.Tensor:
            self.calls.append(
                (
                    input_ids.shape,
                    None if attention_mask is None else attention_mask.shape,
                    None if labels is None else labels.shape,
                )
            )
            return self.proj(torch.ones(1, 2, device=input_ids.device))

    model = _Model()
    out = equalise_residual_variance(
        model,
        dataloader=[
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [1, 2]},
            {"inputs": [3, 4], "attention_mask": [1, 0]},
            {
                "input_ids": torch.tensor([[5, 6]]),
                "attention_mask": torch.tensor([[1, 1]]),
                "labels": torch.tensor([[5, 6]]),
            },
        ],
        windows=3,
        tol=0.0,
        clamp_range=None,
        apply=False,
        allow_empty=False,
        target_modules={"proj": model.proj},
    )

    assert model.calls == [
        (torch.Size([1, 2]), torch.Size([1, 2]), torch.Size([1, 2])),
        (torch.Size([1, 2]), torch.Size([1, 2]), None),
        (torch.Size([1, 2]), torch.Size([1, 2]), torch.Size([1, 2])),
    ]
    assert set(out) == {"proj"}


def test_equalise_residual_variance_falls_back_to_positional_call_after_typeerror() -> (
    None
):
    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.proj.weight.fill_(2.0)
            self.positional_calls = 0

        def forward(self, *args, **kwargs):  # noqa: ANN002,ANN003
            if kwargs:
                raise TypeError("keyword path unavailable")
            self.positional_calls += 1
            input_ids = args[0]
            return self.proj(torch.ones(1, 2, device=input_ids.device))

    model = _Model()
    out = equalise_residual_variance(
        model,
        dataloader=[{"input_ids": [1, 2], "attention_mask": [1, 1]}],
        windows=1,
        tol=0.0,
        clamp_range=None,
        apply=False,
        allow_empty=False,
        target_modules={"proj": model.proj},
    )

    assert model.positional_calls == 1
    assert set(out) == {"proj"}
