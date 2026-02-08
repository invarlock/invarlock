import pytest
import torch
import torch.nn as nn

from invarlock.guards.variance import (
    VarianceGuard,
    _iter_transformer_layers,
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


def test_equalise_residual_variance_supports_moe_mlp_blocks():
    class ToyExpert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w2 = nn.Linear(4, 4, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.w2(x)

    class ToyMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = nn.ModuleList([ToyExpert(), ToyExpert()])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Always route to expert 0 for determinism.
            return self.experts[0](x) * 2.0

    class ToyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = ToyMoE()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.mlp(x)

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


def test_equalise_residual_variance_supports_moe_module_dict_experts():
    class ToyExpert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w2 = nn.Linear(4, 4, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.w2(x)

    class ToyMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Some MoE implementations store experts in ModuleDict (iterates over keys),
            # so equalise_residual_variance must iterate experts via values/_modules.
            self.experts = nn.ModuleDict({"0": ToyExpert(), "1": ToyExpert()})

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.experts["0"](x) * 2.0

    class ToyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block_sparse_moe = ToyMoE()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block_sparse_moe(x)

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([ToyBlock()])

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            x = input_ids.float()
            if x.dim() == 2:
                x = x.unsqueeze(-1).repeat(1, 1, 4)
            for blk in self.model.layers:
                x = blk(x)
            return x

    model = ToyModel()
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


def test_equalise_residual_variance_supports_moe_python_list_experts():
    class ToyExpert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w2 = nn.Linear(4, 4, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.w2(x)

    class ToyMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Keep experts registered for parameters, but expose `experts` as a
            # plain python list (some adapters/wrappers behave like this).
            self._experts = nn.ModuleList([ToyExpert(), ToyExpert()])
            self.experts = list(self._experts)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self._experts[0](x) * 2.0

    class ToyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block_sparse_moe = ToyMoE()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block_sparse_moe(x)

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([ToyBlock()])

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            x = input_ids.float()
            if x.dim() == 2:
                x = x.unsqueeze(-1).repeat(1, 1, 4)
            for blk in self.model.layers:
                x = blk(x)
            return x

    model = ToyModel()
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


def test_equalise_residual_variance_supports_moe_expert_down_proj():
    class ToyExpert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down_proj = nn.Linear(4, 4, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(x)

    class ToyMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = nn.ModuleList([ToyExpert(), ToyExpert()])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.experts[0](x) * 2.0

    class ToyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block_sparse_moe = ToyMoE()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block_sparse_moe(x)

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([ToyBlock()])

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            x = input_ids.float()
            if x.dim() == 2:
                x = x.unsqueeze(-1).repeat(1, 1, 4)
            for blk in self.model.layers:
                x = blk(x)
            return x

    model = ToyModel()
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


def test_equalise_residual_variance_apply_true_scales_moe_expert_outputs():
    class ToyExpert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down_proj = nn.Linear(4, 4, bias=True)
            with torch.no_grad():
                self.down_proj.weight.fill_(2.0)
                self.down_proj.bias.fill_(1.0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(x)

    class ToyMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = nn.ModuleList([ToyExpert()])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.experts[0](x) * 2.0

    class ToyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block_sparse_moe = ToyMoE()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block_sparse_moe(x)

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([ToyBlock()])

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            x = input_ids.float()
            if x.dim() == 2:
                x = x.unsqueeze(-1).repeat(1, 1, 4)
            for blk in self.model.layers:
                x = blk(x)
            return x

    model = ToyModel()
    expert = model.model.layers[0].block_sparse_moe.experts[0]
    original_w = expert.down_proj.weight.detach().clone()
    original_b = expert.down_proj.bias.detach().clone()

    dataloader = [{"input_ids": torch.ones(2, 3, dtype=torch.long)}]
    out = equalise_residual_variance(
        model,
        dataloader,
        windows=1,
        tol=0.0,
        clamp_range=None,
        apply=True,
        allow_empty=False,
    )

    assert "block0.mlp" in out
    assert not torch.equal(expert.down_proj.weight, original_w)
    assert not torch.equal(expert.down_proj.bias, original_b)


def test_equalise_residual_variance_handles_non_iterable_moe_experts():
    class BadMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Non-iterable "experts" should be handled gracefully.
            self.experts = 123
            self.dummy = nn.Parameter(torch.tensor(0.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class ToyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block_sparse_moe = BadMoE()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block_sparse_moe(x)

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([ToyBlock()])

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            x = input_ids.float()
            if x.dim() == 2:
                x = x.unsqueeze(-1).repeat(1, 1, 4)
            for blk in self.model.layers:
                x = blk(x)
            return x

    model = ToyModel()
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
    assert out == {}


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


def test_variance_guard_resolves_moe_targets_with_module_dict_experts():
    class ToyExpert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w2 = nn.Linear(4, 4, bias=False)

    class ToyMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = nn.ModuleDict({"0": ToyExpert(), "1": ToyExpert()})

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class ToyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block_sparse_moe = ToyMoE()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([ToyBlock()])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    guard = VarianceGuard(policy={"scope": "ffn"})
    targets = guard._resolve_target_modules(ToyModel())  # noqa: SLF001
    assert "transformer.h.0.mlp.c_proj" in targets


def test_variance_guard_resolves_moe_targets_with_python_list_experts():
    class ToyExpert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w2 = nn.Linear(4, 4, bias=False)

    class ToyMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._experts = nn.ModuleList([ToyExpert(), ToyExpert()])
            self.experts = list(self._experts)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class ToyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block_sparse_moe = ToyMoE()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([ToyBlock()])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    guard = VarianceGuard(policy={"scope": "ffn"})
    targets = guard._resolve_target_modules(ToyModel())  # noqa: SLF001
    assert "transformer.h.0.mlp.c_proj" in targets


def test_variance_guard_resolves_moe_targets_with_down_proj_experts():
    class ToyExpert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down_proj = nn.Linear(4, 4, bias=False)

    class ToyMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = nn.ModuleList([ToyExpert(), ToyExpert()])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class ToyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = ToyMoE()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([ToyBlock()])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    guard = VarianceGuard(policy={"scope": "ffn"})
    targets = guard._resolve_target_modules(ToyModel())  # noqa: SLF001
    assert "transformer.h.0.mlp.c_proj" in targets


def test_equalise_residual_variance_supports_block_sparse_moe_blocks():
    class ToyExpert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w2 = nn.Linear(4, 4, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.w2(x)

    class ToyMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = nn.ModuleList([ToyExpert()])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.experts[0](x) * 2.0

    class ToyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block_sparse_moe = ToyMoE()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block_sparse_moe(x)

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


def test_iter_transformer_layers_fallback_handles_block_sparse_moe_layers():
    class ToyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = nn.Identity()
            self.block_sparse_moe = nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class WeirdWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.ModuleList([ToyBlock(), ToyBlock()])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for blk in self.backbone:
                x = blk(x)
            return x

    model = WeirdWrapper()
    layers = list(_iter_transformer_layers(model))
    assert len(layers) == 2
    assert all(hasattr(layer, "block_sparse_moe") for layer in layers)


def test_iter_transformer_layers_handles_nested_model_model_layers():
    class ToyInner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(1, 1), nn.Linear(1, 1)])

    class ToyOuter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = ToyInner()

    class ToyWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = ToyOuter()

    model = ToyWrapper()
    layers = list(_iter_transformer_layers(model))
    assert len(layers) == 2
