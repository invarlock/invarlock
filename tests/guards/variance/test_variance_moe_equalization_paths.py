import torch
import torch.nn as nn

from invarlock.guards.variance import VarianceGuard
from invarlock.guards.variance_scaling import (
    equalise_residual_variance,
    iter_transformer_layers,
)


def test_equalise_residual_variance_scale_bias_false_skips_moe_bias_scaling():
    class ExpertsContainer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down_proj = nn.Linear(4, 4, bias=True)
            with torch.no_grad():
                self.down_proj.weight.fill_(2.0)
                self.down_proj.bias.fill_(3.0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(x)

    class ToyMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Experts are stored as a container with a down_proj module. This exercises
            # the fused expert fallback resolution path.
            self.experts = ExpertsContainer()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Force alpha != 1 so we take the scaling branch.
            return self.experts(x) * 2.0

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
    expert = model.model.layers[0].block_sparse_moe.experts.down_proj
    original_w = expert.weight.detach().clone()
    original_b = expert.bias.detach().clone()

    dataloader = [{"input_ids": torch.ones(2, 3, dtype=torch.long)}]
    out = equalise_residual_variance(
        model,
        dataloader,
        windows=1,
        tol=0.0,
        clamp_range=None,
        apply=True,
        allow_empty=False,
        scale_bias=False,
    )

    assert "block0.mlp" in out
    assert not torch.equal(expert.weight, original_w)
    assert torch.equal(expert.bias, original_b)


def test_equalise_residual_variance_moe_fused_resolution_continues_on_bad_proj():
    class WeirdExperts:
        # Not an nn.Module; exercise getattr-based fallback resolution.
        def __init__(self) -> None:
            self.w2 = 123  # Not a tensor; should be ignored via candidate=None path.
            self.down_proj = nn.Parameter(torch.ones(4))  # 1D tensor; dim not in (2,3).

    class ToyMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = WeirdExperts()
            self.dummy = nn.Parameter(torch.tensor(0.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 2.0

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
    # Resolution should not crash; no hooks/scales expected from these bogus projections.
    assert out == {}


def test_equalise_residual_variance_handles_moe_proj_with_non_tensor_weight():
    class FakeProj:
        def __init__(self) -> None:
            self.weight = "not_a_tensor"
            self.bias = "not_a_tensor"

    class ToyExpert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down_proj = FakeProj()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class ToyMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = nn.ModuleList([ToyExpert()])
            self.dummy = nn.Parameter(torch.tensor(0.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 2.0

    class ToyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = ToyMoE()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.mlp(x)

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = nn.Parameter(torch.tensor(0.0))
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
        apply=True,
        allow_empty=False,
    )
    # Should not crash even if the expert "projection" has a non-tensor weight.
    assert "block0.mlp" in out


def test_equalise_residual_variance_direct_targets_skip_no_weight_modules():
    class NoWeightProj(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 2.0

    class ToyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.Module()
            self.attn.c_proj = NoWeightProj()
            self.mlp = nn.Module()
            self.mlp.c_proj = NoWeightProj()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.mlp.c_proj(self.attn.c_proj(x))

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = nn.Parameter(torch.tensor(0.0))
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList([ToyBlock()])

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            x = input_ids.float()
            if x.dim() == 2:
                x = x.unsqueeze(-1).repeat(1, 1, 4)
            for blk in self.transformer.h:
                x = blk(x)
            return x

    out = equalise_residual_variance(
        ToyModel(),
        [{"input_ids": torch.ones(2, 3, dtype=torch.long)}],
        windows=1,
        tol=0.0,
        clamp_range=None,
        apply=True,
        allow_empty=False,
    )

    assert out == {}


def test_equalise_residual_variance_moe_quantized_experts_are_not_reported_applied():
    class QuantizedProj(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.zeros(4, 4, dtype=torch.int8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class ToyExpert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down_proj = QuantizedProj()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(x)

    class ToyMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = nn.ModuleList([ToyExpert()])
            self.dummy = nn.Parameter(torch.tensor(0.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 2.0

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

    out = equalise_residual_variance(
        ToyModel(),
        [{"input_ids": torch.ones(2, 3, dtype=torch.long)}],
        windows=1,
        tol=0.0,
        clamp_range=None,
        apply=True,
        allow_empty=False,
    )

    assert out == {}


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


def test_variance_guard_resolves_fused_moe_targets_with_expert_weight_tensor():
    class FusedExperts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down_proj = nn.Parameter(torch.ones(2, 4, 4))

    class ToyMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = FusedExperts()

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
    layers = list(iter_transformer_layers(model))
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
    layers = list(iter_transformer_layers(model))
    assert len(layers) == 2
