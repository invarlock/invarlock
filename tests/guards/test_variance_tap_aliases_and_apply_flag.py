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
