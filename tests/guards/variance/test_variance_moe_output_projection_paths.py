import torch
import torch.nn as nn

from invarlock.guards.variance_scaling import equalise_residual_variance


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
            self.anchor = nn.Parameter(torch.ones(1))
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
            self.anchor = nn.Parameter(torch.ones(1))
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


def test_equalise_residual_variance_handles_dim_fallback_for_moe_candidates() -> None:
    class _BrokenDimTensor(torch.Tensor):
        @staticmethod
        def __new__(cls):
            return torch.Tensor._make_subclass(cls, torch.ones(2, 2), False)

        def dim(self) -> int:
            raise RuntimeError("dim boom")

        @property
        def ndim(self) -> int:
            return 2

    class _Proj:
        def __init__(self) -> None:
            self.weight = _BrokenDimTensor()

    class _Experts:
        def __init__(self) -> None:
            self.down_proj = _Proj()

    class ToyMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = _Experts()

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

    out = equalise_residual_variance(
        ToyModel(),
        [{"input_ids": torch.ones(2, 3, dtype=torch.long)}],
        windows=1,
        tol=0.0,
        device="cpu",
        clamp_range=None,
        apply=False,
        allow_empty=False,
    )
    assert "block0.mlp" in out


def test_equalise_residual_variance_skips_empty_moe_outputs() -> None:
    class ToyMoE(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

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

    assert (
        equalise_residual_variance(
            ToyModel(),
            [{"input_ids": torch.ones(2, 3, dtype=torch.long)}],
            windows=1,
            tol=0.0,
            device="cpu",
            clamp_range=None,
            apply=False,
            allow_empty=False,
        )
        == {}
    )


def test_equalise_residual_variance_handles_moe_outputs_disappearing_after_hook() -> (
    None
):
    class ToyExpert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down_proj = nn.Linear(4, 4, bias=False)

    class ToyMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = [ToyExpert()]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.experts = object()
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
            self.anchor = nn.Parameter(torch.ones(1))
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([ToyBlock()])

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            x = input_ids.float()
            if x.dim() == 2:
                x = x.unsqueeze(-1).repeat(1, 1, 4)
            for blk in self.model.layers:
                x = blk(x)
            return x

    assert (
        equalise_residual_variance(
            ToyModel(),
            [{"input_ids": torch.ones(2, 3, dtype=torch.long)}],
            windows=1,
            tol=0.0,
            device="cpu",
            clamp_range=None,
            apply=False,
            allow_empty=False,
        )
        == {}
    )


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


def test_equalise_residual_variance_supports_fused_moe_expert_weight_tensors():
    class FusedExperts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Pack expert output projections into a single fused tensor:
            # [n_experts, out, in].
            self.down_proj = nn.Parameter(torch.ones(2, 4, 4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Always route to expert 0 for determinism.
            w = self.down_proj[0]
            return torch.matmul(x, w.t())

    class ToyMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = FusedExperts()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.experts(x) * 2.0

    class ToyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = ToyMoE()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.mlp(x)

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
    original = model.model.layers[0].mlp.experts.down_proj.detach().clone()

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
    assert not torch.equal(model.model.layers[0].mlp.experts.down_proj, original)


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
