import pytest
import torch
import torch.nn as nn

from invarlock.guards.variance import VarianceGuard
from invarlock.guards.variance_scaling import (
    _emit_progress,
    equalise_residual_variance,
    iter_transformer_layers,
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
