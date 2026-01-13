from __future__ import annotations

import torch.nn as nn

from invarlock.guards.variance import equalise_residual_variance


class _BlockAttnMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.c_proj = nn.Linear(2, 2, bias=False)
        self.mlp = nn.Module()
        self.mlp.c_proj = nn.Linear(2, 2, bias=False)

    def forward(self, x):  # noqa: ANN001
        return x


class _BlockAttnNoProj(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()

    def forward(self, x):  # noqa: ANN001
        return x


class _BlockMlpNoProj(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Module()

    def forward(self, x):  # noqa: ANN001
        return x


class _BlockEmpty(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):  # noqa: ANN001
        return x


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList(
            [
                _BlockAttnMlp(),
                _BlockAttnNoProj(),
                _BlockMlpNoProj(),
                _BlockEmpty(),
            ]
        )

    def forward(self, input_ids):  # noqa: ANN001
        return input_ids


def test_equalise_residual_variance_covers_projection_presence_branches() -> None:
    model = _ToyModel()
    scales = equalise_residual_variance(
        model,
        [],
        windows=0,
        allow_empty=True,
        clamp_range=None,
    )
    assert isinstance(scales, dict)
