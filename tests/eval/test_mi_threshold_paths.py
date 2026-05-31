from __future__ import annotations

import types

import numpy as np
import torch
import torch.nn as nn

import invarlock.eval.probes.importance as mi_mod


class _MiBlock(nn.Module):
    def __init__(self, hidden: int, mlp_dim: int) -> None:
        super().__init__()
        self.mlp = types.SimpleNamespace(c_fc=nn.Linear(hidden, mlp_dim, bias=False))


class _MiModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(n_layer=1)
        self.embedding = nn.Embedding(8, 1)
        self.transformer = types.SimpleNamespace(
            h=nn.ModuleList([_MiBlock(1, 1), _MiBlock(1, 1)])
        )
        self.out = nn.Linear(1, 8, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for block in self.transformer.h:
            x = block.mlp.c_fc(x)
        return types.SimpleNamespace(logits=self.out(x))


def test_mi_probe_covers_extra_layer_skip_and_sample_subselection(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        mi_mod,
        "mutual_info_regression",
        lambda features, targets, random_state=42: np.asarray(
            [float(features.mean()) + float(targets.mean()) * 0.0]
        ),
    )

    scores = mi_mod.compute_neuron_mi_scores(
        _MiModel(),
        [torch.randint(0, 8, (1, 10002), dtype=torch.long)],
        oracle_windows=1,
        device="cpu",
    )

    assert len(scores) == 1
    assert tuple(scores[0].shape) == (1,)
