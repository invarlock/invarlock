from __future__ import annotations

import torch
import torch.nn as nn

from invarlock.eval import metrics as metrics_mod
from invarlock.eval.metrics import calculate_lens_metrics_for_model


def _patch_cache_to_return(monkeypatch, payload: dict[str, float]) -> None:
    monkeypatch.setattr(
        metrics_mod.ResultCache,
        "_get_cache_key",
        lambda _self, *_a, **_k: "cache-key",  # noqa: ARG005
    )
    monkeypatch.setattr(
        metrics_mod.ResultCache,
        "get",
        lambda _self, _key: payload,  # noqa: ARG005
    )


def test_calculate_metrics_config_none_applies_overrides_and_uses_cache(
    monkeypatch,
) -> None:
    RealConfig = metrics_mod.MetricsConfig
    monkeypatch.setattr(
        metrics_mod,
        "MetricsConfig",
        lambda: RealConfig(use_cache=False, strict_validation=False),
    )
    _patch_cache_to_return(
        monkeypatch, {"sigma_max": 1.0, "head_energy": 2.0, "mi_gini": 3.0}
    )

    model = nn.Linear(2, 2)
    dataloader = [
        {
            "input_ids": torch.zeros((1, 2), dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }
    ]
    out = calculate_lens_metrics_for_model(
        model,
        dataloader,
        config=None,
        oracle_windows=5,
        device=torch.device("cpu"),
    )
    assert out["sigma_max"] == 1.0


def test_calculate_metrics_config_none_without_overrides_uses_cache(
    monkeypatch,
) -> None:
    RealConfig = metrics_mod.MetricsConfig
    monkeypatch.setattr(
        metrics_mod,
        "MetricsConfig",
        lambda: RealConfig(use_cache=False, strict_validation=False),
    )
    _patch_cache_to_return(
        monkeypatch, {"sigma_max": 0.0, "head_energy": 0.0, "mi_gini": 0.0}
    )

    model = nn.Linear(2, 2)
    dataloader = [
        {
            "input_ids": torch.zeros((1, 1), dtype=torch.long),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
        }
    ]
    out = calculate_lens_metrics_for_model(model, dataloader, config=None)
    assert out["mi_gini"] == 0.0
