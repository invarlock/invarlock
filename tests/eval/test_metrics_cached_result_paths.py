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


def test_calculate_metrics_explicit_config_applies_settings_and_uses_cache(
    monkeypatch,
) -> None:
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
    cfg = metrics_mod.MetricsConfig(
        use_cache=False,
        strict_validation=False,
        oracle_windows=5,
        device=torch.device("cpu"),
    )
    out = calculate_lens_metrics_for_model(
        model,
        dataloader,
        config=cfg,
    )
    assert out["sigma_max"] == 1.0


def test_calculate_metrics_explicit_config_uses_cache(monkeypatch) -> None:
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
    cfg = metrics_mod.MetricsConfig(use_cache=False, strict_validation=False)
    out = calculate_lens_metrics_for_model(model, dataloader, config=cfg)
    assert out["mi_gini"] == 0.0
