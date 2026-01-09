from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn as nn

from invarlock.eval import metrics as metrics_mod


def test_resource_manager_falls_back_to_cpu_when_no_cuda_or_mps(monkeypatch) -> None:
    monkeypatch.setattr(metrics_mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(metrics_mod.torch.backends.mps, "is_available", lambda: False)

    cfg = metrics_mod.MetricsConfig(use_cache=False, strict_validation=False)
    rm = metrics_mod.ResourceManager(cfg)
    assert rm.device.type == "cpu"


def test_mi_gini_optimized_cpu_path_subsamples_when_n_gt_max(monkeypatch) -> None:
    max_per_layer = 5
    L, N, D = 2, 10, 4

    monkeypatch.setattr(metrics_mod.torch, "randperm", lambda n: torch.arange(n))

    class _DepMgr:
        def is_available(self, _name: str) -> bool:  # noqa: ANN001
            return True

        def get_module(self, _name: str):  # noqa: ANN001
            def _mi_scores_fn(feats: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
                assert feats.shape == (max_per_layer, D)
                assert targ.shape == (max_per_layer,)
                return torch.zeros_like(feats[0])

            return _mi_scores_fn

    monkeypatch.setattr(metrics_mod, "DependencyManager", lambda: _DepMgr())

    feats = torch.ones(L, N, D)
    targ = torch.arange(N)
    cfg = metrics_mod.MetricsConfig(use_cache=False, progress_bars=False, strict_validation=False)

    out = metrics_mod._mi_gini_optimized_cpu_path(
        feats, targ, max_per_layer=max_per_layer, config=cfg
    )
    assert math.isnan(out)


def test_locate_transformer_blocks_enhanced_catches_len_typeerror_and_uses_fallback() -> None:
    class _BadLen:
        def __len__(self) -> int:
            raise TypeError("boom")

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.Module()
            self.mlp = nn.Module()

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.h = _BadLen()
            self.block0 = _Block()

    blocks = metrics_mod._locate_transformer_blocks_enhanced(_Model())
    assert isinstance(blocks, list) and blocks


def test_extract_fc1_activations_returns_none_on_block_attr_error(monkeypatch) -> None:
    class _BadBlock:
        def __getattribute__(self, name: str):  # noqa: ANN001
            if name == "mlp":
                raise RuntimeError("boom")
            return super().__getattribute__(name)

    monkeypatch.setattr(metrics_mod, "_locate_transformer_blocks_enhanced", lambda _m: [_BadBlock()])

    cfg = metrics_mod.MetricsConfig(use_cache=False, progress_bars=False, strict_validation=False)
    out = metrics_mod._extract_fc1_activations(
        nn.Linear(2, 2),
        output=SimpleNamespace(hidden_states=[]),
        config=cfg,
    )
    assert out is None


def test_calculate_sigma_max_skips_when_dependency_missing() -> None:
    class _DepMgr:
        def is_available(self, _name: str) -> bool:  # noqa: ANN001
            return False

    cfg = metrics_mod.MetricsConfig(use_cache=False, progress_bars=False, strict_validation=False)
    out = metrics_mod._calculate_sigma_max(
        nn.Linear(2, 2),
        first_batch={"input_ids": [1]},
        dep_manager=_DepMgr(),
        config=cfg,
        device=torch.device("cpu"),
    )
    assert math.isnan(out)


def test_calculate_sigma_max_skips_when_first_batch_missing() -> None:
    class _DepMgr:
        def is_available(self, _name: str) -> bool:  # noqa: ANN001
            return True

    cfg = metrics_mod.MetricsConfig(use_cache=False, progress_bars=False, strict_validation=False)
    out = metrics_mod._calculate_sigma_max(
        nn.Linear(2, 2),
        first_batch=None,
        dep_manager=_DepMgr(),
        config=cfg,
        device=torch.device("cpu"),
    )
    assert math.isnan(out)


def test_calculate_head_energy_returns_nan_when_all_values_non_finite() -> None:
    cfg = metrics_mod.MetricsConfig(
        use_cache=False,
        progress_bars=False,
        strict_validation=False,
        nan_replacement=float("nan"),
        inf_replacement=float("nan"),
    )
    hidden_states_list = [torch.full((1, 1, 2, 1), float("nan"))]
    out = metrics_mod._calculate_head_energy(hidden_states_list, cfg)
    assert math.isnan(out)


def test_calculate_mi_gini_returns_nan_for_missing_dependency_and_missing_activations() -> None:
    class _DepMgrMissing:
        def is_available(self, _name: str) -> bool:  # noqa: ANN001
            return False

    class _DepMgrOk:
        def is_available(self, _name: str) -> bool:  # noqa: ANN001
            return True

    cfg = metrics_mod.MetricsConfig(use_cache=False, progress_bars=False, strict_validation=False)
    activation_data = {"fc1_activations": [torch.zeros(1, 1, 1, 1)], "targets": [torch.zeros(1, 1)]}

    out_missing = metrics_mod._calculate_mi_gini(
        nn.Linear(2, 2),
        activation_data=activation_data,
        dep_manager=_DepMgrMissing(),
        config=cfg,
        device=torch.device("cpu"),
    )
    assert math.isnan(out_missing)

    out_empty = metrics_mod._calculate_mi_gini(
        nn.Linear(2, 2),
        activation_data={"fc1_activations": [], "targets": []},
        dep_manager=_DepMgrOk(),
        config=cfg,
        device=torch.device("cpu"),
    )
    assert math.isnan(out_empty)

