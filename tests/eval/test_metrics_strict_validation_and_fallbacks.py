from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from invarlock.eval import metrics as metrics_mod


def test_resource_manager_falls_back_to_cpu_when_no_cuda_or_mps(monkeypatch) -> None:
    monkeypatch.setattr(metrics_mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        metrics_mod.torch.backends,
        "mps",
        SimpleNamespace(is_available=lambda: False),
        raising=False,
    )

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
    cfg = metrics_mod.MetricsConfig(
        use_cache=False, progress_bars=False, strict_validation=False
    )

    out = metrics_mod._mi_gini_optimized_cpu_path(
        feats, targ, max_per_layer=max_per_layer, config=cfg
    )
    assert math.isnan(out)


def test_locate_transformer_blocks_enhanced_catches_len_typeerror_and_uses_fallback() -> (
    None
):
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

    monkeypatch.setattr(
        metrics_mod, "_locate_transformer_blocks_enhanced", lambda _m: [_BadBlock()]
    )

    cfg = metrics_mod.MetricsConfig(
        use_cache=False, progress_bars=False, strict_validation=False
    )
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

    cfg = metrics_mod.MetricsConfig(
        use_cache=False, progress_bars=False, strict_validation=False
    )
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

    cfg = metrics_mod.MetricsConfig(
        use_cache=False, progress_bars=False, strict_validation=False
    )
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


def test_calculate_mi_gini_returns_nan_for_missing_dependency_and_missing_activations() -> (
    None
):
    class _DepMgrMissing:
        def is_available(self, _name: str) -> bool:  # noqa: ANN001
            return False

    class _DepMgrOk:
        def is_available(self, _name: str) -> bool:  # noqa: ANN001
            return True

    cfg = metrics_mod.MetricsConfig(
        use_cache=False, progress_bars=False, strict_validation=False
    )
    activation_data = {
        "fc1_activations": [torch.zeros(1, 1, 1, 1)],
        "targets": [torch.zeros(1, 1)],
    }

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


def test_calculate_lens_metrics_unwraps_base_model_and_returns_on_no_hidden_states(
    monkeypatch,
) -> None:
    class _Inner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

        def forward(self, **_kwargs):  # noqa: ANN001
            return SimpleNamespace(hidden_states=[], logits=torch.zeros(1, 1, 1))

    class _Wrapped(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_model = _Inner()
            self.p = nn.Parameter(torch.zeros(1))

        def forward(self, **_kwargs):  # noqa: ANN001
            return self.base_model(**_kwargs)

    monkeypatch.setattr(
        metrics_mod,
        "_collect_activations",
        lambda *_a, **_k: {
            "hidden_states": [],
            "fc1_activations": [],
            "targets": [],
            "first_batch": None,
        },
    )

    cfg = metrics_mod.MetricsConfig(
        use_cache=False, progress_bars=False, strict_validation=False
    )
    dataloader = [{"input_ids": torch.zeros(1, 2, dtype=torch.long)}]
    out = metrics_mod.calculate_lens_metrics_for_model(
        _Wrapped(), dataloader, config=cfg
    )
    assert set(out) >= {"sigma_max", "head_energy", "mi_gini"}


def test_calculate_lens_metrics_strict_validation_raises_on_activation_failure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        metrics_mod,
        "_collect_activations",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    cfg = metrics_mod.MetricsConfig(
        use_cache=False, progress_bars=False, strict_validation=True
    )
    dataloader = [{"input_ids": torch.zeros(1, 2, dtype=torch.long)}]
    with pytest.raises(metrics_mod.MetricsError):
        metrics_mod.calculate_lens_metrics_for_model(
            nn.Linear(2, 2), dataloader, config=cfg
        )


def test_calculate_lens_metrics_non_strict_continues_on_activation_failure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        metrics_mod,
        "_collect_activations",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    cfg = metrics_mod.MetricsConfig(
        use_cache=False, progress_bars=False, strict_validation=False
    )
    dataloader = [{"input_ids": torch.zeros(1, 2, dtype=torch.long)}]
    out = metrics_mod.calculate_lens_metrics_for_model(
        nn.Linear(2, 2), dataloader, config=cfg
    )
    assert (
        math.isnan(out["sigma_max"])
        and math.isnan(out["head_energy"])
        and math.isnan(out["mi_gini"])
    )


def test_perform_pre_eval_checks_handles_missing_context_attr_and_no_warning_branch() -> (
    None
):
    class _Cfg:
        n_positions = None
        max_position_embeddings = None

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))
            self.config = _Cfg()

        def forward(self, **_kwargs):  # noqa: ANN001
            return SimpleNamespace(logits=torch.zeros(1, 1, 1))

    cfg = metrics_mod.MetricsConfig(
        use_cache=False, progress_bars=False, strict_validation=False
    )
    dataloader = [{"input_ids": torch.zeros(1, 2, dtype=torch.long)}]
    metrics_mod._perform_pre_eval_checks(_Model(), dataloader, torch.device("cpu"), cfg)


def test_perform_pre_eval_checks_skips_warning_when_seq_len_within_model_limit() -> (
    None
):
    class _Cfg:
        n_positions = 10
        max_position_embeddings = None

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))
            self.config = _Cfg()

        def forward(self, **_kwargs):  # noqa: ANN001
            return SimpleNamespace(logits=torch.zeros(1, 1, 1))

    cfg = metrics_mod.MetricsConfig(
        use_cache=False, progress_bars=False, strict_validation=False
    )
    dataloader = [{"input_ids": torch.zeros(1, 5, dtype=torch.long)}]
    metrics_mod._perform_pre_eval_checks(_Model(), dataloader, torch.device("cpu"), cfg)


def test_extract_fc1_activations_skips_blocks_without_mlp(monkeypatch) -> None:
    class _Block(nn.Module):
        pass

    monkeypatch.setattr(
        metrics_mod, "_locate_transformer_blocks_enhanced", lambda _m: [_Block()]
    )
    cfg = metrics_mod.MetricsConfig(
        use_cache=False, progress_bars=False, strict_validation=False
    )
    out = metrics_mod._extract_fc1_activations(
        nn.Linear(2, 2),
        output=SimpleNamespace(hidden_states=[]),
        config=cfg,
    )
    assert out is None


def test_calculate_sigma_max_all_non_finite_triggers_nan_branch() -> None:
    class _DepMgr:
        def is_available(self, _name: str) -> bool:  # noqa: ANN001
            return True

        def get_module(self, _name: str):  # noqa: ANN001
            def _scan(_model, _batch):  # noqa: ANN001
                class _Gains:
                    values = [float("nan")]

                    def __len__(self) -> int:
                        return 1

                return _Gains()

            return _scan

    cfg = metrics_mod.MetricsConfig(
        use_cache=False,
        progress_bars=False,
        strict_validation=False,
        nan_replacement=float("nan"),
    )
    out = metrics_mod._calculate_sigma_max(
        nn.Linear(2, 2),
        first_batch={"input_ids": torch.zeros(1, 2, dtype=torch.long)},
        dep_manager=_DepMgr(),
        config=cfg,
        device=torch.device("cpu"),
    )
    assert math.isnan(out)


def test_calculate_mi_gini_oom_calls_empty_cache_when_cuda_available(
    monkeypatch,
) -> None:
    called = {"empty_cache": 0}
    monkeypatch.setattr(metrics_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        metrics_mod.torch.cuda,
        "empty_cache",
        lambda: called.__setitem__("empty_cache", called["empty_cache"] + 1),
    )
    monkeypatch.setattr(
        metrics_mod, "_mi_gini_optimized_cpu_path", lambda *_a, **_k: float("nan")
    )

    class _DepMgr:
        def is_available(self, _name: str) -> bool:  # noqa: ANN001
            return True

        def get_module(self, _name: str):  # noqa: ANN001
            def _oom(_feats, _targ):  # noqa: ANN001
                raise RuntimeError("out of memory")

            return _oom

    cfg = metrics_mod.MetricsConfig(
        use_cache=False, progress_bars=False, strict_validation=False
    )
    activation_data = {
        "fc1_activations": [torch.zeros(1, 1, 2, 1)],
        "targets": [torch.zeros(1, 2, dtype=torch.long)],
    }
    assert math.isnan(
        metrics_mod._calculate_mi_gini(
            nn.Linear(2, 2),
            activation_data=activation_data,
            dep_manager=_DepMgr(),
            config=cfg,
            device=torch.device("cpu"),
        )
    )
    assert called["empty_cache"] == 1


def test_calculate_mi_gini_runtime_error_non_oom_takes_raise_path(monkeypatch) -> None:
    class _DepMgr:
        def is_available(self, _name: str) -> bool:  # noqa: ANN001
            return True

        def get_module(self, _name: str):  # noqa: ANN001
            def _boom(_feats, _targ):  # noqa: ANN001
                raise RuntimeError("boom")

            return _boom

    cfg = metrics_mod.MetricsConfig(
        use_cache=False, progress_bars=False, strict_validation=False
    )
    activation_data = {
        "fc1_activations": [torch.zeros(1, 1, 2, 1)],
        "targets": [torch.zeros(1, 2, dtype=torch.long)],
    }
    assert math.isnan(
        metrics_mod._calculate_mi_gini(
            nn.Linear(2, 2),
            activation_data=activation_data,
            dep_manager=_DepMgr(),
            config=cfg,
            device=torch.device("cpu"),
        )
    )


def test_validate_metrics_environment_reports_missing_modules(monkeypatch) -> None:
    class _DepMgr:
        def __init__(self):  # noqa: D401
            self.available_modules = {"ok": object()}
            self.missing_modules = [("missing", "boom")]

        def get_missing_dependencies(self):  # noqa: ANN001
            return self.missing_modules

    monkeypatch.setattr(metrics_mod, "DependencyManager", _DepMgr)
    assert metrics_mod.validate_metrics_environment() is True


def test_validate_perplexity_hits_poor_status_branch() -> None:
    ok, status, _msg = metrics_mod.validate_perplexity(200.0)
    assert ok is True and status == metrics_mod.PerplexityStatus.POOR


def test_forward_loss_causal_handles_models_without_return_dict() -> None:
    class _Out:
        def __init__(self) -> None:
            self.loss = torch.tensor(1.0)
            self.logits = torch.zeros(1, 2, 3)

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, attention_mask=None, labels=None):  # noqa: ANN001
            return _Out()

    loss, logits = metrics_mod._forward_loss_causal(
        _Model(),
        input_ids=torch.zeros(1, 2, dtype=torch.long),
        labels=torch.zeros(1, 2, dtype=torch.long),
    )
    assert isinstance(loss, float) and logits is not None
