import math
import sys
import types

import numpy as np
import pytest
import torch

from invarlock.eval import metrics
from invarlock.eval.metrics import (
    DependencyError,
    DependencyManager,
    InputValidator,
    MetricsConfig,
    ResourceManager,
    ValidationError,
    _gini_vectorized,
    _mi_gini_optimized_cpu_path,
    bootstrap_confidence_interval,
)


class DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)


class DummyConfig:
    n_embd = 4
    n_layer = 2


class DummyModel(DummyModule):
    def __init__(self):
        super().__init__()
        self.config = DummyConfig()


@pytest.fixture(autouse=True)
def patch_virtual_memory(monkeypatch):
    class VM:
        total = 8 * 1024**3
        available = 6 * 1024**3

    monkeypatch.setattr(metrics.psutil, "virtual_memory", lambda: VM())


def test_bootstrap_confidence_interval_basic():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, size=128)
    lower, upper = bootstrap_confidence_interval(
        data, n_bootstrap=256, random_state=rng
    )

    assert lower < 0 < upper
    assert upper - lower > 0


@pytest.mark.parametrize(
    "samples",
    [
        np.empty((0,)),
        [[]],
    ],
)
def test_bootstrap_confidence_interval_validation(samples):
    from invarlock.eval.metrics import ValidationError as MValidationError

    with pytest.raises(MValidationError):
        bootstrap_confidence_interval(samples)


def test_metrics_config_validation(tmp_path):
    cfg = MetricsConfig(use_cache=False)
    assert cfg.oracle_windows == 16

    from invarlock.eval.metrics import ValidationError as MValidationError

    with pytest.raises(MValidationError):
        MetricsConfig(oracle_windows=-1)

    cfg_with_cache = MetricsConfig(use_cache=True, cache_dir=tmp_path)
    assert cfg_with_cache.cache_dir == tmp_path


def test_resource_manager_device_selection(monkeypatch):
    monkeypatch.setattr(metrics.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(metrics.torch.backends.mps, "is_available", lambda: False)

    manager = ResourceManager(MetricsConfig(force_cpu=True, use_cache=False))
    assert manager.device.type == "cpu"
    assert "system_total_gb" in manager.memory_info


def test_resource_manager_cpu_fallback(monkeypatch):
    manager = ResourceManager(MetricsConfig(use_cache=False, force_cpu=True))
    manager.device = torch.device("cuda")
    manager.memory_info = {"gpu_free_gb": 1.0}
    manager.config.cpu_fallback_threshold_gb = 0.1

    assert manager.should_use_cpu_fallback(0.5) is False
    assert manager.should_use_cpu_fallback(0.95) is True


def test_resource_manager_cleanup(monkeypatch):
    called = {}

    monkeypatch.setattr(metrics.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        metrics.torch.cuda,
        "empty_cache",
        lambda: called.setdefault("empty_cache", True),
    )

    class Props:
        total_memory = 4 * 1024**3

    monkeypatch.setattr(
        metrics.torch.cuda, "get_device_properties", lambda *_args, **_kwargs: Props()
    )
    monkeypatch.setattr(
        metrics.torch.cuda, "memory_allocated", lambda *_args, **_kwargs: 0
    )

    manager = ResourceManager(MetricsConfig(use_cache=False))
    manager.cleanup()
    assert called["empty_cache"] is True


def test_dependency_manager_missing_modules(monkeypatch):
    # Ensure optional modules are absent
    monkeypatch.delitem(sys.modules, "invarlock.eval.lens2_mi", raising=False)
    monkeypatch.delitem(sys.modules, "invarlock.eval.lens3", raising=False)

    dep_manager = DependencyManager()
    assert dep_manager.get_missing_dependencies()
    with pytest.raises(DependencyError):
        dep_manager.get_module("mi_scores")


def test_dependency_manager_available_modules(monkeypatch):
    lens2 = types.ModuleType("lens2_mi")
    lens2.mi_scores = lambda *args, **kwargs: None
    lens3 = types.ModuleType("lens3")
    lens3.scan_model_gains = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "invarlock.eval.lens2_mi", lens2)
    monkeypatch.setitem(sys.modules, "invarlock.eval.lens3", lens3)

    dep_manager = DependencyManager()
    assert dep_manager.is_available("mi_scores")
    assert dep_manager.get_module("scan_model_gains") is lens3.scan_model_gains


def test_input_validator_model_and_tensor(monkeypatch):
    config = MetricsConfig(use_cache=False)

    with pytest.raises(ValidationError):
        InputValidator.validate_model(object(), config)

    empty_model = torch.nn.Module()
    # Built-in Module has no parameters but should still pass validation.
    InputValidator.validate_model(empty_model, config)
    InputValidator.validate_model(
        empty_model, MetricsConfig(use_cache=False, strict_validation=False)
    )

    tensor = torch.tensor([1.0, float("nan"), float("inf")])
    sanitized = InputValidator.validate_tensor(
        tensor, "scores", MetricsConfig(use_cache=False, strict_validation=False)
    )
    assert torch.all(torch.isfinite(sanitized))


def test_input_validator_dataloader():
    config = MetricsConfig(use_cache=False)

    with pytest.raises(ValidationError):
        InputValidator.validate_dataloader(None, config)

    with pytest.raises(ValidationError):
        InputValidator.validate_dataloader([], config)

    InputValidator.validate_dataloader(
        [], MetricsConfig(use_cache=False, allow_empty_data=True)
    )


def test_gini_vectorized_and_mi_gini_cpu_path():
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    gini = _gini_vectorized(tensor)
    assert 0 <= gini <= 1

    feats = torch.rand(2, 5, 3)
    targ = torch.rand(5)
    config = MetricsConfig(use_cache=False, mi_gini_subsample_ratio=1.0)
    result = _mi_gini_optimized_cpu_path(feats, targ, max_per_layer=10, config=config)
    assert math.isnan(result) or isinstance(result, float)
