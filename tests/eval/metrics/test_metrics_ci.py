from pathlib import Path

import numpy as np
import pytest
import torch.nn as nn

from invarlock.eval.metrics import (
    MetricsConfig,
    ResourceManager,
    bootstrap_confidence_interval,
)


def test_metrics_config_validation(tmp_path: Path) -> None:
    cfg = MetricsConfig(cache_dir=tmp_path, use_cache=True, max_tokens=512)
    assert cfg.cache_dir == tmp_path
    assert cfg.max_tokens == 512

    from invarlock.eval.metrics import ValidationError as MValidationError

    with pytest.raises(MValidationError):
        MetricsConfig(max_tokens=0)

    with pytest.raises(MValidationError):
        MetricsConfig(memory_limit_gb=-1.0)


def test_resource_manager_cpu_estimates(tmp_path: Path) -> None:
    cfg = MetricsConfig(cache_dir=tmp_path, use_cache=True, force_cpu=True)
    manager = ResourceManager(cfg)

    model = nn.Linear(8, 8)
    estimate = manager.estimate_memory_usage(model, batch_size=2, seq_length=16)
    assert estimate > 0.0
    assert manager.should_use_cpu_fallback(estimate) is False


def test_bootstrap_confidence_interval_contains_mean() -> None:
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=1.0, scale=0.05, size=512)

    lo, hi = bootstrap_confidence_interval(samples, n_bootstrap=300, alpha=0.05)

    sample_mean = float(np.mean(samples))
    assert lo < sample_mean < hi
    assert hi - lo < 0.2  # interval should be reasonably tight for low-variance data
