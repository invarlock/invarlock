import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from invarlock.eval.metrics import (
    DependencyError,
    DependencyManager,
    InputValidator,
    MetricsConfig,
    PerplexityStatus,
    ResourceManager,
    ValidationError,
    _forward_loss_causal,
    bootstrap_confidence_interval,
    validate_perplexity,
)
from invarlock.eval.metrics import ValidationError as MValidationError


def test_bootstrap_confidence_interval_valid_and_errors():
    data = [1.0, 2.0, 3.0, 4.0]
    lo, hi = bootstrap_confidence_interval(
        data, n_bootstrap=200, alpha=0.1, random_state=np.random.default_rng(0)
    )
    assert lo <= hi

    from invarlock.eval.metrics import ValidationError as MValidationError

    with pytest.raises(MValidationError):
        bootstrap_confidence_interval([], n_bootstrap=10)
    with pytest.raises(MValidationError):
        bootstrap_confidence_interval([1.0], n_bootstrap=0)
    with pytest.raises(MValidationError):
        bootstrap_confidence_interval([1.0], alpha=0.0)
    with pytest.raises(MValidationError):
        bootstrap_confidence_interval(np.ones((2, 2)))


def test_metrics_config_validation_and_cache(tmp_path):
    # Valid config creates cache dir when use_cache
    cfg = MetricsConfig(use_cache=True)
    assert cfg.cache_dir is not None and cfg.cache_dir.exists()

    # Invalid arguments raise
    with pytest.raises(MValidationError):
        MetricsConfig(oracle_windows=-1)
    with pytest.raises(MValidationError):
        MetricsConfig(max_tokens=0)
    with pytest.raises(MValidationError):
        MetricsConfig(memory_limit_gb=0)


def test_resource_manager_device_and_estimation(monkeypatch):
    cfg = MetricsConfig(force_cpu=True)
    rm = ResourceManager(cfg)
    assert rm.device.type == "cpu"

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)
            self.config = SimpleNamespace(n_embd=4, n_layer=2)

    est = rm.estimate_memory_usage(Tiny(), batch_size=2, seq_length=8)
    assert est > 0
    # Fallback check on CPU returns False
    assert rm.should_use_cpu_fallback(est) is False

    # cleanup should not error without CUDA
    rm.cleanup()


def test_dependency_manager_missing_modules():
    dm = DependencyManager()
    # In minimal env, likely missing both; assertions are robust to either
    missing = dict(dm.get_missing_dependencies())
    for key in ("lens2_mi", "lens3"):
        if key in missing:
            with pytest.raises(DependencyError):
                dm.get_module("scan_model_gains" if key == "lens3" else "mi_scores")


def test_input_validator_strict_and_lenient():
    v = InputValidator()

    class ModelNoParams(nn.Module):
        def __init__(self):
            super().__init__()

    with pytest.raises(ValidationError):
        v.validate_model(object(), MetricsConfig())

    # Both strict and lenient paths should not raise here since counting
    # parameters may be guarded by fallback; tensor validation covers raising
    v.validate_model(ModelNoParams(), MetricsConfig(strict_validation=True))
    v.validate_model(ModelNoParams(), MetricsConfig(strict_validation=False))

    # Dataloader validation
    with pytest.raises(ValidationError):
        v.validate_dataloader(None, MetricsConfig())
    with pytest.raises(ValidationError):
        v.validate_dataloader([], MetricsConfig(allow_empty_data=False))
    # Allow empty
    v.validate_dataloader([], MetricsConfig(allow_empty_data=True))

    # Tensor validation strict vs lenient
    t = torch.tensor([1.0, float("nan"), float("inf"), -float("inf")])
    with pytest.raises(ValidationError):
        v.validate_tensor(t, "t", MetricsConfig(strict_validation=True))
    t2 = v.validate_tensor(t, "t", MetricsConfig(strict_validation=False))
    assert torch.isfinite(t2).all()


@pytest.mark.parametrize(
    "ppl,expected",
    [
        (25.0, PerplexityStatus.EXCELLENT),
        (75.0, PerplexityStatus.GOOD),
        (150.0, PerplexityStatus.ACCEPTABLE),
        (300.0, PerplexityStatus.POOR),
        (800.0, PerplexityStatus.UNUSABLE),
    ],
)
def test_validate_perplexity_levels(ppl, expected):
    ok, status, _ = validate_perplexity(ppl)
    assert status == expected
    assert ok or status in (PerplexityStatus.UNUSABLE,)


def test_validate_perplexity_thresholds_and_invalids():
    assert validate_perplexity(float("nan"))[0] is False
    assert validate_perplexity(0.5)[0] is False
    # Warning branch
    ok, status, msg = validate_perplexity(
        250.0, vocab_size=None, warn_threshold=200.0, error_threshold=2000.0
    )
    assert ok is True and "warning" in msg.lower()
    # Error branch (unless allow_high)
    ok, status, msg = validate_perplexity(
        5000.0, vocab_size=1000, warn_threshold=200.0, error_threshold=2000.0
    )
    assert ok is False
    ok, status, msg = validate_perplexity(5000.0, vocab_size=1000, allow_high=True)
    assert ok is True


def test_forward_loss_causal_variants():
    B, T, V = 2, 4, 8
    input_ids = torch.randint(0, V, (B, T))
    labels = input_ids.clone()

    class Out:
        def __init__(self, loss, logits=None):
            self.loss = torch.tensor(loss, dtype=torch.float32)
            self.logits = logits

    class DictModel(nn.Module):
        def forward(self, **kwargs):  # noqa: D401
            return Out(loss=1.23, logits=torch.randn(B, T, V))

    class TupleModel(nn.Module):
        def forward(self, input_ids=None, attention_mask=None, labels=None):  # noqa: D401
            return (torch.randn(B, T, V),)

    # Dict-style returning loss
    loss, logits = _forward_loss_causal(DictModel(), input_ids, labels=labels)
    assert math.isfinite(loss) and logits is not None and logits.shape == (B, T, V)

    # Tuple-style without loss
    loss2, logits2 = _forward_loss_causal(TupleModel(), input_ids, labels=labels)
    assert math.isfinite(loss2) and isinstance(logits2, torch.Tensor)
