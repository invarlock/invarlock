import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from invarlock.eval import metrics as M


def test_bootstrap_ci_errors_and_success():
    # Error cases
    import pytest

    with pytest.raises(M.ValidationError):
        M.bootstrap_confidence_interval([], n_bootstrap=10)
    import pytest

    with pytest.raises(M.ValidationError):
        M.bootstrap_confidence_interval([1, 2, 3], alpha=1.5)
    # Success
    lo, hi = M.bootstrap_confidence_interval(
        [1.0, 2.0, 3.0], n_bootstrap=32, random_state=np.random.default_rng(0)
    )
    assert lo <= hi


def test_input_validator_branches():
    class NoParam(torch.nn.Module):
        def parameters(self):  # noqa: D401
            if False:
                yield from ()  # pragma: no cover
            return iter([])

    cfg = M.MetricsConfig(strict_validation=False)
    M.InputValidator.validate_model(NoParam(), cfg)

    # Dataloader empty with allow_empty_data=True
    class Empty:
        def __iter__(self):  # noqa: D401
            return iter(())

    cfg2 = M.MetricsConfig(allow_empty_data=True)
    M.InputValidator.validate_dataloader(Empty(), cfg2)


def test_gini_and_cache_and_blocks():
    # Gini on zero vector returns NaN
    v = torch.zeros(3, 4)
    val = M._gini_vectorized(v)
    assert math.isnan(val)

    # ResultCache get/set/clear
    cfg = M.MetricsConfig(use_cache=True)
    cache = M.ResultCache(cfg)
    dummy_model = torch.nn.Linear(2, 2)
    key = cache._get_cache_key(dummy_model, [1, 2], cfg)
    assert cache.get(key) is None
    cache.set(key, {"sigma_max": 0.0})
    assert cache.get(key) == {"sigma_max": 0.0}
    cache.clear()
    assert cache.get(key) is None

    # Locate transformer blocks enhanced
    class Dummy:
        def __init__(self):
            self.transformer = SimpleNamespace(h=[object()])

    blocks = M._locate_transformer_blocks_enhanced(Dummy())
    assert isinstance(blocks, list)


def test_compute_parameter_deltas():
    m1 = torch.nn.Linear(3, 3)
    m2 = torch.nn.Linear(3, 3)
    with torch.no_grad():
        m2.weight.add_(1.0)
    d = M.compute_parameter_deltas(m1, m2)
    assert d["params_changed"] > 0


def test_unified_compute_ppl_with_window_object():
    # Min window object to satisfy API
    window = SimpleNamespace(input_ids=[[1, 2, 3, 4]], attention_masks=[[1, 1, 1, 1]])

    class DummyLM:
        def __init__(self, vocab=8):
            self.out = torch.nn.Linear(4, vocab)

        def parameters(self):
            yield from self.out.parameters()

        def eval(self):  # pragma: no cover
            return self

        def __call__(self, input_ids=None, attention_mask=None, return_dict=True):
            B, T = input_ids.shape
            logits = self.out(torch.zeros(B, T, 4))
            return SimpleNamespace(logits=logits)

    ppl = M.compute_ppl(DummyLM(), None, window)
    assert isinstance(ppl, float) and ppl >= 1.0


def test_locate_transformer_blocks_patterns():
    class ModelH:
        def __init__(self):
            self.h = [object()]

    class BaseModel:
        def __init__(self):
            self.base_model = ModelH()

    class Wrapped:
        def __init__(self):
            self.transformer = ModelH()

    assert isinstance(M._locate_transformer_blocks_enhanced(Wrapped()), list)
    assert isinstance(M._locate_transformer_blocks_enhanced(BaseModel()), list)


def test_unified_compute_ppl_window_no_valid_tokens_raises():
    window = SimpleNamespace(input_ids=[[1, 1, 1]], attention_masks=[[0, 0, 0]])

    class DummyLM:
        def __init__(self, vocab=8):
            self.out = torch.nn.Linear(4, vocab)

        def parameters(self):
            yield from self.out.parameters()

        def eval(self):  # pragma: no cover
            return self

        def __call__(self, input_ids=None, attention_mask=None, return_dict=True):
            B, T = input_ids.shape
            logits = self.out(torch.zeros(B, T, 4))
            return SimpleNamespace(logits=logits)

    with pytest.raises(M.ValidationError):
        _ = M.compute_ppl(DummyLM(), None, window)
