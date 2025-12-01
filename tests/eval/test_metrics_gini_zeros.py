import torch

from invarlock.eval.metrics import _gini_vectorized


def test_gini_vectorized_all_zeros_returns_nan():
    v = torch.zeros(3, 4)
    g = _gini_vectorized(v)
    assert isinstance(g, float) and (g != g)  # NaN
