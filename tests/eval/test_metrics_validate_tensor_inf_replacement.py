import torch

from invarlock.eval.metrics import InputValidator, MetricsConfig


def test_validate_tensor_inf_replacement_nonstrict():
    t = torch.tensor([float("inf"), -float("inf"), 1.0])
    out = InputValidator.validate_tensor(t, "t", MetricsConfig(strict_validation=False))
    assert torch.isfinite(out).all()
