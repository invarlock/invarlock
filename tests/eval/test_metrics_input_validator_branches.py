from __future__ import annotations

import torch
import torch.nn as nn

from invarlock.core.exceptions import ValidationError
from invarlock.eval.metrics import InputValidator, MetricsConfig


def test_validate_model_noop_when_model_has_parameters() -> None:
    cfg = MetricsConfig(strict_validation=True, use_cache=False)
    model = nn.Linear(2, 2)
    InputValidator.validate_model(model, cfg)


def test_validate_tensor_raises_on_non_tensor_input() -> None:
    cfg = MetricsConfig(strict_validation=True, use_cache=False)
    with torch.no_grad():
        try:
            InputValidator.validate_tensor([1, 2, 3], "x", cfg)  # type: ignore[arg-type]
        except ValidationError as exc:
            assert exc.code == "E402"
        else:
            raise AssertionError("Expected ValidationError")
