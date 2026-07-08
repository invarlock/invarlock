from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from invarlock.core.exceptions import ValidationError
from invarlock.eval.metrics import InputValidator, MetricsConfig
from invarlock.eval.metrics_activation import _perform_pre_eval_checks


def test_validate_model_raises_when_empty_model_in_strict_mode() -> None:
    class EmptyModel(nn.Module):
        def forward(self, **_kwargs):  # noqa: D401, ANN003
            return None

    cfg = MetricsConfig(strict_validation=True, use_cache=False)
    with pytest.raises(ValidationError) as exc_info:
        InputValidator.validate_model(EmptyModel(), cfg)
    assert exc_info.value.details["reason"] == "Model has no parameters"


def test_validate_model_raises_when_parameter_iteration_fails() -> None:
    class BrokenModel(nn.Module):
        def forward(self, **_kwargs):  # noqa: D401, ANN003
            return None

        def parameters(self, recurse: bool = True):
            raise RuntimeError("boom")

    cfg = MetricsConfig(strict_validation=False, use_cache=False)
    with pytest.raises(ValidationError) as exc_info:
        InputValidator.validate_model(BrokenModel(), cfg)
    assert exc_info.value.details["reason"] == "Model parameter iteration failed"
    assert exc_info.value.details["error"] == "boom"


def test_pre_eval_checks_warn_when_context_length_exceeded(tmp_path) -> None:
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(n_positions=2)

        def forward(self, input_ids=None, attention_mask=None, **_kwargs):  # noqa: ANN001
            return SimpleNamespace(logits=torch.zeros(1, 1))

    batch = {
        "input_ids": torch.zeros((1, 3), dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
    }
    dataloader = [batch]
    cfg = MetricsConfig(use_cache=False)
    result = _perform_pre_eval_checks(
        DummyModel(), dataloader, torch.device("cpu"), cfg
    )
    assert result is None
