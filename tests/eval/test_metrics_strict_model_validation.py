from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from invarlock.eval.metrics import (
    InputValidator,
    MetricsConfig,
    _perform_pre_eval_checks,
)


def test_validate_model_logs_when_empty_model_in_strict_mode(monkeypatch) -> None:
    class EmptyModel(nn.Module):
        def forward(self, **_kwargs):  # noqa: D401, ANN003
            return None

    cfg = MetricsConfig(strict_validation=True, use_cache=False)
    seen: list[str] = []
    monkeypatch.setattr(
        "invarlock.eval.metrics.logger.debug", lambda msg: seen.append(str(msg))
    )
    InputValidator.validate_model(EmptyModel(), cfg)
    assert any("Could not count model parameters" in msg for msg in seen)


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
    _perform_pre_eval_checks(DummyModel(), dataloader, torch.device("cpu"), cfg)
