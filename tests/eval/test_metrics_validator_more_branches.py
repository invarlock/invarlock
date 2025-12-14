from __future__ import annotations

import torch

from invarlock.core.exceptions import ValidationError
from invarlock.eval.metrics import InputValidator, MetricsConfig


def test_metrics_config_rejects_negative_oracle_windows() -> None:
    try:
        MetricsConfig(oracle_windows=-1, use_cache=False)
    except ValidationError as exc:
        assert exc.code == "E402"
    else:
        raise AssertionError("Expected ValidationError")


def test_metrics_config_rejects_non_positive_max_tokens() -> None:
    try:
        MetricsConfig(max_tokens=0, use_cache=False)
    except ValidationError as exc:
        assert exc.code == "E402"
    else:
        raise AssertionError("Expected ValidationError")


def test_validate_tensor_nan_strict_raises() -> None:
    cfg = MetricsConfig(strict_validation=True, use_cache=False)
    x = torch.tensor([float("nan")])
    try:
        InputValidator.validate_tensor(x, "x", cfg)
    except ValidationError as exc:
        assert "NaN" in str(exc.details.get("reason", ""))
    else:
        raise AssertionError("Expected ValidationError")


def test_validate_tensor_nan_non_strict_replaces() -> None:
    cfg = MetricsConfig(strict_validation=False, use_cache=False, nan_replacement=0.25)
    x = torch.tensor([float("nan")])
    out = InputValidator.validate_tensor(x, "x", cfg)
    assert torch.isnan(out).any().item() is False
    assert float(out.item()) == 0.25


def test_validate_tensor_inf_strict_raises() -> None:
    cfg = MetricsConfig(strict_validation=True, use_cache=False)
    x = torch.tensor([float("inf")])
    try:
        InputValidator.validate_tensor(x, "x", cfg)
    except ValidationError as exc:
        assert "Inf" in str(exc.details.get("reason", ""))
    else:
        raise AssertionError("Expected ValidationError")


def test_validate_tensor_inf_non_strict_replaces() -> None:
    cfg = MetricsConfig(strict_validation=False, use_cache=False, inf_replacement=123.0)
    x = torch.tensor([float("inf"), float("-inf")])
    out = InputValidator.validate_tensor(x, "x", cfg)
    assert float(out[0].item()) == 123.0
    assert float(out[1].item()) == -123.0


def test_validate_dataloader_empty_branches_raise_or_warn(monkeypatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(
        "invarlock.eval.metrics.logger.warning", lambda msg: seen.append(str(msg))
    )

    cfg_strict = MetricsConfig(allow_empty_data=False, use_cache=False)
    try:
        InputValidator.validate_dataloader([], cfg_strict)
    except ValidationError as exc:
        assert exc.code == "E402"
    else:
        raise AssertionError("Expected ValidationError")

    cfg_allow = MetricsConfig(allow_empty_data=True, use_cache=False)
    InputValidator.validate_dataloader([], cfg_allow)
    assert any("Dataloader is empty" in msg for msg in seen)
