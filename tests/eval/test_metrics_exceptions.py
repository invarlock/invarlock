from __future__ import annotations

import math

import pytest

from invarlock.core.exceptions import MetricsError, ValidationError
from invarlock.eval.metrics import bootstrap_confidence_interval
from invarlock.eval.primary_metric import (
    compute_primary_metric_from_report,
    validate_primary_metric_block,
)


def test_bootstrap_invalid_alpha_raises_validation_error() -> None:
    # alpha outside (0,1) should be treated as validation error E402
    with pytest.raises(ValidationError) as ei:
        bootstrap_confidence_interval([1.0, 2.0, 3.0], alpha=1.5)
    assert getattr(ei.value, "code", None) == "E402"


def test_bootstrap_statistic_error_raises_metrics_error() -> None:
    # A statistic that raises should be wrapped as MetricsError E401
    def bad_stat(_xs):  # pragma: no cover - simple trigger
        raise ZeroDivisionError("boom")

    with pytest.raises(MetricsError) as ei:
        bootstrap_confidence_interval([1.0, 2.0, 3.0], statistic=bad_stat)
    assert getattr(ei.value, "code", None) == "E401"


def test_validate_primary_metric_non_finite_raises_validation_error() -> None:
    # Construct report windows that cause overflow to inf in display space
    # ppl = exp(mean(logloss)); using a huge logloss forces exp overflow
    huge = 1e9
    report = {
        "evaluation_windows": {
            "preview": {"logloss": [huge], "token_counts": [1]},
            "final": {"logloss": [huge], "token_counts": [1]},
        }
    }
    block = compute_primary_metric_from_report(report, kind="ppl_causal")
    # preview/final expected to be inf; validator should raise ValidationError E402
    assert math.isinf(block.get("preview", float("nan")))
    assert math.isinf(block.get("final", float("nan")))
    with pytest.raises(ValidationError) as ei:
        validate_primary_metric_block(block)
    assert getattr(ei.value, "code", None) == "E402"
