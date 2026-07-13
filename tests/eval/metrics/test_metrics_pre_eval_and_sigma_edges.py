import pytest
import torch
import torch.nn as nn

from invarlock.eval.metrics import MetricsConfig
from invarlock.eval.metrics_activation import (
    _calculate_sigma_max,
    _perform_pre_eval_checks,
)


def test_pre_eval_checks_dry_run_failure_logs_warning(caplog: pytest.LogCaptureFixture):
    class BadForward(nn.Module):
        def __init__(self):
            super().__init__()

            class Cfg:
                # Keep short context to also exercise length check
                n_positions = 2

            self.config = Cfg()

        def forward(self, **kwargs):  # force failure
            raise RuntimeError("boom")

    dl = [{"input_ids": torch.ones(1, 4, dtype=torch.long)}]
    caplog.set_level("WARNING", logger="invarlock.eval.metrics_activation")

    result = _perform_pre_eval_checks(
        BadForward().eval(), dl, torch.device("cpu"), MetricsConfig()
    )
    assert result is None
    assert "Input sequence length 4 exceeds model limit 2" in caplog.text
    assert "Pre-evaluation dry run failed: boom" in caplog.text


def test_sigma_max_all_nonfinite_scan_returns_nan():
    class DM:
        def is_available(self, name):
            return name == "scan_model_gains"

        def get_module(self, name):
            def scan_model_gains(model):
                return {"spectral_norms": [float("nan"), float("inf")]}

            return scan_model_gains

    val = _calculate_sigma_max(
        nn.Linear(2, 2),
        DM(),
        MetricsConfig(),
    )
    # Expect NaN result due to all non-finite
    assert isinstance(val, float) and (val != val)
