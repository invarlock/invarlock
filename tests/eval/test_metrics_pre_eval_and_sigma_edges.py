import torch
import torch.nn as nn

from invarlock.eval.metrics import (
    MetricsConfig,
    _calculate_sigma_max,
    _perform_pre_eval_checks,
)


def test_pre_eval_checks_dry_run_failure_logs_warning():
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
    # Should not raise; just exercise warn path
    _perform_pre_eval_checks(
        BadForward().eval(), dl, torch.device("cpu"), MetricsConfig()
    )


def test_sigma_max_no_name_column_and_all_nonfinite_gains():
    # gains object without 'columns' attribute and with only non-finite values
    class Gains:
        def __len__(self):
            return 3

        @property
        def values(self):  # used when 'gain' missing
            return [float("nan"), float("inf"), float("nan")]

    class DM:
        def is_available(self, name):
            return name == "scan_model_gains"

        def get_module(self, name):
            def scan_model_gains(model, first_batch):
                return Gains()

            return scan_model_gains

    val = _calculate_sigma_max(
        nn.Linear(2, 2),
        {"input_ids": torch.ones(1, 4, dtype=torch.long)},
        DM(),
        MetricsConfig(progress_bars=False),
        torch.device("cpu"),
    )
    # Expect NaN result due to all non-finite
    assert isinstance(val, float) and (val != val)
