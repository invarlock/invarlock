from types import SimpleNamespace

import pytest
import torch

from invarlock.eval import metrics as M
from invarlock.eval.metrics_activation import _perform_pre_eval_checks


def test_measure_memory_forward_exception_path():
    window = SimpleNamespace(input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]])

    class Crashy(torch.nn.Module):
        def forward(self, *args, **kwargs):  # noqa: D401
            raise RuntimeError("boom")

        def parameters(self):  # pragma: no cover
            yield from ()

    with pytest.raises(RuntimeError, match="Memory measurement failed for sample 0"):
        M.measure_memory(Crashy(), window, device="cpu")


def test_measure_memory_raises_when_window_has_no_non_empty_samples():
    window = SimpleNamespace(input_ids=[[]], attention_masks=[[]])

    class Tiny(torch.nn.Module):
        def forward(self, *args, **kwargs):  # noqa: D401
            return None

    with pytest.raises(M.ValidationError) as exc_info:
        M.measure_memory(Tiny(), window, device="cpu")
    assert (
        exc_info.value.details["reason"]
        == "memory measurement requires at least one non-empty sample"
    )


def test_pre_eval_checks_dry_run_failure(caplog: pytest.LogCaptureFixture):
    class CrashOnForward(torch.nn.Module):
        def forward(self, *args, **kwargs):  # noqa: D401
            raise RuntimeError("dry run fails")

        def parameters(self):  # pragma: no cover
            yield from ()

        class config:  # noqa: D401
            n_positions = 4

    class Loader:
        def __iter__(self):
            yield {
                "input_ids": torch.ones(1, 6, dtype=torch.long),
                "attention_mask": torch.ones(1, 6, dtype=torch.long),
            }

    caplog.set_level("WARNING", logger="invarlock.eval.metrics_activation")

    result = _perform_pre_eval_checks(
        CrashOnForward(), Loader(), device=torch.device("cpu"), config=M.MetricsConfig()
    )
    assert result is None
    assert "Input sequence length 6 exceeds model limit 4" in caplog.text
    assert "Pre-evaluation dry run failed: dry run fails" in caplog.text
