from types import SimpleNamespace

import torch

from invarlock.eval import metrics as M


def test_measure_memory_forward_exception_path():
    window = SimpleNamespace(input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]])

    class Crashy(torch.nn.Module):
        def forward(self, *args, **kwargs):  # noqa: D401
            raise RuntimeError("boom")

        def parameters(self):  # pragma: no cover
            yield from ()

    mem = M.measure_memory(Crashy(), window, device="cpu")
    assert isinstance(mem, float) and mem >= 0.0


def test_pre_eval_checks_dry_run_failure():
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

    # Should not raise
    M._perform_pre_eval_checks(
        CrashOnForward(), Loader(), device=torch.device("cpu"), config=M.MetricsConfig()
    )
