from __future__ import annotations

import pytest
import torch

from invarlock.eval import metrics_runtime


def test_causal_batch_stats_preserves_scalar_reductions_on_cpu() -> None:
    nll = torch.tensor([[0.25, 1.5, 7.0], [2.25, 0.75, 9.0]])
    valid = torch.tensor([[True, True, False], [True, True, False]])

    expected_sum = float(nll[valid].sum().item())
    expected_count = int(valid.sum().item())

    assert metrics_runtime._causal_batch_stats(nll, valid) == (
        expected_sum,
        expected_count,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_causal_batch_stats_preserves_scalar_reductions_on_cuda() -> None:
    nll = torch.tensor([[0.25, 1.5, 7.0], [2.25, 0.75, 9.0]], device="cuda")
    valid = torch.tensor([[True, True, False], [True, True, False]], device="cuda")

    expected_sum = float(nll[valid].sum().item())
    expected_count = int(valid.sum().item())

    assert metrics_runtime._causal_batch_stats(nll, valid) == (
        expected_sum,
        expected_count,
    )
