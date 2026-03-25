from __future__ import annotations

import torch

from invarlock.core.contracts import enforce_relative_spectral_cap


def test_enforce_relative_spectral_cap_negative_safe_limit_clamps_to_zero() -> None:
    weight = torch.eye(2, dtype=torch.float32)

    capped = enforce_relative_spectral_cap(
        weight.clone(),
        baseline_sigma=1.0,
        cap_ratio=-0.5,
    )

    assert torch.count_nonzero(capped) == 0
