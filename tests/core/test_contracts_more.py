import torch

from invarlock.core import contracts as C


def test_contracts_basic_branches():
    W = torch.randn(4, 4)
    # enforce_relative_spectral_cap returns same type and caps if needed
    out = C.enforce_relative_spectral_cap(W.clone(), baseline_sigma=1.0, cap_ratio=0.5)
    assert isinstance(out, torch.Tensor)

    # rmt_correction_is_monotone true/false branches
    assert C.rmt_correction_is_monotone(
        1.0, baseline_sigma=2.0, max_ratio=5.0, deadband=0.1
    )
    assert not C.rmt_correction_is_monotone(
        10.0, baseline_sigma=2.0, max_ratio=5.0, deadband=0.1
    )
