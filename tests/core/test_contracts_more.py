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


def test_enforce_relative_spectral_cap_noop_when_sigma_below_limit() -> None:
    W = torch.zeros(4, 4)
    out = C.enforce_relative_spectral_cap(W.clone(), baseline_sigma=10.0, cap_ratio=1.0)
    assert torch.allclose(out, W)


def test_spectral_norm_reshape_and_cpu_fallback(monkeypatch) -> None:
    W = torch.randn(2, 3, 4)
    state = {"first": True}
    orig_svd = torch.linalg.svdvals

    def _fake_svd(x):
        if state["first"]:
            state["first"] = False
            raise RuntimeError("svd fail")
        return orig_svd(x)

    monkeypatch.setattr(torch.linalg, "svdvals", _fake_svd)
    val = C._spectral_norm(W)
    assert isinstance(val, float)
    assert val > 0.0
