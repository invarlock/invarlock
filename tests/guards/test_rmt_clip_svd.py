import torch

import invarlock.guards.rmt as R


def test_clip_full_svd_components_and_nonfinite():
    W = torch.randn(6, 4)
    U, S_clipped, Vt = R.clip_full_svd(W, clip_val=0.1, return_components=True)
    assert U is not None and S_clipped is not None and Vt is not None
    assert (S_clipped <= 0.1 + 1e-6).all()

    # Non-finite → returns (None, None, None) when return_components
    W_nan = W.clone()
    W_nan[0, 0] = float("nan")
    U2, S2, Vt2 = R.clip_full_svd(W_nan, clip_val=1.0, return_components=True)
    assert U2 is None and S2 is None and Vt2 is None


def test_clip_full_svd_exception_fallback(monkeypatch):
    W = torch.randn(3, 3)
    # Force svd to raise to exercise fallback
    monkeypatch.setattr(
        R.torch.linalg,
        "svd",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fail")),
    )
    out = R.clip_full_svd(W, clip_val=1.0, return_components=False)
    assert torch.allclose(out, W) or out.shape == W.shape
