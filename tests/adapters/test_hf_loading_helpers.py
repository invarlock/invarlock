from __future__ import annotations

import pytest
import torch


@pytest.mark.unit
def test_resolve_trust_remote_code_defaults_false(monkeypatch):
    from invarlock.adapters.hf_loading import resolve_trust_remote_code

    monkeypatch.delenv("INVARLOCK_TRUST_REMOTE_CODE", raising=False)
    monkeypatch.delenv("ALLOW_REMOTE_CODE", raising=False)
    monkeypatch.delenv("TRUST_REMOTE_CODE_BOOL", raising=False)

    assert resolve_trust_remote_code({}) is False


@pytest.mark.unit
def test_resolve_trust_remote_code_env_opt_in(monkeypatch):
    from invarlock.adapters.hf_loading import resolve_trust_remote_code

    monkeypatch.setenv("INVARLOCK_TRUST_REMOTE_CODE", "1")
    assert resolve_trust_remote_code({}) is True


@pytest.mark.unit
def test_resolve_trust_remote_code_kwargs_override(monkeypatch):
    from invarlock.adapters.hf_loading import resolve_trust_remote_code

    monkeypatch.setenv("INVARLOCK_TRUST_REMOTE_CODE", "0")
    assert resolve_trust_remote_code({"trust_remote_code": True}) is True


@pytest.mark.unit
def test_default_torch_dtype_prefers_bf16_on_supported_cuda(monkeypatch):
    from invarlock.adapters.hf_loading import default_torch_dtype

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    assert default_torch_dtype() is torch.bfloat16


@pytest.mark.unit
def test_default_torch_dtype_falls_back_to_fp16_on_cuda(monkeypatch):
    from invarlock.adapters.hf_loading import default_torch_dtype

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    assert default_torch_dtype() is torch.float16


@pytest.mark.unit
def test_default_torch_dtype_uses_fp16_on_mps(monkeypatch):
    from invarlock.adapters.hf_loading import default_torch_dtype

    if not hasattr(torch.backends, "mps"):
        pytest.skip("MPS backend not available")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert default_torch_dtype() is torch.float16


@pytest.mark.unit
def test_default_torch_dtype_uses_fp32_on_cpu(monkeypatch):
    from invarlock.adapters.hf_loading import default_torch_dtype

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert default_torch_dtype() is torch.float32


@pytest.mark.unit
def test_resolve_torch_dtype_parses_strings(monkeypatch):
    from invarlock.adapters.hf_loading import resolve_torch_dtype

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert resolve_torch_dtype({"torch_dtype": "float16"}) is torch.float16
    assert resolve_torch_dtype({"torch_dtype": "bfloat16"}) is torch.bfloat16
    assert resolve_torch_dtype({"torch_dtype": "auto"}) == "auto"
