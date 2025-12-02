from __future__ import annotations

import builtins
import sys
import types

import pytest


def test_bnb_missing_transformers_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    from invarlock.plugins.hf_bnb_adapter import HF_BNB_Adapter

    # Make importing transformers fail
    real_import = builtins.__import__

    def _imp(name, *a, **k):  # type: ignore[no-untyped-def]
        if name == "transformers":
            raise ImportError("transformers unavailable")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _imp)
    adapter = HF_BNB_Adapter()
    with pytest.raises(Exception) as ei:
        adapter.load_model("gpt2")
    err = ei.value
    from invarlock.core.exceptions import DependencyError

    assert isinstance(err, DependencyError)
    assert getattr(err, "code", "") == "E203"
    assert "DEPENDENCY-MISSING" in str(err)


def test_hf_gpt2_invalid_model_id_maps_to_model_load_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Provide a lightweight transformers stub so import works
    tr = types.ModuleType("transformers")

    class _Auto:
        @staticmethod
        def from_pretrained(*a, **k):  # type: ignore[no-untyped-def]
            raise OSError("bad model id")

    tr.AutoModelForCausalLM = _Auto  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", tr)

    from invarlock.adapters.hf_gpt2 import HF_GPT2_Adapter

    adapter = HF_GPT2_Adapter()
    with pytest.raises(Exception) as ei:
        adapter.load_model("bad-id")
    err = ei.value
    from invarlock.core.exceptions import ModelLoadError

    assert isinstance(err, ModelLoadError)
    assert getattr(err, "code", "") == "E201"
    assert "MODEL-LOAD-FAILED" in str(err)
