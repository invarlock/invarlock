from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path
from types import SimpleNamespace

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


def test_hf_causal_invalid_model_id_maps_to_model_load_error(
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

    from invarlock.adapters.hf_causal import HF_Causal_Adapter

    adapter = HF_Causal_Adapter()
    with pytest.raises(Exception) as ei:
        adapter.load_model("bad-id")
    err = ei.value
    from invarlock.core.exceptions import ModelLoadError

    assert isinstance(err, ModelLoadError)
    assert getattr(err, "code", "") == "E201"
    assert "MODEL-LOAD-FAILED" in str(err)


def test_gptq_missing_runtime_maps_to_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from invarlock.plugins.hf_gptq_adapter import HF_GPTQ_Adapter

    real_import = builtins.__import__

    def _imp(name, *a, **k):  # type: ignore[no-untyped-def]
        if name == "auto_gptq":
            raise ImportError("auto_gptq unavailable")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _imp)
    adapter = HF_GPTQ_Adapter()
    with pytest.raises(Exception) as ei:
        adapter.load_model("demo/model")
    err = ei.value
    from invarlock.core.exceptions import DependencyError

    assert isinstance(err, DependencyError)
    assert getattr(err, "code", "") == "E203"
    assert "DEPENDENCY-MISSING" in str(err)


def test_auto_quantization_probe_unexpected_loader_error_surfaces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from invarlock.adapters import auto as auto_mod

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    def _boom(_payload: str) -> object:
        raise RuntimeError("probe boom")

    monkeypatch.setattr(auto_mod.json, "loads", _boom)

    with pytest.raises(RuntimeError, match="probe boom"):
        auto_mod._detect_quantization_from_path(str(tmp_path))


def test_auto_quantization_probe_tolerates_invalid_utf8_config(tmp_path: Path) -> None:
    from invarlock.adapters import auto as auto_mod

    (tmp_path / "config.json").write_bytes(b"\xff")

    assert auto_mod._detect_quantization_from_path(str(tmp_path)) is None


def test_auto_quantization_probe_tolerates_non_string_model_quant_method() -> None:
    from invarlock.adapters import auto as auto_mod

    class _Model:
        config = type("Cfg", (), {"quantization_config": {"quant_method": 7}})()

    assert auto_mod._detect_quantization_from_model(_Model()) is None


def test_hf_causal_can_handle_surfaces_unexpected_spec_probe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from invarlock.adapters import hf_causal as hf_causal_mod
    from invarlock.adapters.hf_causal import HF_Causal_Adapter

    class ExplodingSpec:
        spec_name = "exploding"

        def matches(self, model, base, layers):  # noqa: ANN001
            raise RuntimeError("spec boom")

    model = SimpleNamespace(
        transformer=SimpleNamespace(h=[object()]),
        config=SimpleNamespace(),
    )

    monkeypatch.setattr(hf_causal_mod, "_SPECS", [ExplodingSpec()])

    with pytest.raises(RuntimeError, match="spec boom"):
        HF_Causal_Adapter().can_handle(model)


def test_hf_causal_describe_raises_when_no_spec_matches() -> None:
    from invarlock.adapters.hf_causal import HF_Causal_Adapter
    from invarlock.core.exceptions import AdapterError

    model = SimpleNamespace(
        transformer=SimpleNamespace(h=[SimpleNamespace()]),
        config=SimpleNamespace(
            model_type="custom",
            num_attention_heads=4,
            hidden_size=16,
            vocab_size=32,
        ),
    )

    with pytest.raises(
        AdapterError, match="no matching HF causal adapter spec"
    ):
        HF_Causal_Adapter().describe(model)


def test_hf_mlm_loader_does_not_fallback_on_unexpected_loader_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from invarlock.adapters import hf_mlm as hf_mlm_mod
    from invarlock.adapters.hf_loading import HFLoaderStrategy
    from invarlock.adapters.hf_mlm import HF_MLM_Adapter
    from invarlock.core.exceptions import ModelLoadError

    calls: list[str] = []
    strategies = iter(
        (
            HFLoaderStrategy("mlm", "direct", "primary", "primary"),
            HFLoaderStrategy("mlm", "auto", "auto", "auto"),
            HFLoaderStrategy("mlm_base", "base", "fallback", "fallback"),
        )
    )

    monkeypatch.setattr(
        hf_mlm_mod,
        "resolve_core_loader_strategy",
        lambda *args, **kwargs: next(strategies),
    )

    class DummyAdapter(HF_MLM_Adapter):
        def _load_pretrained_model(self, loader, model_id, **kwargs):  # noqa: ANN001
            calls.append(str(loader))
            raise RuntimeError("loader boom")

        def _safe_to_device(self, model, device):  # noqa: ANN001
            return model

    with pytest.raises(ModelLoadError, match="MODEL-LOAD-FAILED: primary"):
        DummyAdapter().load_model("demo/model", device="cpu")

    assert calls == ["primary"]


def test_hf_mlm_loader_falls_back_only_for_masked_lm_loader_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from invarlock.adapters import hf_mlm as hf_mlm_mod
    from invarlock.adapters.hf_loading import HFLoaderStrategy
    from invarlock.adapters.hf_mlm import HF_MLM_Adapter

    calls: list[str] = []
    strategies = iter(
        (
            HFLoaderStrategy("mlm", "direct", "primary", "primary"),
            HFLoaderStrategy("mlm", "auto", "auto", "auto"),
            HFLoaderStrategy("mlm_base", "base", "fallback", "fallback"),
        )
    )

    monkeypatch.setattr(
        hf_mlm_mod,
        "resolve_core_loader_strategy",
        lambda *args, **kwargs: next(strategies),
    )

    class DummyAdapter(HF_MLM_Adapter):
        def _load_pretrained_model(self, loader, model_id, **kwargs):  # noqa: ANN001
            calls.append(str(loader))
            if loader != "fallback":
                raise OSError(
                    "Unrecognized configuration class for this kind of AutoModelForMaskedLM"
                )
            return {"loader": loader}

        def _safe_to_device(self, model, device):  # noqa: ANN001
            return model

    loaded = DummyAdapter().load_model("demo/model", device="cpu")

    assert loaded == {"loader": "fallback"}
    assert calls == ["primary", "auto", "fallback"]
