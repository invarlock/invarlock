from __future__ import annotations

from types import SimpleNamespace

from invarlock.adapters.hf_causal import HF_Causal_Adapter
from invarlock.adapters.hf_loading import HFLoaderStrategy
from invarlock.core.exceptions import ModelLoadError

_MISTRAL3_ARCH = "Mistral3For" + "ConditionalGeneration"


def test_hf_causal_direct_fallback_ignores_remote_code_flag(monkeypatch) -> None:
    auto_loader = object()
    direct_loader = object()
    calls: list[tuple[bool, dict[str, object]]] = []

    def fake_resolve_core_loader_strategy(
        *,
        task: str,
        model_id: str,
        kwargs: dict[str, object] | None = None,
        allow_direct_submodule: bool = False,
    ) -> HFLoaderStrategy:
        normalized_kwargs = dict(kwargs or {})
        calls.append((allow_direct_submodule, normalized_kwargs))
        if allow_direct_submodule and not normalized_kwargs:
            return HFLoaderStrategy(
                task=task,
                strategy="direct_submodule",
                loader=direct_loader,
                loader_label=(
                    "transformers.models.mistral3.modeling_mistral3." + _MISTRAL3_ARCH
                ),
                model_type="mistral3",
            )
        return HFLoaderStrategy(
            task=task,
            strategy="auto",
            loader=auto_loader,
            loader_label="transformers.AutoModelForCausalLM",
            model_type="mistral3",
        )

    monkeypatch.setattr(
        "invarlock.adapters.hf_causal.resolve_core_loader_strategy",
        fake_resolve_core_loader_strategy,
    )

    adapter = HF_Causal_Adapter()
    monkeypatch.setattr(adapter, "_safe_to_device", lambda model, device: model)

    def fake_load_pretrained_model(loader, model_id, **kwargs):
        if loader is auto_loader:
            raise ModelLoadError("auto load failed")
        assert loader is direct_loader
        return SimpleNamespace(config=SimpleNamespace(model_type="mistral3"))

    monkeypatch.setattr(adapter, "_load_pretrained_model", fake_load_pretrained_model)

    model = adapter.load_model("/tmp/ministral3-baseline", trust_remote_code=True)

    assert getattr(model.config, "model_type", None) == "mistral3"
    assert adapter._last_loader_strategy == "direct_submodule"
    assert any(
        allow_direct_submodule and resolved_kwargs == {}
        for allow_direct_submodule, resolved_kwargs in calls
    )
