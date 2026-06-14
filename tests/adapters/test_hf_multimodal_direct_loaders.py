from __future__ import annotations

import pytest


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_id", "model_type", "loader_label"),
    [
        (
            "google/gemma-3n-E4B-it",
            "gemma3n",
            "transformers.models.gemma3n.modeling_gemma3n.Gemma3nForConditionalGeneration",
        ),
        (
            "google/gemma-3-4b-it",
            "gemma3",
            "transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration",
        ),
    ],
)
def test_resolve_core_loader_strategy_supports_multimodal_gemma3_variants(
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
    model_type: str,
    loader_label: str,
) -> None:
    import invarlock.adapters.hf_loading as hf_loading

    monkeypatch.setattr(
        hf_loading,
        "_import_symbol",
        lambda module_path, symbol_name: f"{module_path}.{symbol_name}",
    )

    strategy = hf_loading.resolve_core_loader_strategy(
        task="multimodal",
        model_id=model_id,
        allow_direct_submodule=True,
    )

    assert strategy.strategy == "direct_submodule"
    assert strategy.model_type == model_type
    assert strategy.loader_label == loader_label
