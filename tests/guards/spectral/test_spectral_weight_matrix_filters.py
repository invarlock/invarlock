from __future__ import annotations

import torch.nn as nn

from invarlock.guards.spectral import SpectralGuard


class _NoMatrixWeight(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = None


class _MixedWeightModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.skip = _NoMatrixWeight()
        self.linear = nn.Linear(4, 4, bias=False)


class _TinyGemmaLikeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([nn.Module()])
        layer = self.model.language_model.layers[0]
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(4, 4, bias=False)
        layer.mlp = nn.Module()
        layer.mlp.down_proj = nn.Linear(4, 4, bias=False)
        layer.per_layer_projection = nn.Linear(4, 4, bias=False)
        self.model.audio_tower = nn.Module()
        self.model.audio_tower.layers = nn.ModuleList([nn.Module()])
        self.model.audio_tower.layers[0].self_attn = nn.Module()
        self.model.audio_tower.layers[0].self_attn.relative_k_proj = nn.Linear(
            4, 4, bias=False
        )


def test_spectral_guard_prepare_skips_modules_without_matrix_weights() -> None:
    model = _MixedWeightModel()
    guard = SpectralGuard()

    result = guard.prepare(model, adapter=None, calib=None, policy={})

    assert result["ready"] is True
    assert "linear" in guard.baseline_sigmas
    assert "skip" not in guard.baseline_sigmas
    assert "linear" in guard.module_family_map
    assert "skip" not in guard.module_family_map


def test_spectral_guard_module_patterns_scope_gemma_text_blocks() -> None:
    model = _TinyGemmaLikeModel()
    guard = SpectralGuard(scope="all")

    result = guard.prepare(
        model,
        adapter=None,
        calib=None,
        policy={
            "module_include_patterns": [
                "model.language_model.layers.*.self_attn.*",
                "model.language_model.layers.*.mlp.*",
            ]
        },
    )

    assert result["ready"] is True
    assert set(guard.baseline_sigmas) == {
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.mlp.down_proj",
    }
    assert "model.audio_tower.layers.0.self_attn.relative_k_proj" not in (
        guard.baseline_sigmas
    )
    assert "model.language_model.layers.0.per_layer_projection" not in (
        guard.baseline_sigmas
    )


def test_spectral_guard_module_filter_excludes_matching_patterns() -> None:
    guard = SpectralGuard(scope="all")
    guard.module_include_patterns = ("model.*",)
    guard.module_exclude_patterns = ("*.audio_tower.*",)

    assert guard._module_filter_allows("model.language_model.layers.0.mlp") is True
    assert guard._module_filter_allows("model.audio_tower.layers.0.attn") is False
    assert guard._module_filter_allows("other.layers.0.mlp") is False
