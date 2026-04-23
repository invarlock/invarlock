from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_script(script_name: str):
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "evidence_packs"
        / "python"
        / script_name
    )
    spec = importlib.util.spec_from_file_location(
        script_name.replace(".", "_"), script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("script_name", "matcher_name"),
    [
        ("create_edits_batch.py", "_matches_scope"),
        ("create_quant_rtn_model.py", "_should_quantize"),
        ("create_fp8_model.py", "_should_quantize"),
        ("create_pruned_model.py", "_should_prune"),
        ("create_lowrank_model.py", "_should_lowrank"),
    ],
)
def test_ffn_targeting_matches_tensorized_moe_expert_tensors(
    script_name: str, matcher_name: str
) -> None:
    module = _load_script(script_name)
    matcher = getattr(module, matcher_name)

    assert matcher("model.layers.0.mlp.gate.weight", "ffn") is True
    assert matcher("model.layers.0.mlp.experts.gate_up_proj", "ffn") is True
    assert matcher("model.layers.0.mlp.experts.down_proj", "ffn") is True


@pytest.mark.parametrize(
    ("script_name", "matcher_name"),
    [
        ("create_edits_batch.py", "_matches_scope"),
        ("create_quant_rtn_model.py", "_should_quantize"),
        ("create_fp8_model.py", "_should_quantize"),
        ("create_pruned_model.py", "_should_prune"),
        ("create_lowrank_model.py", "_should_lowrank"),
    ],
)
def test_attn_targeting_excludes_tensorized_moe_expert_tensors(
    script_name: str, matcher_name: str
) -> None:
    module = _load_script(script_name)
    matcher = getattr(module, matcher_name)

    assert matcher("model.layers.0.self_attn.q_proj.weight", "attn") is True
    assert matcher("model.layers.0.mlp.experts.gate_up_proj", "attn") is False
    assert matcher("model.layers.0.input_layernorm.weight", "all") is False


@pytest.mark.parametrize(
    ("script_name", "matcher_name"),
    [
        ("create_edits_batch.py", "_matches_scope"),
        ("create_quant_rtn_model.py", "_should_quantize"),
        ("create_fp8_model.py", "_should_quantize"),
        ("create_pruned_model.py", "_should_prune"),
        ("create_lowrank_model.py", "_should_lowrank"),
    ],
)
def test_targeting_excludes_multimodal_vision_paths_but_keeps_language_paths(
    script_name: str, matcher_name: str
) -> None:
    module = _load_script(script_name)
    matcher = getattr(module, matcher_name)

    assert (
        matcher(
            "model.vision_tower.transformer.layers.0.feed_forward.up_proj.weight", "ffn"
        )
        is False
    )
    assert matcher("model.multi_modal_projector.linear.weight", "all") is False
    assert matcher("model.language_model.layers.0.mlp.up_proj.weight", "ffn") is True
