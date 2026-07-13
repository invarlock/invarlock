from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock.edits.quant_rtn_plan import QuantTargetSelector
from invarlock.model_profile import detect_model_profile

transformers = pytest.importorskip("transformers")


@pytest.mark.parametrize(
    ("model_id", "family", "attention", "ffn"),
    [
        (
            "tiiuae/falcon-7b",
            "falcon",
            {
                "self_attention.query_key_value",
                "self_attention.dense",
            },
            {"mlp.dense_h_to_4h", "mlp.dense_4h_to_h"},
        ),
        (
            "ibm-granite/granite-4.1-3b",
            "granite",
            {
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
            },
            {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"},
        ),
        (
            "ibm-granite/granite-4.1-8b",
            "granite",
            {
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
            },
            {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"},
        ),
        (
            "HuggingFaceTB/SmolLM3-3B",
            "smollm3",
            {
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
            },
            {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"},
        ),
    ],
)
def test_supported_causal_profile_has_exact_family_contract(
    model_id: str,
    family: str,
    attention: set[str],
    ffn: set[str],
) -> None:
    profile = detect_model_profile(model_id, adapter="hf_causal")

    assert profile.family == family
    assert profile.default_loss == "causal"
    assert profile.default_metric == "ppl_causal"
    assert profile.invariants == ("rope_rotary_embedding",)
    assert set(profile.module_selectors["attention"]) == attention
    assert set(profile.module_selectors["ffn"]) == ffn


@pytest.mark.parametrize(
    ("model_type", "architecture", "family"),
    [
        ("falcon", "FalconForCausalLM", "falcon"),
        ("granite", "GraniteForCausalLM", "granite"),
        ("smollm3", "SmolLM3ForCausalLM", "smollm3"),
    ],
)
def test_local_config_detects_supported_family_without_name_hint(
    tmp_path: Path,
    model_type: str,
    architecture: str,
    family: str,
) -> None:
    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": model_type, "architectures": [architecture]}),
        encoding="utf-8",
    )

    profile = detect_model_profile(str(model_dir), adapter="auto")

    assert profile.family == family
    assert profile.family != "gpt2"


def _tiny_falcon():
    config = transformers.FalconConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_kv_heads=1,
        ffn_hidden_size=32,
    )
    return transformers.FalconForCausalLM(config)


def _tiny_granite():
    config = transformers.GraniteConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return transformers.GraniteForCausalLM(config)


def _tiny_smollm3():
    config = transformers.SmolLM3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=4,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return transformers.SmolLM3ForCausalLM(config)


@pytest.mark.parametrize(
    ("model_id", "model_factory", "expected_attention", "expected_ffn"),
    [
        (
            "tiiuae/falcon-7b",
            _tiny_falcon,
            {
                "transformer.h.0.self_attention.query_key_value",
                "transformer.h.0.self_attention.dense",
            },
            {
                "transformer.h.0.mlp.dense_h_to_4h",
                "transformer.h.0.mlp.dense_4h_to_h",
            },
        ),
        (
            "ibm-granite/granite-4.1-3b",
            _tiny_granite,
            {
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.k_proj",
                "model.layers.0.self_attn.v_proj",
                "model.layers.0.self_attn.o_proj",
            },
            {
                "model.layers.0.mlp.gate_proj",
                "model.layers.0.mlp.up_proj",
                "model.layers.0.mlp.down_proj",
            },
        ),
        (
            "HuggingFaceTB/SmolLM3-3B",
            _tiny_smollm3,
            {
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.k_proj",
                "model.layers.0.self_attn.v_proj",
                "model.layers.0.self_attn.o_proj",
            },
            {
                "model.layers.0.mlp.gate_proj",
                "model.layers.0.mlp.up_proj",
                "model.layers.0.mlp.down_proj",
            },
        ),
    ],
)
def test_supported_family_selectors_match_transformers_modules_only(
    model_id: str,
    model_factory,
    expected_attention: set[str],
    expected_ffn: set[str],
) -> None:
    model = model_factory()
    selectors = detect_model_profile(model_id, adapter="hf_causal").module_selectors

    attention = QuantTargetSelector(scope="attn", module_selectors=selectors).select(
        model
    )
    ffn = QuantTargetSelector(scope="ffn", module_selectors=selectors).select(model)

    assert {target.name for target in attention} == expected_attention
    assert {target.name for target in ffn} == expected_ffn
    assert all(
        target.selection_reason == "model_profile_selector" for target in attention
    )
    assert all(target.selection_reason == "model_profile_selector" for target in ffn)
    assert "lm_head" not in expected_attention | expected_ffn
    assert "model.embed_tokens" not in expected_attention | expected_ffn
