import pytest

from invarlock.model_profile import (
    ModelProfile,
    detect_model_profile,
    resolve_tokenizer,
)

pytest.importorskip("transformers")


@pytest.mark.parametrize(
    ("model_id", "adapter", "expected_family", "expected_loss"),
    [
        ("openai-community/gpt2", "hf_gpt2_medium", "gpt2", "causal"),
        ("bert-base-uncased", "hf_bert_base", "bert", "mlm"),
        (
            "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "hf_llama_small",
            "llama",
            "causal",
        ),
    ],
)
def test_detect_profile_core(model_id, adapter, expected_family, expected_loss):
    profile = detect_model_profile(model_id=model_id, adapter=adapter)

    assert isinstance(profile, ModelProfile)
    assert profile.family == expected_family
    assert profile.default_loss == expected_loss
    assert callable(profile.make_tokenizer)
    assert isinstance(profile.module_selectors, dict)
    assert isinstance(profile.invariants, tuple)
    assert isinstance(profile.cert_lints, tuple)


def test_tokenizer_factory_matches_family():
    profile = detect_model_profile(
        model_id="hf-internal-testing/tiny-random-LlamaForCausalLM",
        adapter="hf_llama_small",
    )
    tokenizer, hash_value = resolve_tokenizer(profile)

    encoded = tokenizer(
        "The quick brown fox jumps over the lazy dog.",
        truncation=True,
        padding="max_length",
        max_length=32,
    )

    assert isinstance(hash_value, str) and len(hash_value) > 0
    assert any(token_id != 0 for token_id in encoded["input_ids"])
    assert all(mask in (0, 1) for mask in encoded["attention_mask"])


def test_unknown_profile_falls_back_to_conservative_defaults():
    profile = detect_model_profile(
        model_id="my-org/custom-net", adapter="custom_adapter"
    )

    assert profile.family == "unknown"
    assert profile.default_loss == "causal"
    assert "attention" in profile.module_selectors["attention"]
