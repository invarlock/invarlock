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
        ("openai-community/gpt2", "hf_causal", "gpt2", "causal"),
        ("bert-base-uncased", "hf_mlm", "bert", "mlm"),
        ("mistralai/Mistral-7B-v0.1", "hf_causal", "mistral", "causal"),
        ("mistralai/Mixtral-8x7B-v0.1", "hf_causal", "mixtral", "causal"),
        ("Qwen/Qwen2-7B", "hf_causal", "qwen", "causal"),
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


def test_tokenizer_factory_produces_non_zero_tokens(monkeypatch):
    import invarlock.model_profile as mp

    class DummyTokenizer:
        def __init__(self) -> None:
            self.pad_token = None
            self.eos_token = "<eos>"
            self.name_or_path = "dummy"
            self.add_bos_token = False

        def get_vocab(self) -> dict[str, int]:
            return {"<eos>": 0, "hello": 1}

        def __call__(
            self, *_: str, truncation: bool, padding: str, max_length: int, **__: object
        ) -> dict[str, list[int]]:
            assert truncation is True
            assert padding == "max_length"
            return {
                "input_ids": [1] * max_length,
                "attention_mask": [1] * max_length,
            }

    class DummyTokenizerFactory:
        @classmethod
        def from_pretrained(cls, *_: object, **__: object) -> DummyTokenizer:
            return DummyTokenizer()

    monkeypatch.setattr(mp, "AutoTokenizer", DummyTokenizerFactory, raising=False)
    monkeypatch.setattr(mp, "GPT2Tokenizer", DummyTokenizerFactory, raising=False)

    profile = detect_model_profile(
        model_id="mistralai/Mistral-7B-v0.1", adapter="hf_causal"
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
