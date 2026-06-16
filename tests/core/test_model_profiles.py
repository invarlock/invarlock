from types import SimpleNamespace

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
        ("openlm-research/open_llama_7b", "hf_causal", "llama", "causal"),
        ("mistralai/Mistral-7B-v0.1", "hf_causal", "mistral", "causal"),
        (
            "mistralai/Ministral-3-8B-Instruct-2512-BF16",
            "hf_causal",
            "mistral",
            "causal",
        ),
        ("mistralai/Mixtral-8x7B-v0.1", "hf_causal", "mixtral", "causal"),
        ("openai/gpt-oss-20b", "hf_causal", "gpt_oss", "causal"),
        ("Qwen/Qwen2-7B", "hf_causal", "qwen", "causal"),
        ("Qwen/Qwen2.5-7B", "hf_causal", "qwen", "causal"),
        ("Qwen/Qwen2.5-14B", "hf_causal", "qwen", "causal"),
        ("Qwen/Qwen3-8B", "hf_causal", "qwen", "causal"),
        ("Qwen/Qwen3.5-9B", "hf_causal", "qwen", "causal"),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "hf_causal", "llama", "causal"),
        ("unsloth/gemma-2-9b-it-bnb-4bit", "hf_causal", "gemma", "causal"),
        (
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "hf_causal",
            "qwen",
            "causal",
        ),
        ("microsoft/Phi-4-reasoning-plus", "hf_causal", "phi4", "causal"),
        ("allenai/OLMoE-1B-7B-0924", "hf_causal", "olmoe", "causal"),
        ("allenai/OLMo-2-1124-7B", "hf_causal", "olmo", "causal"),
        ("allenai/OLMo-2-1124-13B-Instruct", "hf_causal", "olmo", "causal"),
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


def test_tokenizer_factory_forwards_remote_code_opt_in(monkeypatch):
    import invarlock.model_profile as mp
    from invarlock.runtime_security import runtime_allowances_scope

    class DummyTokenizer:
        pad_token = "<pad>"
        eos_token = "<eos>"
        name_or_path = "dummy"

        def get_vocab(self) -> dict[str, int]:
            return {"<pad>": 0, "<eos>": 1}

    calls: list[dict[str, object]] = []

    class DummyTokenizerFactory:
        @classmethod
        def from_pretrained(cls, *_args: object, **kwargs: object) -> DummyTokenizer:
            calls.append(dict(kwargs))
            return DummyTokenizer()

    monkeypatch.setattr(mp, "AutoTokenizer", DummyTokenizerFactory, raising=False)

    profile = detect_model_profile(
        model_id="local-chatglm-compatible-checkpoint",
        adapter="hf_causal",
        tokenizer_load_kwargs={"trust_remote_code": True},
    )

    with runtime_allowances_scope(allow_remote_code=True):
        tokenizer, hash_value = resolve_tokenizer(profile)

    assert tokenizer.name_or_path == "dummy"
    assert isinstance(hash_value, str) and hash_value
    assert calls
    assert calls[0]["trust_remote_code"] is True


def test_run_environment_attaches_config_remote_code_to_profile() -> None:
    from invarlock.core.run_orchestrator_execute_environment import (
        _detect_model_profile_with_tokenizer_kwargs,
        _extract_tokenizer_load_kwargs_from_cfg,
    )

    cfg = SimpleNamespace(
        model=SimpleNamespace(trust_remote_code=False),
        model_dump=lambda: {"model": {"trust_remote_code": True, "revision": "abc123"}},
    )
    assert _extract_tokenizer_load_kwargs_from_cfg(cfg) == {
        "trust_remote_code": True,
        "revision": "abc123",
    }

    seen: dict[str, object] = {}

    def detector(**kwargs: object) -> SimpleNamespace:
        seen.update(kwargs)
        return SimpleNamespace()

    profile = _detect_model_profile_with_tokenizer_kwargs(
        detector,
        model_id="local-chatglm-compatible-checkpoint",
        adapter="hf_causal",
        tokenizer_load_kwargs={"trust_remote_code": True, "revision": "abc123"},
    )

    assert seen["tokenizer_load_kwargs"] == {
        "trust_remote_code": True,
        "revision": "abc123",
    }
    assert not hasattr(profile, "tokenizer_load_kwargs")

    def legacy_detector(model_id: str, adapter: str) -> SimpleNamespace:
        assert model_id == "local-chatglm-compatible-checkpoint"
        assert adapter == "hf_causal"
        return SimpleNamespace()

    legacy_profile = _detect_model_profile_with_tokenizer_kwargs(
        legacy_detector,
        model_id="local-chatglm-compatible-checkpoint",
        adapter="hf_causal",
        tokenizer_load_kwargs={"trust_remote_code": True, "revision": "abc123"},
    )

    assert legacy_profile.tokenizer_load_kwargs == {
        "trust_remote_code": True,
        "revision": "abc123",
    }


def test_unknown_profile_falls_back_to_conservative_defaults():
    profile = detect_model_profile(
        model_id="my-org/custom-net", adapter="custom_adapter"
    )

    assert profile.family == "unknown"
    assert profile.default_loss == "causal"
    assert "attention" in profile.module_selectors["attention"]


def test_qwen35_profile_exposes_linear_attention_selectors():
    profile = detect_model_profile(
        model_id="Qwen/Qwen3.5-9B",
        adapter="hf_causal",
    )

    assert profile.family == "qwen"
    assert "linear_attn.in_proj_qkv" in profile.module_selectors["attention"]
    assert "linear_attn.out_proj" in profile.module_selectors["attention"]


def test_seq2seq_profile_exposes_t5_attention_and_ffn_selectors():
    profile = detect_model_profile(
        model_id="google/flan-t5-base",
        adapter="hf_seq2seq",
    )

    assert profile.family == "seq2seq"
    assert profile.default_loss == "seq2seq"
    assert "SelfAttention.q" in profile.module_selectors["attention"]
    assert "DenseReluDense.wo" in profile.module_selectors["ffn"]


def test_gpt_oss_profile_exposes_moe_attention_and_ffn_selectors():
    profile = detect_model_profile(
        model_id="openai/gpt-oss-20b",
        adapter="hf_causal",
    )

    assert profile.family == "gpt_oss"
    assert "self_attn.q_proj" in profile.module_selectors["attention"]
    assert "mlp.router" in profile.module_selectors["ffn"]
    assert "mlp.experts" in profile.module_selectors["ffn"]
