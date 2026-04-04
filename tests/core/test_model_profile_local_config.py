from __future__ import annotations

import json
from pathlib import Path

import pytest

import invarlock.model_profile as mp


class _DummyTokenizer:
    def __init__(
        self,
        *,
        name_or_path: str,
        mask_token: str | None = None,
        pad_token: str | None = None,
        eos_token: str | None = "<eos>",
    ) -> None:
        self.name_or_path = name_or_path
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.add_bos_token = False

    def get_vocab(self) -> dict[str, int]:
        return {"<eos>": 0, "hello": 1}


def test_detect_model_profile_uses_local_config_hints_for_auto_adapter(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "bert-local"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bert",
                "architectures": ["BertForMaskedLM"],
            }
        ),
        encoding="utf-8",
    )

    profile = mp.detect_model_profile(str(model_dir), adapter="auto")

    assert profile.family == "bert"
    assert profile.default_loss == "mlm"
    assert profile.default_provider == "hf_text"


def test_local_profile_config_helpers_tolerate_invalid_utf8(tmp_path: Path) -> None:
    model_dir = tmp_path / "broken-local"
    model_dir.mkdir()
    (model_dir / "config.json").write_bytes(b"\xff")
    (model_dir / "special_tokens_map.json").write_bytes(b"\xff")
    (model_dir / "tokenizer_config.json").write_bytes(b"\xff")

    assert mp._read_local_hf_config(str(model_dir)) is None
    assert mp._load_local_tokenizer_metadata(model_dir) == {
        "bos_token": None,
        "cls_token": None,
        "eos_token": None,
        "mask_token": None,
        "pad_token": None,
        "sep_token": None,
        "unk_token": None,
    }


@pytest.mark.parametrize(
    ("payload", "expected_family"),
    [
        (
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
            },
            "llama",
        ),
        (
            {
                "model_type": "gemma3",
                "architectures": ["Gemma3ForConditionalGeneration"],
            },
            "gemma",
        ),
        (
            {
                "model_type": "gemma4",
                "architectures": ["Gemma4ForConditionalGeneration"],
            },
            "gemma",
        ),
        (
            {
                "model_type": "olmo2",
                "architectures": ["Olmo2ForCausalLM"],
            },
            "olmo",
        ),
    ],
)
def test_detect_model_profile_uses_local_config_hints_for_new_causal_families(
    tmp_path: Path,
    payload: dict[str, object],
    expected_family: str,
) -> None:
    model_dir = tmp_path / expected_family
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    profile = mp.detect_model_profile(str(model_dir), adapter="auto")

    assert profile.family == expected_family
    assert profile.default_loss == "causal"
    assert profile.default_provider == "wikitext2"


def test_resolve_tokenizer_uses_model_specific_identifier_for_opt_like_models(
    monkeypatch,
) -> None:
    calls: list[tuple[str, bool | None]] = []

    class _Factory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> _DummyTokenizer:
            calls.append((model_id, kwargs.get("local_files_only")))
            return _DummyTokenizer(name_or_path=model_id)

    monkeypatch.setattr(mp, "AutoTokenizer", _Factory, raising=False)

    profile = mp.detect_model_profile("facebook/opt-125m", adapter="hf_causal")
    tokenizer, _ = mp.resolve_tokenizer(profile)

    assert tokenizer.name_or_path == "facebook/opt-125m"
    assert calls == [("facebook/opt-125m", True)]


def test_resolve_tokenizer_uses_same_origin_fallback_for_local_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    model_dir = tmp_path / "edited-checkpoint"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "mistral",
                "architectures": ["MistralForCausalLM"],
                "_name_or_path": "mistralai/Mistral-7B-v0.1",
            }
        ),
        encoding="utf-8",
    )
    calls: list[tuple[str, bool | None]] = []

    class _Factory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> _DummyTokenizer:
            calls.append((model_id, kwargs.get("local_files_only")))
            if model_id == str(model_dir):
                raise OSError("tokenizer files missing")
            return _DummyTokenizer(name_or_path=model_id)

    monkeypatch.setattr(mp, "AutoTokenizer", _Factory, raising=False)

    profile = mp.detect_model_profile(str(model_dir), adapter="auto")
    tokenizer, _ = mp.resolve_tokenizer(profile)

    assert profile.family == "mistral"
    assert tokenizer.name_or_path == "mistralai/Mistral-7B-v0.1"
    assert calls == [
        (str(model_dir), True),
        ("mistralai/Mistral-7B-v0.1", True),
    ]


def test_resolve_tokenizer_blocks_remote_fallback_when_network_disabled(
    monkeypatch,
) -> None:
    calls: list[tuple[str, bool | None]] = []

    class _Factory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> _DummyTokenizer:
            calls.append((model_id, kwargs.get("local_files_only")))
            raise OSError("missing cached tokenizer files")

    monkeypatch.setattr(mp, "AutoTokenizer", _Factory, raising=False)
    monkeypatch.setattr(mp, "network_allowed", lambda: False)

    profile = mp.detect_model_profile("facebook/opt-125m", adapter="hf_causal")

    with pytest.raises(RuntimeError, match="Network tokenizer downloads are disabled"):
        mp.resolve_tokenizer(profile)

    assert calls == [("facebook/opt-125m", True)]


def test_resolve_tokenizer_uses_remote_fallback_only_after_cache_miss(
    monkeypatch,
) -> None:
    calls: list[tuple[str, bool | None]] = []

    class _Factory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> _DummyTokenizer:
            calls.append((model_id, kwargs.get("local_files_only")))
            if kwargs.get("local_files_only"):
                raise OSError("missing cached tokenizer files")
            return _DummyTokenizer(name_or_path=model_id)

    monkeypatch.setattr(mp, "AutoTokenizer", _Factory, raising=False)
    monkeypatch.setattr(mp, "network_allowed", lambda: True)

    profile = mp.detect_model_profile("facebook/opt-125m", adapter="hf_causal")
    tokenizer, _ = mp.resolve_tokenizer(profile)

    assert tokenizer.name_or_path == "facebook/opt-125m"
    assert calls == [
        ("facebook/opt-125m", True),
        ("facebook/opt-125m", None),
    ]


def test_resolve_tokenizer_retries_remote_on_hf_local_entry_cache_miss(
    monkeypatch,
) -> None:
    calls: list[tuple[str, bool | None]] = []

    class _Factory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> _DummyTokenizer:
            calls.append((model_id, kwargs.get("local_files_only")))
            if kwargs.get("local_files_only"):
                raise OSError(
                    "We couldn't connect to 'https://huggingface.co' to load the files, "
                    "and couldn't find them in the cached files. Outgoing traffic has "
                    "been disabled."
                )
            return _DummyTokenizer(name_or_path=model_id)

    monkeypatch.setattr(mp, "AutoTokenizer", _Factory, raising=False)
    monkeypatch.setattr(mp, "network_allowed", lambda: True)

    profile = mp.detect_model_profile("sshleifer/tiny-gpt2", adapter="hf_causal")
    tokenizer, _ = mp.resolve_tokenizer(profile)

    assert tokenizer.name_or_path == "sshleifer/tiny-gpt2"
    assert calls == [
        ("sshleifer/tiny-gpt2", True),
        ("sshleifer/tiny-gpt2", None),
    ]


def test_resolve_tokenizer_falls_back_to_slow_tokenizer_when_fast_backend_missing(
    monkeypatch,
) -> None:
    calls: list[tuple[str, bool | None, bool | None]] = []

    class _Factory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> _DummyTokenizer:
            calls.append(
                (
                    model_id,
                    kwargs.get("local_files_only"),
                    kwargs.get("use_fast"),
                )
            )
            if kwargs.get("use_fast") is False:
                return _DummyTokenizer(name_or_path=model_id, mask_token="[MASK]")
            raise ValueError(
                "Couldn't instantiate the backend tokenizer from one of: "
                "(1) a `tokenizers` library serialization file, "
                "(2) a slow tokenizer instance to convert or "
                "(3) an equivalent slow tokenizer class to instantiate and convert. "
                "You need to have sentencepiece or tiktoken installed to convert "
                "a slow tokenizer to a fast one."
            )

    monkeypatch.setattr(mp, "AutoTokenizer", _Factory, raising=False)

    profile = mp.detect_model_profile("prajjwal1/bert-tiny", adapter="hf_mlm")
    tokenizer, _ = mp.resolve_tokenizer(profile)

    assert tokenizer.name_or_path == "prajjwal1/bert-tiny"
    assert calls == [
        ("prajjwal1/bert-tiny", True, None),
        ("prajjwal1/bert-tiny", True, False),
    ]


def test_resolve_tokenizer_does_not_retry_remote_on_non_cache_loader_error(
    monkeypatch,
) -> None:
    calls: list[tuple[str, bool | None]] = []

    class _Factory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> _DummyTokenizer:
            calls.append((model_id, kwargs.get("local_files_only")))
            raise RuntimeError("deserializer exploded")

    monkeypatch.setattr(mp, "AutoTokenizer", _Factory, raising=False)
    monkeypatch.setattr(mp, "network_allowed", lambda: True)

    profile = mp.detect_model_profile("facebook/opt-125m", adapter="hf_causal")

    with pytest.raises(RuntimeError, match="deserializer exploded"):
        mp.resolve_tokenizer(profile)

    assert calls == [("facebook/opt-125m", True)]


def test_hash_tokenizer_reraises_unexpected_vocab_failures() -> None:
    class _BrokenTokenizer:
        name_or_path = "broken"

        def get_vocab(self):  # noqa: ANN001
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        mp._hash_tokenizer(_BrokenTokenizer())


def test_resolve_tokenizer_reraises_add_bos_token_mutation_failures(
    monkeypatch,
) -> None:
    class _BrokenBosTokenizer:
        def __init__(self, *, name_or_path: str) -> None:
            self.name_or_path = name_or_path
            self.pad_token = "<eos>"
            self.eos_token = "<eos>"

        def get_vocab(self) -> dict[str, int]:
            return {"<eos>": 0, "hello": 1}

        @property
        def add_bos_token(self) -> bool:
            return False

        @add_bos_token.setter
        def add_bos_token(self, _value: bool) -> None:
            raise RuntimeError("bos-mutation-failed")

    class _Factory:
        @classmethod
        def from_pretrained(
            cls, model_id: str, **_kwargs: object
        ) -> _BrokenBosTokenizer:
            return _BrokenBosTokenizer(name_or_path=model_id)

    monkeypatch.setattr(mp, "AutoTokenizer", _Factory, raising=False)

    profile = mp.detect_model_profile("mistralai/Mistral-7B-v0.1", adapter="hf_causal")

    with pytest.raises(RuntimeError, match="bos-mutation-failed"):
        mp.resolve_tokenizer(profile)


def test_resolve_tokenizer_uses_local_tokenizer_json_fast_path(
    monkeypatch, tmp_path: Path
) -> None:
    tokenizers = pytest.importorskip("tokenizers")

    model_dir = tmp_path / "local-fast-gpt2"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gpt2",
                "architectures": ["GPT2LMHeadModel"],
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "[BOS]",
                "eos_token": "[EOS]",
                "pad_token": "[PAD]",
                "unk_token": "[UNK]",
                "tokenizer_class": "PreTrainedTokenizerFast",
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "bos_token": "[BOS]",
                "eos_token": "[EOS]",
                "pad_token": "[PAD]",
                "unk_token": "[UNK]",
            }
        ),
        encoding="utf-8",
    )

    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {
                "[PAD]": 0,
                "[UNK]": 1,
                "[BOS]": 2,
                "[EOS]": 3,
                "hello": 4,
                "world": 5,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(model_dir / "tokenizer.json"))

    def _unexpected_transformers() -> None:
        raise AssertionError("transformers tokenizer path should not be used")

    monkeypatch.setattr(
        mp,
        "_ensure_transformers_tokenizer_support",
        _unexpected_transformers,
    )

    profile = mp.detect_model_profile(str(model_dir), adapter="hf_causal")
    resolved, hash_value = mp.resolve_tokenizer(profile)
    encoded = resolved(
        "hello world",
        truncation=True,
        padding="max_length",
        max_length=6,
    )

    assert resolved.name_or_path == str(model_dir)
    assert resolved.pad_token == "[PAD]"
    assert resolved.pad_token_id == 0
    assert resolved.eos_token == "[EOS]"
    assert isinstance(hash_value, str) and hash_value
    assert len(encoded["input_ids"]) == 6
    assert encoded["input_ids"][:2] == [4, 5]
    assert encoded["attention_mask"] == [1, 1, 0, 0, 0, 0]
