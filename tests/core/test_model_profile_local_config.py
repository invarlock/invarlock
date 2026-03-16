from __future__ import annotations

import json
from pathlib import Path

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
