from __future__ import annotations

import json
from pathlib import Path

import pytest

import invarlock.eval.data as data_mod
import invarlock.eval.data_support as data_support_mod
from invarlock.eval.data import (
    EvaluationWindow,
    compute_window_hash,
    get_provider,
    list_providers,
)
from invarlock.eval.data_providers import HFSeq2SeqProvider


class _EncodeTokenizer:
    pad_token_id = 0

    def encode(self, text, truncation=True, max_length=8, padding="max_length"):
        base = (sum(ord(ch) for ch in text) % 97) + 1
        ids = [base + idx for idx in range(min(len(text), max_length))]
        return ids


class _CallTokenizer:
    pad_token_id = 3

    def __call__(
        self,
        text,
        truncation=True,
        max_length=8,
        padding="max_length",
        return_attention_mask=True,
    ):
        return {"input_ids": [len(text)]}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_local_jsonl_provider_load_and_windows(tmp_path: Path):
    file_main = tmp_path / "samples.jsonl"
    _write_jsonl(
        file_main,
        [
            {"text": "alpha entry"},
            {"text": "beta entry"},
            {"text": ""},
            {"text": "gamma"},
        ],
    )
    # Secondary directory with matching glob
    dir_extra = tmp_path / "nested"
    dir_extra.mkdir()
    extra_file = dir_extra / "more.jsonl"
    _write_jsonl(extra_file, [{"text": "delta entry"}])

    provider = data_mod.LocalJSONLProvider(
        file=str(file_main),
        path=str(dir_extra),
        data_files=[str(extra_file)],
        max_samples=3,
    )
    texts = provider.load()
    assert texts[:2] == ["alpha entry", "beta entry"]
    tokenizer = _EncodeTokenizer()
    preview, final = provider.windows(
        tokenizer, seq_len=4, preview_n=2, final_n=1, split="validation"
    )
    assert len(preview.input_ids) == 2
    assert len(final.input_ids) == 1


def test_local_jsonl_provider_skips_bad_data_files_entries_and_loads_valid_ones(
    tmp_path: Path,
):
    data_file = tmp_path / "rows.jsonl"
    _write_jsonl(data_file, [{"text": "alpha"}, {"text": "beta"}])

    class _BadValueStr:
        def __str__(self) -> str:
            raise ValueError("bad data_files entry")

    provider = data_mod.LocalJSONLProvider(
        data_files=[_BadValueStr(), str(data_file)],
        max_samples=2,
    )

    assert provider.load() == ["alpha", "beta"]


def test_local_jsonl_provider_propagates_unexpected_data_files_errors() -> None:
    class _BoomStr:
        def __str__(self) -> str:
            raise RuntimeError("boom")

    provider = data_mod.LocalJSONLProvider(data_files=[_BoomStr()])

    with pytest.raises(RuntimeError, match="boom"):
        provider.load()


def test_local_jsonl_provider_no_samples(tmp_path: Path):
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")
    provider = data_mod.LocalJSONLProvider(file=str(empty_file))
    from invarlock.core.exceptions import DataError

    with pytest.raises(DataError):
        provider.windows(_EncodeTokenizer(), preview_n=1, final_n=1)


def test_local_jsonl_provider_fails_closed_on_partial_tokenization_error(
    tmp_path: Path,
):
    file_main = tmp_path / "partial.jsonl"
    _write_jsonl(
        file_main,
        [
            {"text": "alpha entry"},
            {"text": "beta entry"},
        ],
    )

    class _FragileTokenizer(_EncodeTokenizer):
        def encode(self, text, truncation=True, max_length=8, padding="max_length"):
            if text.startswith("beta"):
                raise RuntimeError("boom")
            return super().encode(
                text,
                truncation=truncation,
                max_length=max_length,
                padding=padding,
            )

    provider = data_mod.LocalJSONLProvider(file=str(file_main), max_samples=2)
    from invarlock.core.exceptions import DataError

    with pytest.raises(DataError, match="TOKENIZE-INSUFFICIENT"):
        provider.windows(_FragileTokenizer(), preview_n=1, final_n=1, seq_len=4)


def test_local_jsonl_provider_skips_unreadable_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    readable = tmp_path / "readable.jsonl"
    unreadable = tmp_path / "unreadable.jsonl"
    _write_jsonl(readable, [{"text": "keep-me"}])
    unreadable.write_text('{"text": "drop-me"}\n', encoding="utf-8")

    original_open = Path.open

    def _patched_open(self: Path, *args, **kwargs):
        if self == unreadable:
            raise OSError("permission denied")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _patched_open, raising=True)

    provider = data_mod.LocalJSONLProvider(
        data_files=[str(readable), str(unreadable)],
        max_samples=2,
    )

    assert provider.load() == ["keep-me"]


def test_local_jsonl_pairs_provider_windows_and_labels(tmp_path: Path):
    file_main = tmp_path / "pairs.jsonl"
    _write_jsonl(
        file_main,
        [
            {"source": "hello", "target": "world"},
            {"source": "foo", "target": "bar"},
        ],
    )
    provider = data_mod.LocalJSONLPairsProvider(file=str(file_main), max_samples=2)
    preview, final = provider.windows(_EncodeTokenizer(), preview_n=1, final_n=1)
    assert preview.indices == [0]
    assert provider.last_preview_labels and provider.last_final_labels


def test_local_jsonl_pairs_provider_registry_uses_local_jsonl_owner(tmp_path: Path):
    pairs_file = tmp_path / "pairs.jsonl"
    _write_jsonl(
        pairs_file,
        [
            {"source": "left", "target": "right"},
        ],
    )

    provider = get_provider("local_jsonl_pairs", file=str(pairs_file), max_samples=1)

    assert isinstance(provider, data_mod.LocalJSONLPairsProvider)
    preview, final = provider.windows(_EncodeTokenizer(), preview_n=1, final_n=0)
    assert len(preview.indices) == 1
    assert len(final.indices) == 0


def test_hf_text_provider_windows_and_tokenize(monkeypatch):
    monkeypatch.setattr(data_support_mod, "HAS_DATASETS", True, raising=False)

    def fake_load_dataset(path, name=None, split=None, cache_dir=None, **kwargs):
        return [{"text": "example one"}, {"text": "example two"}]

    monkeypatch.setattr(
        data_support_mod, "load_dataset", fake_load_dataset, raising=False
    )
    provider = data_mod.HFTextProvider(dataset_name="dummy", max_samples=2)
    tok = _EncodeTokenizer()
    prev, fin = provider.windows(tok, preview_n=1, final_n=1, seq_len=4)
    assert len(prev.input_ids) == 1 and len(fin.input_ids) == 1

    # Exercise callable tokenizer branch
    window = provider._simple_tokenize(["short text"], _CallTokenizer(), 4, [0])
    assert isinstance(window, EvaluationWindow)


def test_hf_seq2seq_provider_windows_and_capacity(monkeypatch):
    monkeypatch.setattr(data_support_mod, "HAS_DATASETS", True, raising=False)

    def fake_load_dataset(path, name=None, split=None, cache_dir=None, **kwargs):
        return [
            {"source": "src one", "target": "tgt one"},
            {"source": "src two", "target": "tgt two"},
        ]

    monkeypatch.setattr(
        data_support_mod, "load_dataset", fake_load_dataset, raising=False
    )
    provider = HFSeq2SeqProvider("dummy")
    prev, fin = provider.windows(_EncodeTokenizer(), preview_n=1, final_n=1, seq_len=6)
    assert len(prev.input_ids) == 1 and len(fin.input_ids) == 1
    assert provider.last_preview_labels and provider.last_final_labels
    cap = provider.estimate_capacity(_EncodeTokenizer(), seq_len=4, stride=1)
    assert cap["available_unique"] == 2


def test_hf_seq2seq_provider_uses_seeded_pair_shuffle(monkeypatch):
    monkeypatch.setattr(data_support_mod, "HAS_DATASETS", True, raising=False)

    def fake_load_dataset(path, name=None, split=None, cache_dir=None, **kwargs):
        return [
            {"source": "src zero", "target": "tgt zero"},
            {"source": "src one", "target": "tgt one"},
            {"source": "src two", "target": "tgt two"},
            {"source": "src three", "target": "tgt three"},
        ]

    monkeypatch.setattr(
        data_support_mod, "load_dataset", fake_load_dataset, raising=False
    )
    provider = HFSeq2SeqProvider("dummy")

    first_preview, first_final = provider.windows(
        _EncodeTokenizer(),
        preview_n=2,
        final_n=2,
        seq_len=6,
        seed=7,
    )
    second_preview, second_final = provider.windows(
        _EncodeTokenizer(),
        preview_n=2,
        final_n=2,
        seq_len=6,
        seed=7,
    )

    assert first_preview.indices == second_preview.indices
    assert first_final.indices == second_final.indices
    assert first_preview.indices != [0, 1]


def test_hf_seq2seq_provider_supports_revision_prefix_and_nested_fields(monkeypatch):
    monkeypatch.setattr(data_support_mod, "HAS_DATASETS", True, raising=False)
    captured = {}

    def fake_load_dataset(path, name=None, split=None, cache_dir=None, **kwargs):
        captured.update(
            {
                "path": path,
                "name": name,
                "split": split,
                "cache_dir": cache_dir,
                "kwargs": kwargs,
            }
        )
        return [
            {"translation": {"en": "How old are you?", "de": "Wie alt bist du?"}},
            {"translation": {"en": "That is good.", "de": "Das ist gut."}},
        ]

    monkeypatch.setattr(
        data_support_mod, "load_dataset", fake_load_dataset, raising=False
    )
    provider = HFSeq2SeqProvider(
        "wmt14",
        config_name="de-en",
        revision="abc123",
        src_field="translation.en",
        tgt_field="translation.de",
        src_prefix="translate English to German: ",
    )

    prev, fin = provider.windows(_EncodeTokenizer(), preview_n=1, final_n=1, seq_len=8)

    assert len(prev.input_ids) == 1
    assert len(fin.input_ids) == 1
    assert provider._pairs_cache["validation"][0] == (
        "translate English to German: How old are you?",
        "Wie alt bist du?",
    )
    assert captured == {
        "path": "wmt14",
        "name": "de-en",
        "split": "validation",
        "cache_dir": None,
        "kwargs": {"revision": "abc123"},
    }


def test_compute_window_hash_include_data():
    window = EvaluationWindow(
        input_ids=[[1, 2], [3, 4]],
        attention_masks=[[1, 1], [1, 0]],
        indices=[0, 1],
    )
    digest = compute_window_hash(window, include_data=True)
    assert len(digest) == 64


def test_get_provider_registry_helpers():
    providers = list_providers()
    assert "local_jsonl" in providers
    assert "local_jsonl_pairs" in providers
    from invarlock.core.exceptions import ValidationError

    with pytest.raises(ValidationError):
        get_provider("unknown-provider")
