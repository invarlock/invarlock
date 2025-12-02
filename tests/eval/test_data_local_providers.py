from __future__ import annotations

import json
from pathlib import Path

import pytest

import invarlock.eval.data as data_mod
from invarlock.eval.data import (
    EvaluationWindow,
    compute_window_hash,
    get_provider,
    list_providers,
)


class _EncodeTokenizer:
    pad_token_id = 0

    def encode(self, text, truncation=True, max_length=8):
        ids = list(range(1, min(len(text), max_length) + 1))
        return ids


class _CallTokenizer:
    pad_token_id = 3

    def __call__(self, text, truncation=True, max_length=8):
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


def test_local_jsonl_provider_no_samples(tmp_path: Path):
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")
    provider = data_mod.LocalJSONLProvider(file=str(empty_file))
    from invarlock.core.exceptions import DataError

    with pytest.raises(DataError):
        provider.windows(_EncodeTokenizer(), preview_n=1, final_n=1)


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


def test_hf_text_provider_windows_and_tokenize(monkeypatch):
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True, raising=False)

    def fake_load_dataset(path, name=None, split=None, cache_dir=None):
        return [{"text": "example one"}, {"text": "example two"}]

    monkeypatch.setattr(data_mod, "load_dataset", fake_load_dataset, raising=False)
    provider = data_mod.HFTextProvider(dataset_name="dummy", max_samples=2)
    tok = _EncodeTokenizer()
    prev, fin = provider.windows(tok, preview_n=1, final_n=1, seq_len=4)
    assert len(prev.input_ids) == 1 and len(fin.input_ids) == 1

    # Exercise callable tokenizer branch
    window = provider._simple_tokenize(["short text"], _CallTokenizer(), 4, [0])
    assert isinstance(window, EvaluationWindow)


def test_hf_seq2seq_provider_windows_and_capacity(monkeypatch):
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True, raising=False)

    def fake_load_dataset(path, name=None, split=None, cache_dir=None):
        return [
            {"source": "src one", "target": "tgt one"},
            {"source": "src two", "target": "tgt two"},
        ]

    monkeypatch.setattr(data_mod, "load_dataset", fake_load_dataset, raising=False)
    provider = data_mod.HFSeq2SeqProvider("dummy")
    prev, fin = provider.windows(_EncodeTokenizer(), preview_n=1, final_n=1, seq_len=6)
    assert len(prev.input_ids) == 1 and len(fin.input_ids) == 1
    assert provider.last_preview_labels and provider.last_final_labels
    cap = provider.estimate_capacity(_EncodeTokenizer(), seq_len=4, stride=1)
    assert cap["available_unique"] == 2


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
    from invarlock.core.exceptions import ValidationError

    with pytest.raises(ValidationError):
        get_provider("unknown-provider")
