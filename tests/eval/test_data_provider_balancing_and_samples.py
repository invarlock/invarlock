from pathlib import Path
from types import SimpleNamespace

import pytest

import invarlock.eval.data_support as data_support_mod
from invarlock.eval.data import (
    WikiText2Provider,
)
from invarlock.eval.providers.seq2seq import Seq2SeqProvider


def _data_module_path() -> Path:
    return Path(__file__).resolve().parents[2] / "src/invarlock/eval/data.py"


def _data_support_module_path() -> Path:
    return Path(__file__).resolve().parents[2] / "src/invarlock/eval/data_support.py"


class DummyTok:
    def encode(self, text, max_length, truncation, padding):
        # Simple tokenizer: map chars to ids
        ids = list(range(1, min(len(text) + 1, max_length + 1)))
        pad_len = max_length - len(ids)
        if pad_len > 0:
            ids += [0] * pad_len
        return ids


def test_wikitext2_load_does_not_override_explicit_cache_dir_on_lock_error(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(data_support_mod, "HAS_DATASETS", True)

    def fake_load_dataset(*args, **kwargs):  # noqa: ARG001
        raise PermissionError(
            "Operation not permitted: '/Users/test/.cache/huggingface/datasets/sample.lock'"
        )

    monkeypatch.setattr(data_support_mod, "load_dataset", fake_load_dataset)
    provider = WikiText2Provider(cache_dir=tmp_path / "explicit")

    with pytest.raises(PermissionError, match="Operation not permitted"):
        provider.load(split="validation", max_samples=5)


def test_seq2seq_provider_windows_and_masks():
    class _Tok:
        pad_token_id = 0

    provider = Seq2SeqProvider(n=2, src_len=3, tgt_len=2, pad_id=0)
    prev, fin = provider.windows(_Tok(), seq_len=4, preview_n=1, final_n=1)
    assert len(prev.input_ids[0]) == 4
    assert provider.last_final_labels and provider.last_final_labels[0][-1] == -100


def test_seq2seq_attention_mask_tracks_padding():
    class Tok:
        pad_token_id = 0

    provider = Seq2SeqProvider(n=1, src_len=3, tgt_len=2, pad_id=0)
    prev, _ = provider.windows(Tok(), seq_len=4, preview_n=1, final_n=0)
    assert prev.attention_masks[0] == [1, 1, 1, 0]


def test_seq2seq_provider_capacity(monkeypatch):
    class DummySeq2Seq:
        def __init__(self, **kwargs):
            self._n = kwargs.get("n", 1)

        def batches(self, seed, batch_size):  # noqa: ARG002
            yield {
                "src_ids": [[1, 2, 0, 0]] * 3,
                "src_mask": [[1, 1, 0, 0]] * 3,
                "tgt_ids": [[3, 4]] * 3,
            }

    monkeypatch.setattr(
        "invarlock.eval.providers.seq2seq.Seq2SeqProvider", DummySeq2Seq, raising=False
    )
    provider = Seq2SeqProvider(n=1)
    cap = provider.estimate_capacity(tokenizer=None, seq_len=4, stride=2)
    assert cap["examples_available"] >= 1
    assert cap["tokens_available"] >= 4


def test_wt2_frequency_fallback_lone_candidate(monkeypatch):
    # Force datasets present
    monkeypatch.setattr(data_support_mod, "HAS_DATASETS", True)
    pt = WikiText2Provider()
    # Provide enough long texts
    monkeypatch.setattr(pt, "load", lambda **kw: ["z" * 30] * 21)

    class Tok:
        def __call__(self, text, **kw):  # noqa: ARG002
            # Produce 4 tokens with 2 real tokens
            return {"input_ids": [1, 2, 0, 0], "attention_mask": [1, 1, 0, 0]}

    # Odd total to exercise lone-candidate branch
    prev, fin = pt.windows(Tok(), seq_len=4, preview_n=3, final_n=2, seed=7)
    assert len(prev) == 3 and len(fin) == 2


def test_wikitext2_balancing_swap_reverted(monkeypatch):
    """Craft deterministic difficulties so balancing swap worsens gap and reverts."""
    monkeypatch.setattr(data_support_mod, "HAS_DATASETS", True)
    provider = WikiText2Provider()
    monkeypatch.setattr(provider, "load", lambda **kw: ["z" * 40] * 6)

    def collector(texts, indices, tokenizer, seq_len):  # noqa: ARG001
        return [(idx, [idx + 1, 0, 0, 0], [1, 0, 0, 0], 1) for idx in indices]

    def difficulty_scorer(candidates):
        for rank, candidate in enumerate(
            sorted(candidates, key=lambda c: c["dataset_index"])
        ):
            candidate["difficulty"] = float(rank + 1)
        return True

    monkeypatch.setattr(provider, "_collect_tokenized_samples", collector)
    monkeypatch.setattr(provider, "_score_candidates_byte_ngram", difficulty_scorer)
    preview, final = provider.windows(
        SimpleNamespace(), seq_len=4, preview_n=3, final_n=3, seed=42
    )
    # Balancing should keep the original ordering (preview picks indices 0,3,4)
    assert sorted(preview.indices) == [0, 3, 4]
    assert sorted(final.indices) == [1, 2, 5]


def test_wikitext2_duplicate_indices_skipped(monkeypatch):
    monkeypatch.setattr(data_support_mod, "HAS_DATASETS", True)
    pt = WikiText2Provider()
    monkeypatch.setattr(pt, "load", lambda **kw: ["text " + str(i) for i in range(40)])

    def collector(texts, indices, tokenizer, seq_len):  # noqa: ARG001
        out = []
        for idx in indices:
            seq = [idx + 1, 0, 0, 0]
            mask = [1, 0, 0, 0]
            out.append((idx, seq, mask, 1))
            if idx % 2 == 0:
                # Duplicate entry should be skipped via used_indices branch
                out.append((idx, list(seq), list(mask), 1))
        return out

    monkeypatch.setattr(pt, "_collect_tokenized_samples", collector)
    prev, final = pt.windows(
        SimpleNamespace(), seq_len=4, preview_n=4, final_n=3, seed=11
    )
    assert len(prev) == 4 and len(final) == 3


def test_collect_tokenized_samples_warns_on_failure(monkeypatch):
    monkeypatch.setattr(data_support_mod, "HAS_DATASETS", True)
    pt = WikiText2Provider()

    class BadTokenizer:
        def __call__(self, text, **kwargs):  # noqa: ARG001
            raise ValueError("boom")

    with pytest.warns(UserWarning):
        res = pt._collect_tokenized_samples(["alpha"], [0], BadTokenizer(), 4)
    assert res == []
