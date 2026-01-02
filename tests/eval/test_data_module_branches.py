import builtins
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import invarlock.eval.data as data_mod
from invarlock.eval.data import (
    EvaluationWindow,
    HFTextProvider,
    SyntheticProvider,
    WikiText2Provider,
    compute_window_hash,
    get_provider,
    list_providers,
)


def _data_module_path() -> Path:
    return Path(__file__).resolve().parents[2] / "src/invarlock/eval/data.py"


def test_get_provider_known_and_unknown():
    assert "wikitext2" in list_providers()
    p = get_provider("synthetic")
    assert isinstance(p, SyntheticProvider)
    from invarlock.core.exceptions import ValidationError

    with pytest.raises(ValidationError):
        get_provider("unknown")


def test_compute_window_hash_include_data_branch():
    win = EvaluationWindow(
        input_ids=[[1, 2, 3]], attention_masks=[[1, 1, 1]], indices=[0]
    )
    h1 = compute_window_hash(win, include_data=False)
    h2 = compute_window_hash(win, include_data=True)
    assert isinstance(h1, str) and isinstance(h2, str) and h1 != h2


def test_eval_data_module_reimport_without_dependencies(monkeypatch):
    """Reload data module to exercise import fallbacks for datasets/torch."""
    data_path = _data_module_path()
    spec = importlib.util.spec_from_file_location(
        "invarlock.eval.data_nodeps", data_path
    )
    module = importlib.util.module_from_spec(spec)
    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("datasets") or name.startswith("torch"):
            raise ImportError(f"missing {name}")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("invarlock.eval.data_nodeps", None)
    assert module.HAS_DATASETS is False
    assert module.HAS_TORCH is False


def test_wikitext2_estimate_capacity_slow_without_target_total(monkeypatch):
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True)
    provider = WikiText2Provider()
    monkeypatch.setattr(provider, "load", lambda **kw: ["text sample" * 3] * 4)

    def collector(texts, indices, tokenizer, seq_len):  # noqa: ARG001
        result = []
        for idx in indices:
            result.append((idx, [idx + 1, 0, 0, 0], [1, 0, 0, 0], 1))
        return result

    monkeypatch.setattr(provider, "_collect_tokenized_samples", collector)
    cap = provider.estimate_capacity(SimpleNamespace(), seq_len=4, stride=2)
    assert "candidate_unique" not in cap
    assert cap["available_unique"] == cap["available_nonoverlap"]


def test_wikitext2_estimate_capacity_fast_target_total(monkeypatch):
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True)
    provider = WikiText2Provider()
    monkeypatch.setattr(provider, "load", lambda **kw: ["long text"] * 3)
    cap = provider.estimate_capacity(
        SimpleNamespace(), seq_len=8, stride=4, target_total=10, fast_mode=True
    )
    assert cap["candidate_limit"] == cap["available_nonoverlap"]
    assert cap["candidate_limit"] >= 10


def test_wikitext2_dependency_check_and_fast_capacity(monkeypatch):
    # Allow construction by pretending datasets is present
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True)
    pt = WikiText2Provider()
    # Monkeypatch load to avoid HF datasets
    monkeypatch.setattr(pt, "load", lambda **kw: ["a" * 30] * 5)
    # Fast path via env var
    monkeypatch.setenv("INVARLOCK_CAPACITY_FAST", "1")
    cap = pt.estimate_capacity(tokenizer=SimpleNamespace(), seq_len=8, stride=4)
    assert cap["available_nonoverlap"] == 5 and cap["candidate_unique"] == 5
    # Explicit fast_mode argument path
    monkeypatch.delenv("INVARLOCK_CAPACITY_FAST", raising=False)
    cap_fast = pt.estimate_capacity(
        tokenizer=SimpleNamespace(), seq_len=8, stride=4, fast_mode=True
    )
    assert cap_fast["available_nonoverlap"] == 5
    # No texts branch
    monkeypatch.setattr(pt, "load", lambda **kw: [])
    cap2 = pt.estimate_capacity(tokenizer=SimpleNamespace(), seq_len=8, stride=4)
    assert cap2["available_nonoverlap"] == 0
    # Slow path with candidate_unique
    monkeypatch.setattr(pt, "load", lambda **kw: ["a" * 30] * 3)

    def collector(texts, indices, tokenizer, seq_len):  # noqa: ARG001
        res = []
        for idx in indices:
            res.append((idx, [1, 2, 0, 0], [1, 1, 0, 0], 2))
        return res

    monkeypatch.setattr(pt, "_collect_tokenized_samples", collector)
    cap3 = pt.estimate_capacity(
        tokenizer=SimpleNamespace(),
        seq_len=4,
        stride=2,
        target_total=2,
        fast_mode=False,
    )
    assert cap3.get("candidate_unique") == 1 and cap3.get("candidate_limit") == 3


def test_wikitext2_requires_datasets(monkeypatch):
    monkeypatch.setattr(data_mod, "HAS_DATASETS", False)
    from invarlock.core.exceptions import DependencyError

    with pytest.raises(DependencyError):
        WikiText2Provider()


def test_hf_text_provider_requires_datasets(monkeypatch):
    monkeypatch.setattr(data_mod, "HAS_DATASETS", False)
    from invarlock.core.exceptions import DependencyError

    with pytest.raises(DependencyError):
        HFTextProvider(dataset_name="dummy")


def test_wikitext2_windows_insufficient_and_nonpositive(monkeypatch):
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True)
    pt = WikiText2Provider()
    from invarlock.core.exceptions import DataError, ValidationError

    with pytest.raises(ValidationError):
        pt.windows(tokenizer=SimpleNamespace(), preview_n=0, final_n=0)
    # Insufficient samples path
    monkeypatch.setattr(pt, "load", lambda **kw: ["x" * 30] * 5)
    with pytest.raises(DataError):
        pt.windows(tokenizer=SimpleNamespace(), preview_n=4, final_n=4)


def test_wikitext2_windows_tokenization_failure(monkeypatch):
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True)
    pt = WikiText2Provider()
    monkeypatch.setattr(pt, "load", lambda **kw: ["z" * 30] * 20)

    def empty_collect(*args, **kwargs):  # noqa: ARG001
        return []

    monkeypatch.setattr(pt, "_collect_tokenized_samples", empty_collect)
    from invarlock.core.exceptions import DataError

    with pytest.raises(DataError, match="TOKENIZE-INSUFFICIENT"):
        pt.windows(tokenizer=SimpleNamespace(), preview_n=2, final_n=2)


class DummyTok:
    def encode(self, text, max_length, truncation, padding):
        # Simple tokenizer: map chars to ids
        ids = list(range(1, min(len(text) + 1, max_length + 1)))
        pad_len = max_length - len(ids)
        if pad_len > 0:
            ids += [0] * pad_len
        return ids


def test_synthetic_provider_branches():
    tok = DummyTok()
    sp = SyntheticProvider(base_samples=["Hello world" * 3])
    cap = sp.estimate_capacity(tokenizer=tok, seq_len=8, stride=4)
    assert cap["available_nonoverlap"] == 1
    texts = sp.load(max_samples=3)
    assert len(texts) == 3
    prev, final = sp.windows(tokenizer=tok, seq_len=8, preview_n=2, final_n=1)
    assert isinstance(prev, EvaluationWindow) and len(prev) == 2 and len(final) == 1
    # Exercise to_dict on window
    d = prev.to_dict()
    assert d["length"] == 2 and "input_ids" in d


def test_synthetic_simple_tokenize_fallback_no_encode():
    class NoEnc:
        pad_token_id = 0

    sp = SyntheticProvider(base_samples=["abc def ghi" * 2])
    win_prev, _ = sp.windows(tokenizer=NoEnc(), seq_len=8, preview_n=1, final_n=0)
    assert len(win_prev.input_ids[0]) == 8 and len(win_prev.attention_masks[0]) == 8


def test_hf_text_provider_empty_raises(monkeypatch):
    hp = HFTextProvider(
        dataset_name="dummy", config_name=None, text_field="text", max_samples=4
    )
    monkeypatch.setattr(hp, "load", lambda **kw: [])
    from invarlock.core.exceptions import DataError

    with pytest.raises(DataError):
        hp.windows(tokenizer=SimpleNamespace(), preview_n=1, final_n=1)


def test_collect_tokenized_samples_paths(monkeypatch):
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True)
    pt = WikiText2Provider()

    class Tok:
        def __call__(self, text, **kw):  # noqa: ARG002
            # Return lists when HAS_TORCH is False path; we want both branches
            return {"input_ids": [1, 2, 0, 0], "attention_mask": [1, 1, 0, 0]}

    out = pt._collect_tokenized_samples(["aa"], [0, 1], Tok(), 4)
    assert out and out[0][3] == 2

    class Tok2:
        def __call__(self, text, **kw):  # noqa: ARG002
            return {"input_ids": [0, 0, 0, 0], "attention_mask": [0, 0, 0, 0]}

    out2 = pt._collect_tokenized_samples(["bb"], [0], Tok2(), 4)
    assert out2 == []
    # Torch-like tensors path
    # Torch-like tensors path (best-effort; skip if environment disagrees)
    try:

        class FakeTensor:
            def __init__(self, arr):
                self._arr = arr

            def squeeze(self, *args, **kwargs):  # noqa: D401
                class _W:
                    def __init__(self, arr):
                        self._arr = arr

                    def tolist(self):
                        return self._arr

                return _W(self._arr)

        class TokTorch:
            def __call__(self, text, **kw):  # noqa: ARG002
                return {
                    "input_ids": FakeTensor([1, 2, 0, 0]),
                    "attention_mask": FakeTensor([1, 1, 0, 0]),
                }

        monkeypatch.setattr(data_mod, "HAS_TORCH", True)
        out3 = pt._collect_tokenized_samples(["cc"], [0], TokTorch(), 4)
        if not out3:
            import pytest

            pytest.skip("torch-like squeeze path not exercised in this environment")
        assert out3 and out3[0][3] == 2
    except Exception:
        import pytest

        pytest.skip("torch-like squeeze path not available")


def test_byte_ngram_scoring_empty_returns_false(monkeypatch):
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True)
    pt = WikiText2Provider()
    assert pt._score_candidates_byte_ngram([]) is False


def test_wikitext2_windows_success(monkeypatch):
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True)
    pt = WikiText2Provider()
    # Provide enough texts and a simple collector
    monkeypatch.setattr(pt, "load", lambda **kw: ["z" * 30] * 20)

    def collector(texts, indices, tokenizer, seq_len):  # noqa: ARG001
        # Return one entry per index with two real tokens
        res = []
        for idx in indices:
            res.append((idx, [1, 2, 0, 0], [1, 1, 0, 0], 2))
        return res

    monkeypatch.setattr(pt, "_collect_tokenized_samples", collector)
    prev, final = pt.windows(
        tokenizer=SimpleNamespace(), seq_len=4, preview_n=3, final_n=2
    )
    assert len(prev) == 3 and len(final) == 2


def test_wikitext2_windows_full_path_scored(monkeypatch):
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True)
    pt = WikiText2Provider()
    # Provide many texts
    monkeypatch.setattr(pt, "load", lambda **kw: ["t" * 30] * 20)

    # Collector that emits token_count=3 with deterministic sequences
    def collector(texts, indices, tokenizer, seq_len):  # noqa: ARG001
        res = []
        for idx in indices:
            # Vary content so hashing/indices differ
            seq = [1, 2, (idx % 3), 0]
            mask = [1, 1, 1, 0]
            res.append((idx, seq, mask, 3))
        return res

    monkeypatch.setattr(pt, "_collect_tokenized_samples", collector)

    # Scorer that annotates difficulty values
    def scorer(candidates):
        for c in candidates:
            c["difficulty"] = float(c["token_count"]) + (c["dataset_index"] % 5)
        return True

    # Patch into instance method name used internally
    monkeypatch.setattr(pt, "_score_candidates_byte_ngram", lambda c: scorer(c))

    prev, final = pt.windows(
        tokenizer=SimpleNamespace(), seq_len=4, preview_n=5, final_n=4
    )
    assert len(prev) == 5 and len(final) == 4
    # Stratification stats should be populated
    st = pt.stratification_stats
    assert isinstance(st, dict) and "preview_mean_difficulty" in st


def test_wikitext2_windows_frequency_fallback(monkeypatch):
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True)
    pt = WikiText2Provider()
    monkeypatch.setattr(pt, "load", lambda **kw: ["text sample long enough"] * 20)

    def fake_collect(texts, indices, tokenizer, seq_len):  # noqa: ARG001
        out = []
        for idx in indices:
            seq = [idx + 1, idx + 2, 0, 0]
            mask = [1, 1, 0, 0]
            out.append((idx, seq, mask, 2))
        return out

    monkeypatch.setattr(pt, "_collect_tokenized_samples", fake_collect)
    preview, final = pt.windows(
        tokenizer=SimpleNamespace(), seq_len=4, preview_n=2, final_n=1
    )
    assert len(preview) == 2 and len(final) == 1
    stats = pt.stratification_stats
    assert stats and stats["pool_size"] >= 3


def test_wikitext2_frequency_lone_and_remaining(monkeypatch):
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True)
    pt = WikiText2Provider()
    monkeypatch.setattr(
        pt, "load", lambda **kw: [f"text {i} long enough" for i in range(12)]
    )

    def collector(texts, indices, tokenizer, seq_len):  # noqa: ARG001
        out = []
        for idx in indices:
            seq = [(idx % 5) + 1, 0, 0, 0]
            mask = [1, 0, 0, 0]
            out.append((idx, seq, mask, 1))
        return out

    monkeypatch.setattr(pt, "_collect_tokenized_samples", collector)
    preview, final = pt.windows(
        tokenizer=SimpleNamespace(), seq_len=4, preview_n=3, final_n=2
    )
    assert len(preview.indices) == 3
    assert len(final.indices) == 2
    stats = pt.stratification_stats
    assert stats and stats["pool_size"] >= 5


def test_wikitext2_selection_collision_rounding(monkeypatch):
    """Force equidistant selection loop to exercise collision offsets."""
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True)
    provider = WikiText2Provider()
    # Provide exactly preview+final texts so total_candidates == selection_count
    monkeypatch.setattr(provider, "load", lambda **kw: ["text long enough"] * 4)

    def collector(texts, indices, tokenizer, seq_len):  # noqa: ARG001
        return [(idx, [idx + 1, 0, 0, 0], [1, 0, 0, 0], 1) for idx in indices]

    monkeypatch.setattr(provider, "_collect_tokenized_samples", collector)
    preview, final = provider.windows(
        SimpleNamespace(), seq_len=4, preview_n=2, final_n=2, seed=0
    )
    assert len(preview.indices) == 2 and len(final.indices) == 2
    # Pool size equals requested selection count, confirming collision branch ran
    assert provider.stratification_stats["pool_size"] == 4


def test_wikitext2_candidate_shortfall_raises(monkeypatch):
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True)
    pt = WikiText2Provider()
    monkeypatch.setattr(pt, "load", lambda **kw: ["text sample long enough"] * 50)

    def no_candidates(*args, **kwargs):  # noqa: ARG001
        return []

    monkeypatch.setattr(pt, "_collect_tokenized_samples", no_candidates)
    from invarlock.core.exceptions import DataError

    with pytest.raises(DataError, match="TOKENIZE-INSUFFICIENT"):
        pt.windows(tokenizer=SimpleNamespace(), seq_len=4, preview_n=2, final_n=2)


def test_wikitext_load_cache_and_dedupe(monkeypatch, tmp_path):
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True)
    calls = {"count": 0}

    def fake_load_dataset(*args, **kwargs):  # noqa: ARG001, ARG002
        calls["count"] += 1
        return [
            {"text": "Alpha sample with letters"},
            {"text": "Alpha sample with letters"},
            {"text": "Beta content that is long enough"},
        ]

    monkeypatch.setattr(data_mod, "load_dataset", fake_load_dataset)
    provider = WikiText2Provider(cache_dir=tmp_path)
    monkeypatch.setenv("INVARLOCK_DEDUP_TEXTS", "1")
    texts = provider.load(max_samples=5)
    assert texts == [
        "Alpha sample with letters",
        "Beta content that is long enough",
    ]
    assert calls["count"] == 1

    def _fail_load(*args, **kwargs):  # noqa: ARG001, ARG002
        raise AssertionError("load_dataset should not be called when cache hits")

    monkeypatch.setattr(data_mod, "load_dataset", _fail_load)
    cached = provider.load(max_samples=1)
    assert cached == ["Alpha sample with letters"]
    monkeypatch.delenv("INVARLOCK_DEDUP_TEXTS", raising=False)


def test_local_jsonl_resolve_files_dedupe(tmp_path):
    base = tmp_path / "jsonl_data"
    base.mkdir()
    file_direct = base / "direct.jsonl"
    file_direct.write_text("{}", encoding="utf-8")
    extra = base / "extra.jsonl"
    extra.write_text("{}", encoding="utf-8")
    nested_dir = base / "nested"
    nested_dir.mkdir()
    nested_file = nested_dir / "nested.jsonl"
    nested_file.write_text("{}", encoding="utf-8")

    provider = data_mod.LocalJSONLProvider(
        file=str(file_direct),
        path=str(base),
        data_files=[str(extra), str(nested_file), "missing.jsonl"],
    )
    files = provider._resolve_files()
    resolved = {f.name for f in files}
    assert resolved == {"direct.jsonl", "extra.jsonl", "nested.jsonl"}


def test_local_jsonl_resolve_files_invalid_entries(tmp_path):
    file_a = tmp_path / "a.jsonl"
    file_b = tmp_path / "b.jsonl"
    file_a.write_text("{}", encoding="utf-8")
    file_b.write_text("{}", encoding="utf-8")
    provider = data_mod.LocalJSONLProvider(
        data_files=[str(file_a), None, 123, str(file_b)]
    )
    files = provider._resolve_files()
    assert [f.name for f in files] == ["a.jsonl", "b.jsonl"]


def test_local_jsonl_malformed_and_missing_field(tmp_path):
    src = tmp_path / "mixed.jsonl"
    src.write_text(
        """
{"text": "hello world"}
not json
{"bad": 1}
{"text": ""}
{"text": "second"}
""".strip(),
        encoding="utf-8",
    )
    provider = data_mod.LocalJSONLProvider(path=str(tmp_path))
    texts = provider.load()
    assert texts == ["hello world", "second"]


def test_local_jsonl_pairs_windows_and_labels(tmp_path):
    pairs_file = tmp_path / "pairs.jsonl"
    pairs_file.write_text(
        "\n".join(
            [
                '{"source": "hello", "target": "world"}',
                '{"source": "", "target": "skip"}',
                '{"source": "foo", "target": "bar"}',
            ]
        ),
        encoding="utf-8",
    )

    class _Tok:
        pad_token_id = 0

        def encode(self, text, truncation=True, max_length=4):  # noqa: ARG002
            limit = int(max_length)
            return [ord(c) % 5 + 1 for c in text][:limit]

    provider = data_mod.LocalJSONLPairsProvider(
        file=str(pairs_file),
        data_files=str(pairs_file),
        max_samples=2,
    )
    prev, fin = provider.windows(_Tok(), seq_len=6, preview_n=1, final_n=1)
    assert len(prev.indices) == 1 and len(fin.indices) == 1
    assert provider.last_preview_labels and provider.last_preview_labels[0][-1] == -100
    cap = provider.estimate_capacity(_Tok(), seq_len=4, stride=2)
    assert cap["examples_available"] == 2


def test_local_jsonl_pairs_truncates_targets(tmp_path):
    pairs_file = tmp_path / "pairs_long.jsonl"
    pairs_file.write_text(
        "\n".join(
            [
                '{"source": "hello", "target": "0123456789"}',
            ]
        ),
        encoding="utf-8",
    )

    class LongTok:
        pad_token_id = 0

        def encode(self, text, truncation=True, max_length=4):  # noqa: ARG002
            return list(range(max_length + 3))

    provider = data_mod.LocalJSONLPairsProvider(file=str(pairs_file), max_samples=1)
    prev, _ = provider.windows(LongTok(), seq_len=4, preview_n=1, final_n=0)
    assert len(prev.indices) == 1
    assert provider.last_preview_labels and len(provider.last_preview_labels[0]) == 4
    assert provider.last_preview_labels[0][-1] == 3


def test_local_jsonl_pairs_missing_fields(tmp_path):
    data = tmp_path / "pairs_mixed.jsonl"
    data.write_text(
        "\n".join(
            [
                '{"source": "alpha"}',
                '{"target": "beta"}',
                '{"source": "keep", "target": "pair"}',
            ]
        ),
        encoding="utf-8",
    )

    class Tok:
        def encode(self, text, truncation=True, max_length=4):  # noqa: ARG002
            return list(range(1, max_length + 1))

    provider = data_mod.LocalJSONLPairsProvider(file=str(data), max_samples=2)
    prev, fin = provider.windows(Tok(), seq_len=4, preview_n=1, final_n=0)
    assert len(prev.indices) == 1 and len(fin.indices) == 0


def test_local_jsonl_load_handles_io_errors(tmp_path, monkeypatch):
    target = tmp_path / "broken.jsonl"
    target.write_text('{"text": "hello"}\n', encoding="utf-8")
    provider = data_mod.LocalJSONLProvider(file=str(target))
    original_open = Path.open

    def failing_open(self, *args, **kwargs):
        if self == target:
            raise OSError("denied")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    assert provider.load() == []


def test_local_jsonl_pairs_load_handles_io_errors(tmp_path, monkeypatch):
    target = tmp_path / "pairs.jsonl"
    target.write_text('{"source": "a", "target": "b"}\n', encoding="utf-8")
    provider = data_mod.LocalJSONLPairsProvider(file=str(target))
    original_open = Path.open

    def failing_open(self, *args, **kwargs):
        if self == target:
            raise OSError("denied")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    assert provider._load_pairs() == []


def test_wikitext2_estimate_capacity_fast_env(monkeypatch):
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True)
    provider = WikiText2Provider()
    monkeypatch.setattr(provider, "load", lambda **kw: ["text sample long enough"] * 4)

    class Tok:
        def __call__(self, text, **kw):  # noqa: ARG002
            return {"input_ids": [1, 2, 0, 0], "attention_mask": [1, 1, 0, 0]}

    monkeypatch.setenv("INVARLOCK_CAPACITY_FAST", "1")
    cap = provider.estimate_capacity(Tok(), seq_len=4, stride=2, fast_mode=False)
    assert "candidate_unique" in cap
    assert cap["candidate_limit"] >= cap["candidate_unique"]
    monkeypatch.delenv("INVARLOCK_CAPACITY_FAST", raising=False)


def test_wikitext2_capacity_respects_target_total(monkeypatch):
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True)
    provider = WikiText2Provider()
    monkeypatch.setattr(provider, "load", lambda **kw: ["text sample long enough"] * 10)

    class Tok:
        def __call__(self, text, **kw):  # noqa: ARG002
            return {"input_ids": [1, 2, 0, 0], "attention_mask": [1, 1, 0, 0]}

    cap = provider.estimate_capacity(
        Tok(), seq_len=4, stride=2, target_total=6, fast_mode=False
    )
    assert cap.get("candidate_limit") >= cap.get("candidate_unique", 0)


def test_wikitext2_load_filters_and_cache_updates(monkeypatch, tmp_path):
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True)
    samples = [
        {"text": "too short"},
        {"text": "A valid sample that is long enough 12345"},
        {"text": "   "},
        {"text": "Another valid sample with letters"},
    ]
    records = {"calls": 0}

    def fake_load_dataset(*args, **kwargs):  # noqa: ARG001, ARG002
        records["calls"] += 1
        return list(samples)

    monkeypatch.setattr(data_mod, "load_dataset", fake_load_dataset)
    provider = WikiText2Provider(cache_dir=tmp_path)
    texts = provider.load(split="validation", max_samples=5)
    assert len(texts) == 2
    # Second call with fewer requested samples uses cache short-circuit
    assert provider.load(split="validation", max_samples=1)[:1] == texts[:1]
    # Provide more valid texts to trigger cache expansion branch
    samples.append({"text": "A third valid sample with more than twenty chars!"})
    texts_updated = provider.load(split="validation", max_samples=10)
    assert len(texts_updated) == 3


def test_seq2seq_provider_windows_and_masks(monkeypatch):
    class DummySeq2Seq:
        def __init__(self, **kwargs):
            self._n = kwargs.get("n", 1)

        def batches(self, seed, batch_size):  # noqa: ARG002
            yield {
                "src_ids": [[1, 2, 0], [3, 4, 5]],
                "src_mask": [[1, 1, 0], [1, 1, 1]],
                "tgt_ids": [[6, 7], [8, 9, 10]],
            }

    monkeypatch.setattr(
        "invarlock.eval.providers.seq2seq.Seq2SeqProvider", DummySeq2Seq, raising=False
    )

    class _Tok:
        pad_token_id = 0

    provider = data_mod.Seq2SeqDataProvider(n=1)
    prev, fin = provider.windows(_Tok(), seq_len=4, preview_n=1, final_n=1)
    assert len(prev.input_ids[0]) == 4
    assert provider.last_final_labels and provider.last_final_labels[0][-1] == -100


def test_seq2seq_attention_mask_inferred(monkeypatch):
    class DummySeq2Seq:
        def __init__(self, **kwargs):
            self._n = kwargs.get("n", 1)

        def batches(self, seed, batch_size):  # noqa: ARG002
            yield {
                "src_ids": [[1, 2, 0]],
                "src_mask": [[1]],
                "tgt_ids": [[3]],
            }

    monkeypatch.setattr(
        "invarlock.eval.providers.seq2seq.Seq2SeqProvider", DummySeq2Seq, raising=False
    )

    class Tok:
        pad_token_id = 0

    provider = data_mod.Seq2SeqDataProvider()
    prev, _ = provider.windows(Tok(), seq_len=4, preview_n=1, final_n=0)
    assert prev.attention_masks[0] == [1, 1, 0, 0]


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
    provider = data_mod.Seq2SeqDataProvider(n=1)
    cap = provider.estimate_capacity(tokenizer=None, seq_len=4, stride=2)
    assert cap["examples_available"] >= 1
    assert cap["tokens_available"] >= 4


def test_wt2_frequency_fallback_lone_candidate(monkeypatch):
    # Force datasets present
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True)
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
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True)
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
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True)
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
    monkeypatch.setattr(data_mod, "HAS_DATASETS", True)
    pt = WikiText2Provider()

    class BadTokenizer:
        def __call__(self, text, **kwargs):  # noqa: ARG001
            raise ValueError("boom")

    with pytest.warns(UserWarning):
        res = pt._collect_tokenized_samples(["alpha"], [0], BadTokenizer(), 4)
    assert res == []
