from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import invarlock.eval.data_hf_common as hf_common_mod
import invarlock.eval.data_hf_providers as hf_providers_mod
import invarlock.eval.data_hf_seq2seq as hf_seq2seq_mod
import invarlock.eval.data_providers as data_providers_mod
import invarlock.eval.data_stratification as stratification_mod
import invarlock.eval.data_support as data_support_mod
from invarlock.core.exceptions import DataError, ValidationError
from invarlock.eval.data_stratification import stratify_wikitext_candidates
from invarlock.eval.data_support import (
    estimate_wikitext2_capacity,
    score_candidates_byte_ngram,
)


def test_wikitext2_capacity_helper_fast_and_slow_paths(monkeypatch):
    calls: list[tuple[str, object]] = []

    def load_fn(*, split: str, max_samples: int):
        calls.append(("load", (split, max_samples)))
        return ["alpha", "beta", "gamma", "delta"]

    def collect_fn(texts, indices, tokenizer, seq_len):  # noqa: ARG001
        calls.append(("collect", tuple(indices)))
        return [(idx, [idx + 1, idx + 2], [1, 1], 2) for idx in indices]

    monkeypatch.setenv("INVARLOCK_CAPACITY_FAST", "1")
    fast = estimate_wikitext2_capacity(
        load_fn=load_fn,
        collect_tokenized_samples_fn=collect_fn,
        tokenizer=SimpleNamespace(),
        seq_len=8,
        stride=4,
    )
    assert fast["available_nonoverlap"] == 4
    assert fast["candidate_unique"] == 4
    assert calls == [("load", ("validation", 2000))]

    monkeypatch.delenv("INVARLOCK_CAPACITY_FAST", raising=False)
    calls.clear()
    slow = estimate_wikitext2_capacity(
        load_fn=load_fn,
        collect_tokenized_samples_fn=collect_fn,
        tokenizer=SimpleNamespace(),
        seq_len=8,
        stride=4,
        target_total=2,
    )
    assert slow["available_nonoverlap"] == 4
    assert slow["candidate_limit"] == 4
    assert slow["candidate_unique"] == 4
    assert calls[0] == ("load", ("validation", 2000))
    assert calls[1] == ("collect", (0, 1, 2, 3))
    assert calls[2] == ("collect", (0, 1, 2, 3))


def test_byte_ngram_helper_is_deterministic_and_mutates_candidates():
    candidates = [{"text": "alpha"}, {"text": None}]
    profile = score_candidates_byte_ngram(
        candidates,
        order=4,
        pad_token=256,
        alpha=1.0,
    )
    assert profile is not None
    assert profile["mode"] == "byte_ngram"
    assert all("difficulty" in candidate for candidate in candidates)

    candidates_second = [{"text": "alpha"}, {"text": None}]
    profile_second = score_candidates_byte_ngram(
        candidates_second,
        order=4,
        pad_token=256,
        alpha=1.0,
    )
    assert profile_second is not None
    assert [candidate["difficulty"] for candidate in candidates] == pytest.approx(
        [candidate["difficulty"] for candidate in candidates_second], rel=0, abs=0
    )


def test_stratification_helper_builds_balanced_windows():
    candidates = [
        {
            "dataset_index": idx,
            "difficulty": float(idx),
            "input_ids": [idx, idx + 1],
            "attention_mask": [1, 1],
        }
        for idx in range(8)
    ]
    preview, final, stats = stratify_wikitext_candidates(
        candidates,
        preview_n=3,
        final_n=3,
        reserve=2,
        batch_size_used=8,
    )
    assert len(preview) == 3
    assert len(final) == 3
    assert stats["pool_size"] == 6
    assert stats["batch_size_used"] == 8
    assert "difficulty_gap" in stats


def test_stratification_rejects_empty_or_undersized_candidate_pools() -> None:
    with pytest.raises(ValidationError, match="preview/final must be positive"):
        stratify_wikitext_candidates(
            [], preview_n=0, final_n=0, reserve=0, batch_size_used=0
        )

    candidate = {
        "dataset_index": 0,
        "difficulty": 1.0,
        "input_ids": [1],
        "attention_mask": [1],
    }
    with pytest.raises(DataError, match="candidate pool insufficient"):
        stratify_wikitext_candidates(
            [candidate], preview_n=1, final_n=1, reserve=0, batch_size_used=1
        )


def test_stratified_position_selection_never_duplicates_candidates() -> None:
    assert stratification_mod._select_stratified_positions(3, 5) == [0, 1, 2]
    assert stratification_mod._select_stratified_positions(8, 0) == []


def test_difficulty_balancer_handles_empty_equal_and_shared_candidates() -> None:
    stratification_mod._balance_candidate_difficulty([], [{"difficulty": 1.0}])
    assert stratification_mod._mean_difficulty([]) == 0.0

    preview = [{"difficulty": 1.0}]
    final = [{"difficulty": 1.0}]
    stratification_mod._balance_candidate_difficulty(preview, final)
    assert preview[0]["difficulty"] == final[0]["difficulty"]

    shared = {"difficulty": 2.0}
    preview = [shared]
    final = [shared]
    stratification_mod._balance_candidate_difficulty(preview, final)
    assert preview == final == [shared]


@pytest.mark.parametrize(
    ("preview_difficulties", "final_difficulties"),
    [
        ([0.0, 2.0], [8.0, 10.0]),
        ([8.0, 10.0], [0.0, 2.0]),
    ],
)
def test_difficulty_balancer_never_worsens_the_observed_gap(
    preview_difficulties: list[float], final_difficulties: list[float]
) -> None:
    preview = [{"difficulty": value} for value in preview_difficulties]
    final = [{"difficulty": value} for value in final_difficulties]
    old_gap = abs(
        stratification_mod._mean_difficulty(preview)
        - stratification_mod._mean_difficulty(final)
    )

    stratification_mod._balance_candidate_difficulty(preview, final)

    new_gap = abs(
        stratification_mod._mean_difficulty(preview)
        - stratification_mod._mean_difficulty(final)
    )
    assert new_gap <= old_gap


@pytest.mark.parametrize(("preview_n", "final_n"), [(0, 3), (3, 0)])
def test_stratification_supports_one_sided_windows(
    preview_n: int, final_n: int
) -> None:
    candidates = [
        {
            "dataset_index": idx,
            "difficulty": float(idx),
            "input_ids": [idx],
            "attention_mask": [1],
        }
        for idx in range(3)
    ]

    preview, final, stats = stratify_wikitext_candidates(
        candidates,
        preview_n=preview_n,
        final_n=final_n,
        reserve=0,
        batch_size_used=3,
    )

    assert len(preview) == preview_n
    assert len(final) == final_n
    if preview_n == 0:
        assert stats["preview_mean_difficulty"] == 0.0
        assert stats["preview_std_difficulty"] == 0.0
    else:
        assert stats["final_mean_difficulty"] == 0.0
        assert stats["final_std_difficulty"] == 0.0


def test_stratification_fails_closed_if_position_selector_underfills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [
        {
            "dataset_index": idx,
            "difficulty": float(idx),
            "input_ids": [idx],
            "attention_mask": [1],
        }
        for idx in range(2)
    ]
    monkeypatch.setattr(
        stratification_mod, "_select_stratified_positions", lambda *_args: [0]
    )

    with pytest.raises(DataError, match="candidate pool insufficient"):
        stratify_wikitext_candidates(
            candidates, preview_n=1, final_n=1, reserve=0, batch_size_used=2
        )


def test_stratification_fails_closed_when_requested_capacities_are_impossible() -> None:
    candidates = [
        {
            "dataset_index": idx,
            "difficulty": float(idx),
            "input_ids": [idx],
            "attention_mask": [1],
        }
        for idx in range(2)
    ]

    with pytest.raises(DataError, match="failed to allocate"):
        stratify_wikitext_candidates(
            candidates, preview_n=-1, final_n=3, reserve=0, batch_size_used=2
        )


def test_stratification_detects_window_constructor_cardinality_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BrokenWindow:
        def __init__(self, **_kwargs) -> None:
            pass

        def __len__(self) -> int:
            return 0

    candidates = [
        {
            "dataset_index": idx,
            "difficulty": float(idx),
            "input_ids": [idx],
            "attention_mask": [1],
        }
        for idx in range(2)
    ]
    monkeypatch.setattr(stratification_mod, "EvaluationWindow", _BrokenWindow)

    with pytest.raises(DataError, match="window stratification mismatch"):
        stratify_wikitext_candidates(
            candidates, preview_n=1, final_n=1, reserve=0, batch_size_used=2
        )


def test_hf_dataset_helpers_resolve_facade_and_nested_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    assert hf_common_mod.facade_attr("absent_attribute", sentinel) is sentinel
    assert hf_common_mod.field_value({"nested": {"value": 3}}, "nested.value") == 3
    assert hf_common_mod.field_value({"nested": "not-a-map"}, "nested.value") is None
    assert hf_common_mod.field_value({"value": 3}, "") is None

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        data_providers_mod,
        "load_dataset_with_cache_fallback",
        lambda *args, **kwargs: calls.append((args, kwargs)) or ["loaded"],
        raising=False,
    )
    assert hf_common_mod.load_dataset_from_facade("dataset", split="test") == ["loaded"]
    assert calls == [(("dataset",), {"split": "test"})]


def test_hf_dataset_helpers_call_configured_dependency_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []
    monkeypatch.setattr(
        hf_common_mod,
        "facade_attr",
        lambda _name, _fallback: messages.append,
    )

    hf_common_mod.require_dataset("datasets is required")

    assert messages == ["datasets is required"]


def test_hf_dataset_helper_falls_back_when_facade_is_not_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback = object()
    monkeypatch.delitem(sys.modules, "invarlock.eval.data_providers")

    assert hf_common_mod.facade_attr("anything", fallback) is fallback


def test_hf_text_provider_filters_invalid_rows_and_caches_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hf_providers_mod, "require_dataset", lambda _message: None)
    calls = 0

    def load_dataset(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return [
            {},
            {"text": None},
            {"text": "   "},
            {"text": "first"},
            {"text": "second"},
            {"text": "ignored after limit"},
        ]

    monkeypatch.setattr(hf_providers_mod, "load_dataset_from_facade", load_dataset)
    provider = hf_providers_mod.HFTextProvider(max_samples=2)

    assert provider.load(split="test") == ["first", "second"]
    assert provider.load(split="test") == ["first", "second"]
    assert calls == 1


def test_hf_text_unique_sample_collection_deduplicates_token_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hf_providers_mod, "require_dataset", lambda _message: None)
    provider = hf_providers_mod.HFTextProvider()

    def tokenize(_texts, _tokenizer, _seq_len, positions):
        by_position = {
            0: ([1], [1]),
            1: ([1, 2], [1, 1]),
            2: ([1, 2], [1, 1]),
            3: ([3, 4], [1, 1]),
        }
        pairs = [by_position[position] for position in positions]
        return data_support_mod.EvaluationWindow(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            list(positions),
        )

    monkeypatch.setattr(provider, "_simple_tokenize", tokenize)

    samples = provider._collect_unique_window_samples(
        ["a", "b", "c", "d"],
        object(),
        seq_len=2,
        positions=[0, 1, 2, 3],
        target_total=3,
    )
    assert [sample[0] for sample in samples] == [1, 3]
    assert provider._collect_unique_window_samples(
        ["a", "b", "c", "d"],
        object(),
        seq_len=2,
        positions=[0, 1, 2, 3],
        target_total=1,
    ) == [(1, [1, 2], [1, 1])]


def test_hf_text_provider_tokenizes_and_builds_disjoint_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hf_providers_mod, "require_dataset", lambda _message: None)
    provider = hf_providers_mod.HFTextProvider(max_samples=4)
    monkeypatch.setattr(
        hf_providers_mod,
        "tokenize_texts_padded",
        lambda texts, _tokenizer, _seq_len, *, positions: (
            [[position + 1, position + 2] for position in positions],
            [[1, 1] for _ in texts],
            list(positions),
        ),
    )
    tokenized = provider._simple_tokenize(["a", "b"], object(), 2, [4, 5])
    assert tokenized.indices == [4, 5]

    monkeypatch.setattr(provider, "load", lambda **_kwargs: ["a", "b", "c"])
    monkeypatch.setattr(
        provider,
        "_collect_unique_window_samples",
        lambda *_args, **_kwargs: [
            (0, [1, 2], [1, 1]),
            (1, [3, 4], [1, 1]),
            (2, [5, 6], [1, 1]),
        ],
    )
    preview, final = provider.windows(object(), preview_n=1, final_n=2)
    assert preview.indices == [0]
    assert final.indices == [1, 2]
    assert (
        provider.estimate_capacity(object(), seq_len=8, stride=4)["candidate_unique"]
        == 3
    )


def test_hf_text_windows_fail_closed_on_empty_invalid_and_duplicate_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hf_providers_mod, "require_dataset", lambda _message: None)
    provider = hf_providers_mod.HFTextProvider()
    monkeypatch.setattr(provider, "load", lambda **_kwargs: [])
    with pytest.raises(DataError, match="produced no samples"):
        provider.windows(object(), preview_n=1, final_n=1)

    monkeypatch.setattr(provider, "load", lambda **_kwargs: ["only"])
    with pytest.raises(ValidationError, match="must be positive"):
        provider.windows(object(), preview_n=0, final_n=0)

    monkeypatch.setattr(
        provider, "_collect_unique_window_samples", lambda *_a, **_k: []
    )
    with pytest.raises(DataError, match="enough unique tokenized samples"):
        provider.windows(object(), preview_n=1, final_n=1)


def test_hf_seq2seq_provider_filters_rows_applies_prefixes_and_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hf_seq2seq_mod, "require_dataset", lambda _message: None)
    calls = 0

    def load_dataset(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return [
            "not-a-row",
            {"source": "", "target": "missing source"},
            {"source": "missing target", "target": None},
            {"source": " alpha ", "target": " beta "},
            {"source": "gamma", "target": "delta"},
            {"source": "ignored", "target": "after limit"},
        ]

    monkeypatch.setattr(hf_seq2seq_mod, "load_dataset_from_facade", load_dataset)
    provider = hf_seq2seq_mod.HFSeq2SeqProvider(
        "public/dataset", src_prefix="Q: ", tgt_prefix="A: ", max_samples=2
    )

    assert provider._load_pairs("test") == [
        ("Q: alpha", "A: beta"),
        ("Q: gamma", "A: delta"),
    ]
    assert provider._load_pairs("test") == [
        ("Q: alpha", "A: beta"),
        ("Q: gamma", "A: delta"),
    ]
    assert calls == 1


def test_hf_seq2seq_splits_labels_by_original_preview_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hf_seq2seq_mod, "require_dataset", lambda _message: None)
    provider = hf_seq2seq_mod.HFSeq2SeqProvider("public/dataset")
    window = data_support_mod.EvaluationWindow(
        [[10], [20], [30]], [[1], [1], [1]], [2, 0, 1]
    )

    preview, final, preview_labels, final_labels = provider._split_by_preview_positions(
        window, [[102], [100], [101]], preview_positions=[0, 2]
    )

    assert preview.indices == [2, 0]
    assert final.indices == [1]
    assert preview_labels == [[102], [100]]
    assert final_labels == [[101]]


def test_hf_seq2seq_windows_reject_empty_pairs_and_record_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hf_seq2seq_mod, "require_dataset", lambda _message: None)
    provider = hf_seq2seq_mod.HFSeq2SeqProvider("public/dataset")
    monkeypatch.setattr(provider, "_load_pairs", lambda _split: [])
    with pytest.raises(DataError, match="produced no pairs"):
        provider.windows(object(), preview_n=1, final_n=1)

    monkeypatch.setattr(provider, "_load_pairs", lambda _split: [("q", "a")] * 3)
    capacity = provider.estimate_capacity(object(), seq_len=8, stride=4)
    assert capacity["examples_available"] == 3
    assert capacity["tokens_available"] == 24


def test_hf_seq2seq_windows_tokenize_combined_pairs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hf_seq2seq_mod, "require_dataset", lambda _message: None)
    provider = hf_seq2seq_mod.HFSeq2SeqProvider("public/dataset")
    monkeypatch.setattr(
        provider,
        "_load_pairs",
        lambda _split: [("q0", "a0"), ("q1", "a1"), ("q2", "a2")],
    )

    def tokenize(pairs, *, tokenizer, seq_len, positions):
        del pairs, tokenizer, seq_len
        return (
            data_support_mod.EvaluationWindow(
                [[position] for position in positions],
                [[1] for _ in positions],
                list(positions),
            ),
            [[position + 100] for position in positions],
        )

    monkeypatch.setattr(hf_seq2seq_mod, "tokenize_combined_pairs", tokenize)

    preview, final = provider.windows(
        object(), preview_n=1, final_n=2, seed=7, seq_len=8
    )

    assert len(preview) == 1
    assert len(final) == 2
    assert provider.last_preview_labels is not None
    assert provider.last_final_labels is not None
