from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from examples.integrations.evaluator_transaction import corpora

ROOT = Path(__file__).resolve().parents[2]


def test_quick_profile_remains_bound_to_the_existing_102_record_corpus() -> None:
    profile = corpora.corpus_profile("quick")
    records = json.loads(
        (ROOT / "examples/integrations/lm-evaluation-harness/records.json").read_text(
            encoding="utf-8"
        )
    )
    payload = corpora.records_jsonl(records)

    assert profile.record_count == 102
    assert profile.context_length == 64
    assert profile.dataset_sha256 == hashlib.sha256(payload).hexdigest()
    assert corpora.profile_for_dataset(payload) == profile


def test_flagship_profile_freezes_a_balanced_400_record_selection() -> None:
    profile = corpora.corpus_profile("flagship")
    selection = corpora.flagship_selection()

    assert profile.record_count == 400
    assert profile.context_length == 256
    assert len(selection.indices) == 400
    assert len(set(selection.indices)) == 400
    assert selection.indices == tuple(sorted(selection.indices))
    assert sum(item.sample_count for item in selection.strata) == 400
    assert {item.sample_count for item in selection.strata} == {100}
    assert corpora.index_digest(selection.indices) == selection.indices_sha256


def test_selection_ranking_is_deterministic_and_stratified() -> None:
    candidates = tuple(
        corpora.SelectionCandidate(
            source_index=index,
            prompt_token_count=5 if index < 5 else 15,
            source_line_sha256=hashlib.sha256(str(index).encode()).hexdigest(),
        )
        for index in range(10)
    )
    strata = (
        corpora.SelectionStratum(0, 9, 2),
        corpora.SelectionStratum(10, 20, 2),
    )

    first = corpora.derive_selected_indices(candidates, strata, seed="fixed-seed")
    second = corpora.derive_selected_indices(candidates, strata, seed="fixed-seed")

    assert first == second
    assert len(first) == 4
    assert sum(index < 5 for index in first) == 2
    assert sum(index >= 5 for index in first) == 2


def test_selection_rejects_overlapping_or_underfilled_strata() -> None:
    candidate = corpora.SelectionCandidate(
        source_index=1,
        prompt_token_count=5,
        source_line_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match="overlap"):
        corpora.derive_selected_indices(
            (candidate,),
            (
                corpora.SelectionStratum(0, 5, 1),
                corpora.SelectionStratum(5, 10, 1),
            ),
            seed="fixed-seed",
        )
    with pytest.raises(ValueError, match="enough eligible records"):
        corpora.derive_selected_indices(
            (candidate,),
            (corpora.SelectionStratum(0, 4, 1),),
            seed="fixed-seed",
        )


def test_lambada_projection_preserves_the_native_final_word_boundary() -> None:
    source = b'{"text":"The answer is Paris"}\n{"text":"A final word"}\n'

    records = corpora.project_lambada_records(source, (0, 1))

    assert records == [
        {
            "expected": " Paris",
            "id": "lambada-openai-0000",
            "prompt": "The answer is",
        },
        {
            "expected": " word",
            "id": "lambada-openai-0001",
            "prompt": "A final",
        },
    ]


@pytest.mark.parametrize(
    "source, message",
    [
        (b'{"text":"no-boundary"}\n', "final-word boundary"),
        (b'{"text":1}\n', "text field"),
        (b'{"text":"ok value"}\n{"text":"extra value"}\n', "source index"),
    ],
)
def test_lambada_projection_rejects_invalid_or_unselected_source_rows(
    source: bytes, message: str
) -> None:
    indices = (0,) if b"extra" not in source else (2,)
    with pytest.raises(ValueError, match=message):
        corpora.project_lambada_records(source, indices)


def test_profile_lookup_rejects_unknown_or_tampered_datasets() -> None:
    with pytest.raises(ValueError, match="unknown corpus profile"):
        corpora.corpus_profile("wide")
    with pytest.raises(ValueError, match="not a pinned evaluator corpus"):
        corpora.profile_for_dataset(b"{}\n")


def test_local_flagship_source_loader_enforces_size_hash_and_regular_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b'{"text":"A final word"}\n'
    metadata = {
        "byte_length": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "url": "https://invalid.example/source",
    }
    monkeypatch.setattr(corpora, "flagship_source", lambda: metadata)
    source = tmp_path / "source.jsonl"
    source.write_bytes(payload)

    assert corpora.load_flagship_source(source) == payload
    source.write_bytes(payload + b"x")
    with pytest.raises(RuntimeError, match="pinned size"):
        corpora.load_flagship_source(source)


def test_flagship_provenance_exposes_replay_inputs_without_copying_source_text() -> (
    None
):
    provenance = corpora.corpus_provenance(corpora.corpus_profile("flagship"))

    assert provenance["record_count"] == 400
    assert provenance["source"]["revision"] == (
        "900124bf3b8235c6daf21033af9948b3f07346c4"
    )
    assert provenance["selection"]["indices_sha256"] == (
        "3e4c040854483e76b851283b8157ef3b0243efa9dde8a35059026aef4da0707f"
    )
    assert "indices" not in provenance["selection"]
