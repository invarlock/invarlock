from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from examples.integrations.evaluator_transaction import corpora

ROOT = Path(__file__).resolve().parents[2]


def _profile(
    *, record_count: int, dataset_sha256: str = "0" * 64
) -> corpora.CorpusProfile:
    return corpora.CorpusProfile(
        key="test",
        profile_id="test-v1",
        dataset_name="test",
        split="test",
        record_count=record_count,
        dataset_sha256=dataset_sha256,
        context_length=256,
        minimum_side_accuracy=0.05,
        maximum_interval_width_pp=10.0,
    )


class _FakeTokenizer:
    def __init__(self) -> None:
        self._decoded: dict[tuple[int, ...], str] = {}

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        if add_special_tokens:
            count = 300 if text.startswith("long ") else 10
            return {"input_ids": list(range(count))}
        if text == " multi":
            return {"input_ids": [1, 2]}
        token_ids = [100 + len(self._decoded)]
        self._decoded[tuple(token_ids)] = text
        return {"input_ids": token_ids}

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        value = self._decoded[tuple(token_ids)]
        return " changed" if value == " lossy" else value


def _install_synthetic_flagship(
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    source_rows = [
        {"text": "alpha answer"},
        {"text": "beta result"},
        {"text": "no-boundary"},
        {"text": "gamma multi"},
        {"text": "delta lossy"},
        {"text": "long prompt huge"},
    ]
    source_payload = corpora.records_jsonl(source_rows)
    lines = source_payload.splitlines()
    strata = (corpora.SelectionStratum(0, 256, 2),)
    candidates = tuple(
        corpora.SelectionCandidate(
            source_index=index,
            prompt_token_count=10,
            source_line_sha256=hashlib.sha256(lines[index]).hexdigest(),
        )
        for index in (0, 1)
    )
    indices = corpora.derive_selected_indices(candidates, strata, seed="synthetic")
    records = corpora.project_lambada_records(source_payload, indices)
    dataset_payload = corpora.records_jsonl(records)
    selection = corpora.FlagshipSelection(
        seed="synthetic",
        eligible_record_count=2,
        strata=strata,
        indices=indices,
        indices_sha256=corpora.index_digest(indices),
        selected_source_lines_sha256=hashlib.sha256(
            "".join(
                hashlib.sha256(lines[index]).hexdigest() for index in indices
            ).encode("ascii")
        ).hexdigest(),
    )
    profile = corpora.CorpusProfile(
        key="flagship",
        profile_id="synthetic-flagship-v1",
        dataset_name="synthetic-lambada",
        split="test",
        record_count=2,
        dataset_sha256=hashlib.sha256(dataset_payload).hexdigest(),
        context_length=256,
        minimum_side_accuracy=0.05,
        maximum_interval_width_pp=10.0,
    )
    source = {
        "byte_length": len(source_payload),
        "record_count": len(source_rows),
        "sha256": hashlib.sha256(source_payload).hexdigest(),
        "url": "https://invalid.example/source",
    }
    monkeypatch.setattr(corpora, "flagship_source", lambda: source)
    monkeypatch.setattr(corpora, "flagship_selection", lambda: selection)
    monkeypatch.setattr(corpora, "corpus_profile", lambda _key: profile)
    return SimpleNamespace(
        profile=profile,
        records=records,
        selection=selection,
        source=source,
        source_payload=source_payload,
        tokenizers=(_FakeTokenizer(), _FakeTokenizer()),
    )


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


def test_profiles_expose_closed_descriptors_policies_and_quick_provenance() -> None:
    quick = corpora.corpus_profile("quick")

    assert corpora.profile_for_descriptor(quick.dataset_descriptor()) == quick
    assert quick.acceptance_policy()["resolved_policy"]["metrics"]["exact_match"] == {
        "delta_min_pp": -20.0,
        "maximum_interval_width_pp": 20.0,
        "minimum_record_count": 102,
        "minimum_side_accuracy": 0.2,
    }
    assert corpora.corpus_provenance(quick) == {
        "dataset_name": quick.dataset_name,
        "dataset_sha256": quick.dataset_sha256,
        "profile_id": quick.profile_id,
        "record_count": 102,
    }
    changed = quick.dataset_descriptor()
    changed["path"] = "elsewhere.jsonl"
    with pytest.raises(ValueError, match="descriptor is not a pinned"):
        corpora.profile_for_descriptor(changed)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"{\n", "valid JSONL"),
        (b"\xff\n", "valid JSONL"),
        (b'{"expected":"x","id":"one"}\n', "complete records"),
        (b'{"expected":"x","id":"one","prompt":""}\n', "complete records"),
        (
            b'{"prompt":"p","id":"one","expected":"x"}\n',
            "not canonical",
        ),
    ],
)
def test_dataset_validation_rejects_malformed_incomplete_or_noncanonical_records(
    payload: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        corpora.validate_dataset_records(payload, _profile(record_count=1))


def test_dataset_validation_rejects_duplicate_record_ids() -> None:
    payload = corpora.records_jsonl(
        [
            {"expected": " one", "id": "same", "prompt": "first"},
            {"expected": " two", "id": "same", "prompt": "second"},
        ]
    )

    with pytest.raises(ValueError, match="IDs are not unique"):
        corpora.validate_dataset_records(payload, _profile(record_count=2))


def test_manifest_selection_and_source_metadata_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    monkeypatch.setattr(corpora, "_FLAGSHIP_MANIFEST", manifest_path)

    manifest_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain an object"):
        corpora._manifest()

    manifest_path.write_text(
        json.dumps(
            {
                "selection": {
                    "eligible_record_count": 1,
                    "indices": [1],
                    "indices_sha256": "0" * 64,
                    "seed": "seed",
                    "selected_source_lines_sha256": "0" * 64,
                    "strata": [
                        {
                            "maximum_prompt_tokens": 10,
                            "minimum_prompt_tokens": 0,
                            "sample_count": 1,
                        }
                    ],
                },
                "source": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="indices do not match"):
        corpora.flagship_selection()
    with pytest.raises(ValueError, match="source metadata is invalid"):
        corpora.flagship_source()


def test_selection_rejects_duplicate_source_records_across_strata() -> None:
    candidates = (
        corpora.SelectionCandidate(1, 5, "a" * 64),
        corpora.SelectionCandidate(1, 15, "b" * 64),
    )
    strata = (
        corpora.SelectionStratum(0, 9, 1),
        corpora.SelectionStratum(10, 20, 1),
    )

    with pytest.raises(ValueError, match="duplicate records"):
        corpora.derive_selected_indices(candidates, strata, seed="fixed")


def test_lambada_projection_rejects_invalid_json_source_row() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        corpora.project_lambada_records(b"{\n", (0,))


def test_local_flagship_source_loader_rejects_unsafe_or_changed_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b'{"text":"A final word"}\n'
    metadata = {
        "byte_length": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "url": "https://invalid.example/source",
    }
    monkeypatch.setattr(corpora, "flagship_source", lambda: metadata)

    with pytest.raises(RuntimeError, match="opened safely"):
        corpora.load_flagship_source(tmp_path / "missing.jsonl")
    with pytest.raises(RuntimeError, match="pinned size"):
        corpora.load_flagship_source(tmp_path)

    wrong = tmp_path / "wrong.jsonl"
    wrong.write_bytes(b"x" * len(payload))
    with pytest.raises(RuntimeError, match="pinned identity"):
        corpora.load_flagship_source(wrong)

    source = tmp_path / "source.jsonl"
    source.write_bytes(payload)
    real_fstat = corpora.os.fstat
    calls = 0

    def changed_fstat(descriptor: int) -> Any:
        nonlocal calls
        calls += 1
        observed = real_fstat(descriptor)
        if calls == 1:
            return observed
        return SimpleNamespace(
            st_ctime_ns=observed.st_ctime_ns,
            st_dev=observed.st_dev,
            st_ino=observed.st_ino,
            st_mode=observed.st_mode,
            st_mtime_ns=observed.st_mtime_ns + 1,
            st_size=observed.st_size,
        )

    monkeypatch.setattr(corpora.os, "fstat", changed_fstat)
    with pytest.raises(RuntimeError, match="changed while being read"):
        corpora.load_flagship_source(source)


def test_local_flagship_source_loader_rejects_platform_without_nofollow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(corpora.os, "O_NOFOLLOW", None)
    monkeypatch.setattr(
        corpora,
        "flagship_source",
        lambda: {"byte_length": 1, "sha256": "0" * 64, "url": "https://invalid"},
    )
    with pytest.raises(RuntimeError, match="secure benchmark source loading"):
        corpora.load_flagship_source(tmp_path / "source")


def test_network_flagship_source_loader_is_bounded_and_identity_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b'{"text":"A final word"}\n'
    metadata = {
        "byte_length": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "url": "https://example.test/source",
    }
    monkeypatch.setattr(corpora, "flagship_source", lambda: metadata)
    requested: list[tuple[str, int]] = []

    class Response:
        def __init__(self, body: bytes) -> None:
            self.body = body
            self.offset = 0

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, size: int) -> bytes:
            chunk = self.body[self.offset : self.offset + size]
            self.offset += len(chunk)
            return chunk

    def urlopen(request: Any, *, timeout: int) -> Response:
        requested.append((request.full_url, timeout))
        return Response(payload)

    monkeypatch.setattr(corpora.urllib.request, "urlopen", urlopen)
    assert corpora.load_flagship_source() == payload
    assert requested == [(metadata["url"], 120)]

    monkeypatch.setattr(
        corpora.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(payload + b"x"),
    )
    with pytest.raises(RuntimeError, match="pinned identity"):
        corpora.load_flagship_source()


def test_flagship_records_apply_all_eligibility_filters_and_reproduce_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _install_synthetic_flagship(monkeypatch)

    assert corpora.flagship_records(fixture.source_payload, fixture.tokenizers) == (
        fixture.records
    )


def test_flagship_records_reject_source_or_tokenizer_set_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _install_synthetic_flagship(monkeypatch)

    with pytest.raises(ValueError, match="pinned identity"):
        corpora.flagship_records(fixture.source_payload + b"x", fixture.tokenizers)
    with pytest.raises(ValueError, match="tokenizer set is incomplete"):
        corpora.flagship_records(fixture.source_payload, fixture.tokenizers[:1])


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (b"{\n", "not valid JSON"),
        (b'{"text":1}\n', "invalid text field"),
    ],
)
def test_flagship_records_reject_malformed_source_rows(
    row: bytes, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = {
        "byte_length": len(row),
        "record_count": 1,
        "sha256": hashlib.sha256(row).hexdigest(),
    }
    monkeypatch.setattr(corpora, "flagship_source", lambda: metadata)
    with pytest.raises(ValueError, match=message):
        corpora.flagship_records(row, (_FakeTokenizer(), _FakeTokenizer()))


def test_flagship_records_bind_selection_line_and_dataset_digests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _install_synthetic_flagship(monkeypatch)

    changed_count = corpora.FlagshipSelection(
        seed=fixture.selection.seed,
        eligible_record_count=3,
        strata=fixture.selection.strata,
        indices=fixture.selection.indices,
        indices_sha256=fixture.selection.indices_sha256,
        selected_source_lines_sha256=fixture.selection.selected_source_lines_sha256,
    )
    monkeypatch.setattr(corpora, "flagship_selection", lambda: changed_count)
    with pytest.raises(ValueError, match="selection does not match"):
        corpora.flagship_records(fixture.source_payload, fixture.tokenizers)

    changed_lines = corpora.FlagshipSelection(
        seed=fixture.selection.seed,
        eligible_record_count=fixture.selection.eligible_record_count,
        strata=fixture.selection.strata,
        indices=fixture.selection.indices,
        indices_sha256=fixture.selection.indices_sha256,
        selected_source_lines_sha256="0" * 64,
    )
    monkeypatch.setattr(corpora, "flagship_selection", lambda: changed_lines)
    with pytest.raises(ValueError, match="selected source lines"):
        corpora.flagship_records(fixture.source_payload, fixture.tokenizers)

    monkeypatch.setattr(corpora, "flagship_selection", lambda: fixture.selection)
    changed_profile = corpora.CorpusProfile(
        key=fixture.profile.key,
        profile_id=fixture.profile.profile_id,
        dataset_name=fixture.profile.dataset_name,
        split=fixture.profile.split,
        record_count=fixture.profile.record_count,
        dataset_sha256="0" * 64,
        context_length=fixture.profile.context_length,
        minimum_side_accuracy=fixture.profile.minimum_side_accuracy,
        maximum_interval_width_pp=fixture.profile.maximum_interval_width_pp,
    )
    monkeypatch.setattr(corpora, "corpus_profile", lambda _key: changed_profile)
    with pytest.raises(ValueError, match="derived dataset"):
        corpora.flagship_records(fixture.source_payload, fixture.tokenizers)


def test_quick_records_revalidate_the_checked_in_corpus() -> None:
    records = corpora.quick_records()

    assert len(records) == 102
    assert records[0]["id"] == "causal-cloze-000"
