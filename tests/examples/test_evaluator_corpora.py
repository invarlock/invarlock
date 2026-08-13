from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
        delta_min_pp=-2.0,
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
    assert profile.acceptance_policy()["resolved_policy"]["metrics"]["exact_match"] == {
        "delta_min_pp": -20.0,
        "maximum_interval_width_pp": 20.0,
        "minimum_record_count": 102,
        "minimum_side_accuracy": 0.20,
    }
    assert corpora.profile_for_dataset(payload) == profile


def test_flagship_profile_freezes_the_balanced_400_record_suite() -> None:
    profile = corpora.corpus_profile("flagship")
    corpus_path = (
        ROOT
        / "examples/integrations/evaluator_transaction/mmlu_pro_qwen_instruct_400.jsonl"
    )
    payload = corpus_path.read_bytes()
    records = [json.loads(line) for line in payload.splitlines()]

    assert profile.record_count == 400
    assert profile.context_length == 1024
    assert profile.dataset_name == "TIGER-Lab/MMLU-Pro"
    assert profile.split == "test-balanced-400"
    assert profile.dataset_sha256 == hashlib.sha256(payload).hexdigest()
    assert {record["expected"] for record in records} == set("ABCDEFGHIJ")
    assert {
        answer: sum(record["expected"] == answer for record in records)
        for answer in "ABCDEFGHIJ"
    } == dict.fromkeys("ABCDEFGHIJ", 40)
    assert corpora.profile_for_dataset(payload) == profile


def test_flagship_policy_is_frozen_before_model_execution() -> None:
    policy = corpora.corpus_profile("flagship").acceptance_policy()

    assert policy["resolved_policy"]["metrics"]["exact_match"] == {
        "delta_min_pp": -2.0,
        "maximum_interval_width_pp": 10.0,
        "minimum_record_count": 400,
        "minimum_side_accuracy": 0.20,
    }


def test_flagship_provenance_binds_the_shared_qualification_suite() -> None:
    provenance = corpora.corpus_provenance(corpora.corpus_profile("flagship"))

    assert provenance["source"] == {
        "dataset": "TIGER-Lab/MMLU-Pro",
        "license": "MIT",
        "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "split": "test",
    }
    assert provenance["qualification_suite"] == {
        "artifact": "text_qwen_instruct",
        "artifact_sha256": "52b568fcbead27884b1c8e375c4e05111bcae25e40000e23c770675869e4a5b8",
        "manifest_sha256": "1cb979170d16328b02b69b32d0ab9670365064ba3c112eed001515c549334d44",
        "selection_algorithm": "balanced-bipartite-sha256-v1",
    }


def test_flagship_records_are_loaded_from_the_pinned_bundled_artifact() -> None:
    records = corpora.flagship_records()

    assert len(records) == 400
    assert records[0]["id"] == "mmlu_pro_00075"
    assert records[-1]["id"] == "mmlu_pro_12236"
    assert records[0]["prompt"].startswith("<|im_start|>system\n")
    assert records[0]["prompt"].endswith("<|im_start|>assistant\n")


def test_profile_lookup_rejects_unknown_and_tampered_material() -> None:
    with pytest.raises(ValueError, match="unknown corpus profile"):
        corpora.corpus_profile("wide")
    with pytest.raises(ValueError, match="not a pinned evaluator corpus"):
        corpora.profile_for_dataset(b"{}\n")
    with pytest.raises(ValueError, match="not a pinned evaluator corpus"):
        corpora.profile_for_descriptor({"sha256": "0" * 64})


def test_dataset_validation_rejects_malformed_duplicate_and_noncanonical_rows() -> None:
    profile = _profile(record_count=2)
    records = [
        {"expected": "A", "id": "one", "prompt": "Question one"},
        {"expected": "B", "id": "two", "prompt": "Question two"},
    ]
    canonical = corpora.records_jsonl(records)

    corpora.validate_dataset_records(canonical, profile)
    with pytest.raises(ValueError, match="valid JSONL"):
        corpora.validate_dataset_records(b"{\n", profile)
    with pytest.raises(ValueError, match="2 complete records"):
        corpora.validate_dataset_records(corpora.records_jsonl(records[:1]), profile)
    duplicate = [records[0], {**records[1], "id": "one"}]
    with pytest.raises(ValueError, match="not unique"):
        corpora.validate_dataset_records(corpora.records_jsonl(duplicate), profile)
    with pytest.raises(ValueError, match="not canonical"):
        corpora.validate_dataset_records(
            b'{"expected":"A","id":"one","prompt":"Question one"}\n'
            b'{"expected":"B","id":"two","prompt":"Question two"}\n',
            profile,
        )


def test_compact_jsonl_is_deterministic_and_unicode_preserving() -> None:
    records = [{"prompt": "café", "id": "one", "expected": "A"}]

    assert corpora.records_jsonl(records, compact=True) == (
        b'{"expected":"A","id":"one","prompt":"caf\xc3\xa9"}\n'
    )


def test_secure_corpus_reader_rejects_symlinks_and_wrong_sizes(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    source.write_bytes(b"{}\n")
    link = tmp_path / "link.jsonl"
    link.symlink_to(source)

    with pytest.raises(RuntimeError, match="opened safely"):
        corpora._read_regular_file(link, expected_bytes=3)
    with pytest.raises(RuntimeError, match="pinned size"):
        corpora._read_regular_file(source, expected_bytes=4)


def test_flagship_provenance_rejects_qualification_manifest_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changed = tmp_path / "qualification.json"
    changed.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(corpora, "_QUALIFICATION_MANIFEST", changed)

    with pytest.raises(ValueError, match="pinned digest"):
        corpora.corpus_provenance(corpora.corpus_profile("flagship"))
