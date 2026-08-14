from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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
    records = corpora.qualification_records(profile)
    payload = corpora.records_jsonl(records, compact=True)

    assert profile.record_count == 400
    assert profile.context_length == 1024
    assert profile.dataset_name == "TIGER-Lab/MMLU-Pro/qwen35-no-think"
    assert profile.split == "test-balanced-400"
    assert profile.dataset_sha256 == hashlib.sha256(payload).hexdigest()
    assert {record["expected"] for record in records} == set("ABCDEFGHIJ")
    assert {
        answer: sum(record["expected"] == answer for record in records)
        for answer in "ABCDEFGHIJ"
    } == dict.fromkeys("ABCDEFGHIJ", 40)
    assert corpora.profile_for_dataset(payload) == profile


def test_deployment_profile_freezes_a_tokenizer_qualified_400_record_suite() -> None:
    profile = corpora.corpus_profile("deployment")
    records = corpora.qualification_records(profile)
    payload = corpora.records_jsonl(records)

    assert profile.profile_id == "lambada-openai-qwen35-0.8b-400-v1"
    assert profile.record_count == 400
    assert profile.context_length == 256
    assert profile.dataset_sha256 == (
        "e4a0e431b8b64130cbbf6e8fb3ed7b5769744d18ca6499d2088f2e1b3fb36dda"
    )
    assert profile.dataset_sha256 == hashlib.sha256(payload).hexdigest()
    assert all(record["id"].startswith("lambada-openai-") for record in records)
    assert all(record["expected"].startswith(" ") for record in records)
    assert len({record["id"] for record in records}) == 400
    assert corpora.profile_for_dataset(payload) == profile
    assert profile.acceptance_policy()["resolved_policy"]["metrics"]["exact_match"] == {
        "delta_min_pp": -20.0,
        "maximum_interval_width_pp": 10.0,
        "minimum_record_count": 400,
        "minimum_side_accuracy": 0.05,
    }


def test_deployment_provenance_binds_source_selection_and_manifest() -> None:
    provenance = corpora.corpus_provenance(corpora.corpus_profile("deployment"))

    assert provenance["source"] == {
        "repository": "EleutherAI/lambada_openai",
        "revision": "900124bf3b8235c6daf21033af9948b3f07346c4",
        "path": "data/lambada_test.jsonl",
        "license": "MIT",
        "byte_length": 1819752,
        "record_count": 5153,
        "sha256": "4aa8d02cd17c719165fc8a7887fddd641f43fcafa4b1c806ca8abc31fabdb226",
    }
    assert provenance["selection"]["algorithm"] == "stratified-sha256-v1"
    assert provenance["selection"]["eligible_record_count"] == 4052
    assert len(provenance["selection"]["indices"]) == 400
    assert provenance["selection_manifest"] == {
        "path": "deployment_corpus.json",
        "byte_length": 6644,
        "sha256": "a774b4369658d6f6c4910b03968008c59e55a3cd04b3737c34373074f583df77",
    }
    assert provenance["model_profile"] == ("qwen35-0.8b-base-to-post-trained-bf16-v1")


def test_portability_profile_renders_the_same_semantic_ids_for_gemma() -> None:
    qwen = corpora.qualification_records(corpora.corpus_profile("flagship"))
    profile = corpora.corpus_profile("portability")
    gemma = corpora.qualification_records(profile)
    payload = corpora.records_jsonl(gemma, compact=True)

    assert profile.profile_id == "mmlu-pro-gemma4-12b-no-think-400-v1"
    assert profile.dataset_sha256 == hashlib.sha256(payload).hexdigest()
    assert [record["id"] for record in gemma] == [record["id"] for record in qwen]
    assert [record["expected"] for record in gemma] == [
        record["expected"] for record in qwen
    ]
    assert all(
        record["prompt"].startswith("<bos><|turn>system\n")
        and record["prompt"].endswith("<|turn>model\n<|channel>thought\n<channel|>")
        for record in gemma
    )
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
        "manifest_sha256": "1cb979170d16328b02b69b32d0ab9670365064ba3c112eed001515c549334d44",
        "selection_algorithm": "balanced-bipartite-sha256-v1",
        "semantic_artifact": "text_semantic_bank",
        "semantic_byte_length": 339013,
        "semantic_sha256": "18a88db999d8157ef051fee0eac4ad48b291853c970a5dc709d40b35e2da4430",
    }
    assert provenance["model_profile"] == (
        "qwen35-9b-base-to-post-trained-bf16-singleton-v1"
    )
    assert provenance["rendering"] == {
        "algorithm": "qwen-chatml-disable-thinking-v1",
        "suffix": "<think>\n\n</think>\n\n",
    }


def test_flagship_records_are_loaded_from_the_pinned_bundled_artifact() -> None:
    records = corpora.flagship_records()
    no_thinking_suffix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    assert len(records) == 400
    assert records[0]["id"] == "mmlu_pro_00075"
    assert records[-1]["id"] == "mmlu_pro_12236"
    assert records[0]["prompt"].startswith("<|im_start|>system\n")
    assert all(record["prompt"].endswith(no_thinking_suffix) for record in records)


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


def test_json_manifest_loader_rejects_unreadable_and_nonobject_payloads(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    sequence = tmp_path / "sequence.json"
    sequence.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="test manifest is unavailable or invalid"):
        corpora._json(malformed, label="test manifest")
    with pytest.raises(ValueError, match="test manifest must contain an object"):
        corpora._json(sequence, label="test manifest")


def test_flagship_provenance_rejects_suite_metadata_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = corpora.corpus_profile("flagship")
    records = corpora.qualification_records(profile)
    qualification = corpora._qualification_manifest()
    changed = json.loads(json.dumps(qualification))
    changed["selected_ids"]["text"][0] = "substituted-record"
    monkeypatch.setattr(corpora, "qualification_records", lambda _profile: records)
    monkeypatch.setattr(corpora, "_qualification_manifest", lambda: changed)

    with pytest.raises(ValueError, match="disagrees with its qualification suite"):
        corpora.corpus_provenance(profile)


def test_secure_corpus_reader_requires_nofollow_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "records.jsonl"
    source.write_bytes(b"{}\n")
    monkeypatch.delattr(corpora.os, "O_NOFOLLOW", raising=False)

    with pytest.raises(RuntimeError, match="secure bundled corpus loading"):
        corpora._read_regular_file(source, expected_bytes=3)


def test_secure_corpus_reader_reads_exact_bytes_and_detects_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "records.jsonl"
    payload = b'{"id":"one"}\n'
    source.write_bytes(payload)

    assert corpora._read_regular_file(source, expected_bytes=len(payload)) == payload

    original_fstat = corpora.os.fstat
    calls = 0

    def changed_fstat(descriptor: int) -> object:
        nonlocal calls
        calls += 1
        observed = original_fstat(descriptor)
        if calls == 1:
            return observed
        return SimpleNamespace(
            st_mode=observed.st_mode,
            st_size=observed.st_size,
            st_dev=observed.st_dev,
            st_ino=observed.st_ino,
            st_mtime_ns=observed.st_mtime_ns + 1,
            st_ctime_ns=observed.st_ctime_ns,
        )

    monkeypatch.setattr(corpora.os, "fstat", changed_fstat)
    with pytest.raises(RuntimeError, match="changed while being read"):
        corpora._read_regular_file(source, expected_bytes=len(payload))


def test_deployment_manifest_and_records_reject_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(corpora, "_read_regular_file", lambda *_args, **_kwargs: b"{}")
    with pytest.raises(RuntimeError, match="manifest does not match"):
        corpora._deployment_manifest()

    manifest = json.loads(
        (
            ROOT / "examples/integrations/evaluator_transaction/deployment_corpus.json"
        ).read_text(encoding="utf-8")
    )
    monkeypatch.setattr(corpora, "_deployment_manifest", lambda: manifest)
    with pytest.raises(RuntimeError, match="corpus does not match"):
        corpora.deployment_records()


def test_deployment_records_reject_selection_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = corpora._deployment_manifest()
    records = corpora.deployment_records()
    records[0]["id"] = "lambada-openai-substituted"
    payload = corpora.records_jsonl(records)
    changed = json.loads(json.dumps(manifest))
    changed["derived_dataset"]["byte_length"] = len(payload)
    changed["derived_dataset"]["sha256"] = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(corpora, "_deployment_manifest", lambda: changed)
    monkeypatch.setattr(
        corpora, "_read_regular_file", lambda *_args, **_kwargs: payload
    )

    with pytest.raises(RuntimeError, match="selected source rows"):
        corpora.deployment_records()


def test_semantic_corpus_rejects_digest_and_shape_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(corpora, "_read_regular_file", lambda *_args, **_kwargs: b"")
    with pytest.raises(RuntimeError, match="pinned identity"):
        corpora._semantic_records()

    payload = b'{"id":"incomplete"}\n'
    manifest = corpora._manifest()
    changed = json.loads(json.dumps(manifest))
    changed["qualification_suite"]["semantic_byte_length"] = len(payload)
    changed["qualification_suite"]["semantic_sha256"] = hashlib.sha256(
        payload
    ).hexdigest()
    monkeypatch.setattr(corpora, "_manifest", lambda: changed)
    monkeypatch.setattr(
        corpora, "_read_regular_file", lambda *_args, **_kwargs: payload
    )
    with pytest.raises(RuntimeError, match="semantic corpus is incomplete"):
        corpora._semantic_records()


def test_semantic_rendering_rejects_invalid_choices_and_profile_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = {
        "answer": "A",
        "id": "one",
        "options": ["only choice"],
        "question": "Invalid?",
    }
    with pytest.raises(RuntimeError, match="invalid choices"):
        corpora._question_body(invalid)

    valid = {**invalid, "options": ["first", "second"]}
    with pytest.raises(ValueError, match="does not use the semantic corpus"):
        corpora._render_record(valid, _profile(record_count=1))
    with pytest.raises(ValueError, match="maintained GPU profile"):
        corpora.qualification_records(corpora.corpus_profile("quick"))

    mismatched = replace(
        corpora.corpus_profile("flagship"),
        record_count=1,
        dataset_sha256="0" * 64,
    )
    monkeypatch.setattr(corpora, "_semantic_records", lambda: [valid])
    with pytest.raises(RuntimeError, match="pinned identity"):
        corpora.qualification_records(mismatched)
