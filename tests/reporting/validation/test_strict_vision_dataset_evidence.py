from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from invarlock.eval.data import VisionTextProvider
from invarlock.reporting.verify_strict_vision import (
    append_strict_vision_evidence_errors,
)
from invarlock.vision_dataset_evidence import (
    build_report_evidence_from_run_report,
    canonical_json_bytes,
    semantic_digest,
    validate_dataset_evidence,
    validate_evaluation_materialization_binding,
)
from tests.scripts._support_model_evidence import load_script_module


class _FakeImage:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def save(self, handle, *, format: str) -> None:
        del format
        handle.write(self.payload)


@pytest.fixture(scope="module")
def strict_vision_lifecycle(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, object]:
    root = tmp_path_factory.mktemp("strict-vision-v1")
    materializer = load_script_module("materialize_vision_text_dataset")
    config = materializer.MaterializeConfig(
        dataset="public/vision-benchmark",
        split="validation",
        revision="a" * 40,
        config_name="default",
        image_field="image",
        prompt_field="question",
        answer_field="answer",
        answers_field=None,
        id_field="id",
        prompt_template="{question}",
        max_samples=800,
        seed=42,
        shuffle=False,
        image_format="png",
    )
    rows = [
        {
            "id": f"record-{index:04d}",
            "question": f"Question {index}?",
            "answer": f"answer-{index}",
            "image": _FakeImage(f"image-{index}".encode()),
        }
        for index in range(800)
    ]
    summary = materializer.materialize_rows(rows, output_dir=root, config=config)
    provider = VisionTextProvider(path=str(root / "manifest.jsonl"))
    examples = provider.examples()
    identity = {
        "tokenizer_sha256": "sha256:" + ("1" * 64),
        "processor_sha256": "sha256:" + ("2" * 64),
        "chat_template_sha256": "sha256:" + ("3" * 64),
    }

    def window(arm_examples: list[dict[str, object]]) -> dict[str, object]:
        bindings = [
            {
                "id": item["id"],
                "dataset_record_sha256": item["dataset_record_sha256"],
                "image_sha256": item["image_sha256"],
                "materialization_digest": item["materialization_digest"],
                "manifest_sha256": item["manifest_sha256"],
                "record_sha256": item["record_sha256"],
            }
            for item in arm_examples
        ]
        return {
            "input_records": copy.deepcopy(bindings),
            "processor_identity": copy.deepcopy(identity),
            "records": copy.deepcopy(bindings),
        }

    run_report = {
        "meta": {"seed": 42},
        "evaluation_windows": {
            "preview": window(examples[:400]),
            "final": window(examples[400:]),
        },
    }
    evidence = build_report_evidence_from_run_report(run_report)
    assert evidence is not None
    evaluation_report = {
        "dataset": {"provider": "vision_text"},
        "dataset_evidence": evidence,
        "evaluation_windows": run_report["evaluation_windows"],
        "provenance": {"provider_digest": {"dataset_evidence": evidence}},
    }
    return {
        "evaluation_report": evaluation_report,
        "root": root,
        "summary": summary,
    }


def _refresh_digest(evidence: dict[str, object]) -> None:
    evidence.pop("semantic_digest", None)
    evidence["semantic_digest"] = semantic_digest(evidence)


def test_materialization_to_report_to_strict_verifier_lifecycle(
    strict_vision_lifecycle: dict[str, object],
) -> None:
    report = strict_vision_lifecycle["evaluation_report"]
    assert isinstance(report, dict)
    errors: list[str] = []
    append_strict_vision_evidence_errors(errors, report)
    assert errors == []

    evidence = report["dataset_evidence"]
    assert isinstance(evidence, dict)
    assert evidence["sampling"] == {
        "final": 400,
        "preview": 400,
        "seed": 42,
        "shuffle": False,
        "total": 800,
    }
    records = evidence["records"]
    assert isinstance(records, list)
    assert {record["id"] for record in records[:400]}.isdisjoint(
        {record["id"] for record in records[400:]}
    )


def test_catalog_binding_rejects_a_self_consistent_different_record_set(
    strict_vision_lifecycle: dict[str, object],
) -> None:
    root = strict_vision_lifecycle["root"]
    report = strict_vision_lifecycle["evaluation_report"]
    assert isinstance(root, Path)
    assert isinstance(report, dict)
    materialization = json.loads(
        (root / "dataset_evidence.json").read_text(encoding="utf-8")
    )
    evaluation = copy.deepcopy(report["dataset_evidence"])
    assert isinstance(evaluation, dict)
    evaluation["records"] = [
        {
            **record,
            "id": f"different-{index:04d}",
            "dataset_record_sha256": "sha256:" + ("a" * 64),
            "image_sha256": "b" * 64,
            "record_sha256": "sha256:" + ("c" * 64),
        }
        for index, record in enumerate(evaluation["records"])
    ]
    _refresh_digest(evaluation)

    errors = validate_evaluation_materialization_binding(
        materialization,
        evaluation,
        strict_counts=True,
    )

    assert "evaluation dataset_evidence records do not match materialization" in errors


def test_report_evidence_requires_canonical_record_id(
    strict_vision_lifecycle: dict[str, object],
) -> None:
    report = copy.deepcopy(strict_vision_lifecycle["evaluation_report"])
    record = report["evaluation_windows"]["preview"]["input_records"][0]
    record["example_id"] = record.pop("id")

    assert build_report_evidence_from_run_report(report) is None


def test_report_evidence_requires_one_exact_manifest_snapshot(
    strict_vision_lifecycle: dict[str, object],
) -> None:
    report = copy.deepcopy(strict_vision_lifecycle["evaluation_report"])
    record = report["evaluation_windows"]["final"]["input_records"][0]
    record["manifest_sha256"] = "sha256:" + ("f" * 64)

    assert build_report_evidence_from_run_report(report) is None


def test_semantic_digest_excludes_timestamps_and_ids_are_length_safe(
    strict_vision_lifecycle: dict[str, object],
) -> None:
    summary = strict_vision_lifecycle["summary"]
    assert isinstance(summary, dict)
    before = summary["semantic_digest"]
    timestamp_variant = copy.deepcopy(summary)
    timestamp_variant["generated_at"] = "2099-01-01T00:00:00Z"
    assert (
        validate_dataset_evidence(
            timestamp_variant,
            strict_counts=True,
            require_runtime_identity=False,
            allow_materialization_summary_fields=True,
        )
        == []
    )
    assert timestamp_variant["semantic_digest"] == before
    assert semantic_digest(["1", "23"]) != semantic_digest(["12", "3"])
    assert canonical_json_bytes(["1", "23"]) == b'["1","23"]'


def test_materialization_outputs_are_public_safe(
    strict_vision_lifecycle: dict[str, object],
) -> None:
    root = strict_vision_lifecycle["root"]
    assert isinstance(root, Path)
    manifest = (root / "manifest.jsonl").read_text(encoding="utf-8")
    summary = (root / "materialization_summary.json").read_text(encoding="utf-8")
    evidence = (root / "dataset_evidence.json").read_text(encoding="utf-8")
    for payload in (manifest, summary, evidence):
        assert str(root) not in payload
        assert "/Users/" not in payload
        assert "/private/" not in payload


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ("v2", "requires dataset_evidence.v1"),
        ("unbound_hash", "record_sha256 is invalid"),
        ("missing_processor", "requires runtime_identity"),
        ("missing_template", "requires runtime_identity"),
        ("missing_scope", "exact v1 attestation_scope"),
        ("forked_schedule", "exactly 800/400/400"),
        ("path_leak", "local or private paths"),
        ("unsupported_field", "unsupported fields"),
    ],
)
def test_strict_verifier_rejects_legacy_omission_and_tamper(
    strict_vision_lifecycle: dict[str, object],
    mutation: str,
    expected: str,
) -> None:
    report = copy.deepcopy(strict_vision_lifecycle["evaluation_report"])
    assert isinstance(report, dict)
    evidence = report["dataset_evidence"]
    assert isinstance(evidence, dict)
    if mutation == "v2":
        evidence["schema"] = "dataset_evidence.v2"
        _refresh_digest(evidence)
    elif mutation == "unbound_hash":
        del evidence["records"][0]["record_sha256"]
        _refresh_digest(evidence)
    elif mutation == "missing_template":
        del evidence["runtime_identity"]["chat_template_sha256"]
        _refresh_digest(evidence)
    elif mutation == "missing_processor":
        del evidence["runtime_identity"]["processor_sha256"]
        _refresh_digest(evidence)
    elif mutation == "missing_scope":
        del evidence["attestation_scope"]
        _refresh_digest(evidence)
    elif mutation == "forked_schedule":
        evidence["records"] = evidence["records"][:400]
        for record in evidence["records"][200:]:
            record["arm"] = "final"
        evidence["sampling"] = {
            "final": 200,
            "preview": 200,
            "seed": 42,
            "shuffle": False,
            "total": 400,
        }
        _refresh_digest(evidence)
    elif mutation == "unsupported_field":
        evidence["claimed_quality"] = "perfect"
        _refresh_digest(evidence)
    else:
        evidence["attestation_scope"]["local_path"] = "/opt/example-data"

    report["provenance"]["provider_digest"]["dataset_evidence"] = copy.deepcopy(
        evidence
    )
    errors: list[str] = []
    append_strict_vision_evidence_errors(errors, report)
    assert expected in "; ".join(errors)


def test_strict_verifier_rejects_window_binding_tamper(
    strict_vision_lifecycle: dict[str, object],
) -> None:
    report = copy.deepcopy(strict_vision_lifecycle["evaluation_report"])
    report["evaluation_windows"]["preview"]["records"][0]["image_sha256"] = "0" * 64
    errors: list[str] = []
    append_strict_vision_evidence_errors(errors, report)
    assert any("not materialization-bound" in error for error in errors)


def test_strict_verifier_rejects_window_manifest_byte_mismatch(
    strict_vision_lifecycle: dict[str, object],
) -> None:
    report = copy.deepcopy(strict_vision_lifecycle["evaluation_report"])
    report["evaluation_windows"]["preview"]["input_records"][0]["manifest_sha256"] = (
        "sha256:" + ("f" * 64)
    )
    errors: list[str] = []
    append_strict_vision_evidence_errors(errors, report)
    assert any("does not bind manifest bytes" in error for error in errors)


def test_strict_verifier_ignores_non_vision_reports() -> None:
    errors: list[str] = []
    append_strict_vision_evidence_errors(errors, {"dataset": "not-a-record"})
    append_strict_vision_evidence_errors(
        errors,
        {"dataset": {"provider": " local_jsonl "}},
    )
    assert errors == []


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("provenance", "exactly match provenance.provider_digest"),
        ("windows", "requires evaluation_windows"),
        ("arm", "requires evaluation_windows.preview"),
        ("identity", "processor_identity must match dataset_evidence"),
        ("record_lists", "requires bound input/output records"),
        ("count", "record count does not match dataset_evidence"),
        ("malformed_input", "record 0 is malformed"),
        ("input_binding", "input record 0 is not materialization-bound"),
        ("output_binding", "output record 0 is not materialization-bound"),
        ("output_manifest", "output record 0 does not bind manifest bytes"),
    ],
)
def test_strict_verifier_rejects_incomplete_or_forked_runtime_windows(
    strict_vision_lifecycle: dict[str, object],
    mutation: str,
    expected: str,
) -> None:
    report = copy.deepcopy(strict_vision_lifecycle["evaluation_report"])
    assert isinstance(report, dict)
    if mutation == "provenance":
        report["provenance"] = {"provider_digest": {}}
    elif mutation == "windows":
        report["evaluation_windows"] = []
    elif mutation == "arm":
        report["evaluation_windows"]["preview"] = []
    elif mutation == "identity":
        report["evaluation_windows"]["preview"]["processor_identity"] = {}
    elif mutation == "record_lists":
        report["evaluation_windows"]["preview"]["input_records"] = {}
    elif mutation == "count":
        report["evaluation_windows"]["preview"]["records"].pop()
    elif mutation == "malformed_input":
        report["evaluation_windows"]["preview"]["input_records"][0] = "bad"
    elif mutation == "input_binding":
        report["evaluation_windows"]["preview"]["input_records"][0]["id"] = (
            "wrong-record"
        )
    elif mutation == "output_binding":
        report["evaluation_windows"]["preview"]["records"][0]["id"] = "wrong-record"
    else:
        report["evaluation_windows"]["preview"]["records"][0]["manifest_sha256"] = (
            "sha256:" + ("f" * 64)
        )

    errors: list[str] = []
    append_strict_vision_evidence_errors(errors, report)
    assert expected in "; ".join(errors)


def test_provider_rejects_post_attestation_image_mutation(
    strict_vision_lifecycle: dict[str, object],
) -> None:
    root = strict_vision_lifecycle["root"]
    assert isinstance(root, Path)
    first_record = json.loads(
        (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    image_path = root / first_record["image_path"]
    original = image_path.read_bytes()
    image_path.write_bytes(b"mutated-image-bytes")
    try:
        with pytest.raises(Exception, match="image_sha256 is invalid"):
            VisionTextProvider(path=str(root / "manifest.jsonl")).examples()
    finally:
        image_path.write_bytes(original)


@pytest.mark.parametrize("field", ["dataset_record_sha256", "record_sha256"])
def test_provider_rejects_manifest_binding_mutation(
    strict_vision_lifecycle: dict[str, object],
    field: str,
) -> None:
    root = strict_vision_lifecycle["root"]
    assert isinstance(root, Path)
    manifest_path = root / "manifest.jsonl"
    original = manifest_path.read_text(encoding="utf-8")
    lines = original.splitlines()
    record = json.loads(lines[0])
    record["source"][field] = "sha256:" + ("f" * 64)
    lines[0] = json.dumps(record, sort_keys=True)
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        with pytest.raises(Exception, match="manifest bytes do not match"):
            VisionTextProvider(path=str(manifest_path)).examples()
    finally:
        manifest_path.write_text(original, encoding="utf-8")


@pytest.mark.parametrize("field", ["prompt", "answers"])
def test_provider_rejects_manifest_content_mutation(
    strict_vision_lifecycle: dict[str, object],
    field: str,
) -> None:
    root = strict_vision_lifecycle["root"]
    assert isinstance(root, Path)
    manifest_path = root / "manifest.jsonl"
    original = manifest_path.read_text(encoding="utf-8")
    lines = original.splitlines()
    record = json.loads(lines[0])
    if field == "prompt":
        record["prompt"] = "tampered prompt"
    else:
        record["answers"][0] = "tampered answer"
        record["answer"] = "tampered answer"
    lines[0] = json.dumps(record, sort_keys=True)
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        with pytest.raises(Exception, match="manifest bytes do not match"):
            VisionTextProvider(path=str(manifest_path)).examples()
    finally:
        manifest_path.write_text(original, encoding="utf-8")
