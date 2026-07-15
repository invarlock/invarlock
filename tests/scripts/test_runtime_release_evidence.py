from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts.release import runtime_release_evidence as evidence

SOURCE_COMMIT = "a" * 40
SOURCE_ARCHIVE_SHA256 = "b" * 64
IMAGE_DIGEST = "sha256:" + "c" * 64


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_json(path: Path, value: object, *, newline: bool = False) -> Path:
    path.write_bytes(_canonical(value) + (b"\n" if newline else b""))
    return path


def _gguf_summary(path: Path) -> Path:
    return _write_json(
        path,
        {
            "evidence_sha256": "d" * 64,
            "fixture_revision": "e" * 40,
            "format_version": evidence.GGUF_FORMAT,
            "image_digest": IMAGE_DIGEST,
            "runs": 2,
            "status": "ok",
        },
        newline=True,
    )


def _tensorrt_summary(path: Path) -> Path:
    return _write_json(
        path,
        {
            "candidate_image_digest": IMAGE_DIGEST,
            "engine_bundle_tree_sha256": "1" * 64,
            "format_version": evidence.TENSORRT_FORMAT,
            "gpu_count": 2,
            "ok": True,
            "output_sha256": "2" * 64,
            "runtime_provider_receipt_sha256": "3" * 64,
            "tokenizer_sha256": "4" * 64,
        },
        newline=True,
    )


def _side(prefix: str) -> dict[str, str]:
    def digest(label: str) -> str:
        return hashlib.sha256(f"{prefix}-{label}".encode()).hexdigest()

    return {
        "artifact_identity_sidecar_sha256": digest("artifact"),
        "evaluation_report_sha256": digest("report"),
        "provider_receipt_sidecar_sha256": digest("provider"),
        "runtime_manifest_sha256": digest("manifest"),
        "scoring_observation_sidecar_sha256": digest("observation"),
    }


def _behavior_receipt(path: Path) -> Path:
    return _write_json(
        path,
        {
            "baseline": _side("1"),
            "baseline_score": 1.0,
            "claim_set": evidence.BEHAVIOR_CLAIM_SET,
            "format_version": evidence.BEHAVIOR_RECEIPT_FORMAT,
            "metric": "exact_match",
            "policy_digest": "sha256:" + "6" * 64,
            "regression": 0.25,
            "schedule_sha256": "7" * 64,
            "subject": _side("8"),
            "subject_score": 0.75,
            "verdict": "pass",
        },
    )


def _build(tmp_path: Path, *, behavior: bool = True) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    output = tmp_path / "runtime-release-evidence.tar.gz"
    evidence.build_asset(
        output=output,
        source_commit=SOURCE_COMMIT,
        source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        qualification_summaries={
            "llama_cpp": _gguf_summary(tmp_path / "gguf.json"),
            "tensorrt_llm": _tensorrt_summary(tmp_path / "tensorrt.json"),
        },
        behavioral_receipts=(
            [_behavior_receipt(tmp_path / "behavior.json")] if behavior else []
        ),
    )
    return output


def _build_multiple_qualifications(tmp_path: Path, *, reverse: bool = False) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    qualifications = [
        (
            "llama_cpp:cpu-reference",
            _gguf_summary(tmp_path / "gguf.json"),
        ),
        (
            "tensorrt_llm:pair-a",
            _tensorrt_summary(tmp_path / "tensorrt-a.json"),
        ),
        (
            "tensorrt_llm:pair-b",
            _write_json(
                tmp_path / "tensorrt-b.json",
                {
                    "candidate_image_digest": IMAGE_DIGEST,
                    "engine_bundle_tree_sha256": "5" * 64,
                    "format_version": evidence.TENSORRT_FORMAT,
                    "gpu_count": 2,
                    "ok": True,
                    "output_sha256": "6" * 64,
                    "runtime_provider_receipt_sha256": "7" * 64,
                    "tokenizer_sha256": "4" * 64,
                },
                newline=True,
            ),
        ),
    ]
    if reverse:
        qualifications.reverse()
    output = tmp_path / "runtime-release-evidence.tar.gz"
    evidence.build_asset(
        output=output,
        source_commit=SOURCE_COMMIT,
        source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        qualification_summaries=dict(qualifications),
        behavioral_receipts=[],
    )
    return output


def test_asset_is_deterministic_closed_and_distinguishes_claim_scopes(
    tmp_path: Path,
) -> None:
    first = _build(tmp_path / "first")
    second = _build(tmp_path / "second")

    assert first.read_bytes() == second.read_bytes()
    result = evidence.validate_asset(
        first,
        expected_source_commit=SOURCE_COMMIT,
        expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        expected_providers=frozenset({"llama_cpp", "tensorrt_llm"}),
        require_behavioral_claim=True,
    )
    assert result["status"] == "ok"
    assert result["qualification_count"] == 2
    assert result["behavioral_claim_count"] == 1

    files = evidence._read_archive(first.read_bytes())
    behavior_paths = {name for name in files if name.startswith("receipts/claim-")}
    assert len(behavior_paths) == 1
    assert set(files) - behavior_paths == {
        "index.json",
        "receipts/llama_cpp-qualification.json",
        "receipts/tensorrt_llm-qualification.json",
        "summaries/llama_cpp-qualification-summary.json",
        "summaries/tensorrt_llm-qualification-summary.json",
    }
    index = json.loads(files["index.json"])
    assert {entry["claim_scope"] for entry in index["qualifications"]} == {
        evidence.QUALIFICATION_SCOPE
    }
    assert index["behavioral_claims"][0]["claim_scope"] == evidence.BEHAVIOR_SCOPE
    public_payload = b"".join(files.values())
    for forbidden in (
        b"/root/",
        b"/Users/",
        b"raw.log",
    ):
        assert forbidden not in public_payload
    gguf_summary = json.loads(files["summaries/llama_cpp-qualification-summary.json"])
    tensorrt_summary = json.loads(
        files["summaries/tensorrt_llm-qualification-summary.json"]
    )
    assert gguf_summary["fixture_revision"] == "e" * 40
    assert gguf_summary["evidence_sha256"] == "d" * 64
    assert tensorrt_summary["engine_bundle_tree_sha256"] == "1" * 64
    assert tensorrt_summary["output_sha256"] == "2" * 64
    assert tensorrt_summary["runtime_provider_receipt_sha256"] == "3" * 64
    assert tensorrt_summary["tokenizer_sha256"] == "4" * 64


def test_asset_supports_multiple_named_qualifications_per_provider(
    tmp_path: Path,
) -> None:
    first = _build_multiple_qualifications(tmp_path / "first")
    second = _build_multiple_qualifications(tmp_path / "second", reverse=True)

    assert first.read_bytes() == second.read_bytes()
    result = evidence.validate_asset(
        first,
        expected_source_commit=SOURCE_COMMIT,
        expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        expected_providers=frozenset({"llama_cpp", "tensorrt_llm"}),
        expected_qualifications=frozenset(
            {
                "llama_cpp:cpu-reference",
                "tensorrt_llm:pair-a",
                "tensorrt_llm:pair-b",
            }
        ),
    )
    assert result["qualification_count"] == 3
    assert result["qualified_provider_count"] == 2

    files = evidence._read_archive(first.read_bytes())
    assert set(files) == {
        "index.json",
        "receipts/llama_cpp-cpu-reference-qualification.json",
        "receipts/tensorrt_llm-pair-a-qualification.json",
        "receipts/tensorrt_llm-pair-b-qualification.json",
        "summaries/llama_cpp-cpu-reference-qualification-summary.json",
        "summaries/tensorrt_llm-pair-a-qualification-summary.json",
        "summaries/tensorrt_llm-pair-b-qualification-summary.json",
    }
    index = json.loads(files["index.json"])
    assert [
        (entry["provider_name"], entry["qualification_name"])
        for entry in index["qualifications"]
    ] == [
        ("llama_cpp", "cpu-reference"),
        ("tensorrt_llm", "pair-a"),
        ("tensorrt_llm", "pair-b"),
    ]
    for entry in index["qualifications"]:
        receipt = json.loads(files[entry["receipt_path"]])
        assert receipt["qualification_name"] == entry["qualification_name"]


def test_exact_qualification_expectation_rejects_missing_named_run(
    tmp_path: Path,
) -> None:
    asset = _build_multiple_qualifications(tmp_path)
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="qualification set"):
        evidence.validate_asset(
            asset,
            expected_source_commit=SOURCE_COMMIT,
            expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
            expected_qualifications=frozenset(
                {"llama_cpp:cpu-reference", "tensorrt_llm:pair-a"}
            ),
        )


@pytest.mark.parametrize(
    "qualification_key",
    [
        "tensorrt_llm:../pair-a",
        "tensorrt_llm:PAIR-A",
        "tensorrt_llm:pair/a",
        "tensorrt_llm:pair a",
        "tensorrt_llm:",
        "tensorrt_llm:pair-a:extra",
        "tensorrt_llm:" + "a" * 33,
    ],
)
def test_builder_rejects_unsafe_or_noncanonical_qualification_names(
    tmp_path: Path, qualification_key: str
) -> None:
    with pytest.raises(
        evidence.RuntimeReleaseEvidenceError, match="qualification name"
    ):
        evidence.build_asset(
            output=tmp_path / "asset.tar.gz",
            source_commit=SOURCE_COMMIT,
            source_archive_sha256=SOURCE_ARCHIVE_SHA256,
            qualification_summaries={
                qualification_key: _tensorrt_summary(tmp_path / "tensorrt.json")
            },
            behavioral_receipts=[],
        )


def test_builder_rejects_legacy_named_mix_and_accepts_reproducible_summaries(
    tmp_path: Path,
) -> None:
    summary = _tensorrt_summary(tmp_path / "tensorrt.json")
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="all be named"):
        evidence.build_asset(
            output=tmp_path / "mixed.tar.gz",
            source_commit=SOURCE_COMMIT,
            source_archive_sha256=SOURCE_ARCHIVE_SHA256,
            qualification_summaries={
                "tensorrt_llm": summary,
                "tensorrt_llm:pair-a": summary,
            },
            behavioral_receipts=[],
        )

    result = evidence.build_asset(
        output=tmp_path / "reproduced.tar.gz",
        source_commit=SOURCE_COMMIT,
        source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        qualification_summaries={
            "tensorrt_llm:pair-a": summary,
            "tensorrt_llm:pair-b": summary,
        },
        behavioral_receipts=[],
    )
    assert result["qualification_count"] == 2


def test_cli_builds_and_pins_multiple_named_qualifications(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    gguf = _gguf_summary(tmp_path / "gguf.json")
    pair_a = _tensorrt_summary(tmp_path / "pair-a.json")
    pair_b = _write_json(
        tmp_path / "pair-b.json",
        {
            "candidate_image_digest": IMAGE_DIGEST,
            "engine_bundle_tree_sha256": "5" * 64,
            "format_version": evidence.TENSORRT_FORMAT,
            "gpu_count": 2,
            "ok": True,
            "output_sha256": "6" * 64,
            "runtime_provider_receipt_sha256": "7" * 64,
            "tokenizer_sha256": "4" * 64,
        },
        newline=True,
    )
    asset = tmp_path / "asset.tar.gz"

    assert (
        evidence.main(
            [
                "build",
                "--source-commit",
                SOURCE_COMMIT,
                "--source-archive-sha256",
                SOURCE_ARCHIVE_SHA256,
                "--qualification",
                f"llama_cpp:cpu-reference={gguf}",
                "--qualification",
                f"tensorrt_llm:pair-a={pair_a}",
                "--qualification",
                f"tensorrt_llm:pair-b={pair_b}",
                "--output",
                str(asset),
            ]
        )
        == 0
    )
    build_result = json.loads(capsys.readouterr().out)
    assert build_result["qualification_count"] == 3

    assert (
        evidence.main(
            [
                "validate",
                "--asset",
                str(asset),
                "--expected-source-commit",
                SOURCE_COMMIT,
                "--expected-source-archive-sha256",
                SOURCE_ARCHIVE_SHA256,
                "--expected-provider",
                "llama_cpp",
                "--expected-provider",
                "tensorrt_llm",
                "--expected-qualification",
                "llama_cpp:cpu-reference",
                "--expected-qualification",
                "tensorrt_llm:pair-a",
                "--expected-qualification",
                "tensorrt_llm:pair-b",
            ]
        )
        == 0
    )
    validation_result = json.loads(capsys.readouterr().out)
    assert validation_result["qualification_count"] == 3
    assert validation_result["qualified_provider_count"] == 2


def test_validator_binds_names_and_accepts_reproducible_named_qualification(
    tmp_path: Path,
) -> None:
    asset = _build_multiple_qualifications(tmp_path / "source")
    original = evidence._read_archive(asset.read_bytes())
    receipt_path = "receipts/tensorrt_llm-pair-b-qualification.json"

    renamed = dict(original)
    receipt = json.loads(renamed[receipt_path])
    receipt["qualification_name"] = "pair-x"
    renamed[receipt_path] = _canonical(receipt)
    index = json.loads(renamed["index.json"])
    pair_b_entry = next(
        item
        for item in index["qualifications"]
        if item.get("qualification_name") == "pair-b"
    )
    pair_b_entry["receipt_sha256"] = evidence._sha256(renamed[receipt_path])
    renamed["index.json"] = _canonical(index)
    renamed_asset = tmp_path / "renamed.tar.gz"
    evidence._write_archive(renamed_asset, renamed)
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="binding"):
        evidence.validate_asset(
            renamed_asset,
            expected_source_commit=SOURCE_COMMIT,
            expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        )

    duplicated = dict(original)
    pair_a_summary_path = "summaries/tensorrt_llm-pair-a-qualification-summary.json"
    pair_b_summary_path = "summaries/tensorrt_llm-pair-b-qualification-summary.json"
    duplicated[pair_b_summary_path] = duplicated[pair_a_summary_path]
    duplicated_receipt = evidence._qualification_receipt(
        provider_name="tensorrt_llm",
        qualification_name="pair-b",
        summary_payload=duplicated[pair_b_summary_path],
        source_commit=SOURCE_COMMIT,
        source_archive_sha256=SOURCE_ARCHIVE_SHA256,
    )
    duplicated[receipt_path] = _canonical(duplicated_receipt)
    index = json.loads(duplicated["index.json"])
    pair_b_entry = next(
        item
        for item in index["qualifications"]
        if item.get("qualification_name") == "pair-b"
    )
    pair_b_entry["summary_sha256"] = evidence._sha256(duplicated[pair_b_summary_path])
    pair_b_entry["receipt_sha256"] = evidence._sha256(duplicated[receipt_path])
    duplicated["index.json"] = _canonical(index)
    duplicated_asset = tmp_path / "duplicated.tar.gz"
    evidence._write_archive(duplicated_asset, duplicated)
    result = evidence.validate_asset(
        duplicated_asset,
        expected_source_commit=SOURCE_COMMIT,
        expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        expected_qualifications=frozenset(
            {
                "llama_cpp:cpu-reference",
                "tensorrt_llm:pair-a",
                "tensorrt_llm:pair-b",
            }
        ),
    )
    assert result["qualification_count"] == 3


def test_validation_requires_independent_source_asset_and_provider_bindings(
    tmp_path: Path,
) -> None:
    asset = _build(tmp_path)
    digest = evidence._sha256(asset.read_bytes())

    evidence.validate_asset(
        asset,
        expected_source_commit=SOURCE_COMMIT,
        expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        expected_asset_sha256=digest,
        expected_providers=frozenset({"llama_cpp", "tensorrt_llm"}),
    )
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="source binding"):
        evidence.validate_asset(
            asset,
            expected_source_commit="f" * 40,
            expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        )
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="asset digest"):
        evidence.validate_asset(
            asset,
            expected_source_commit=SOURCE_COMMIT,
            expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
            expected_asset_sha256="f" * 64,
        )
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="provider set"):
        evidence.validate_asset(
            asset,
            expected_source_commit=SOURCE_COMMIT,
            expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
            expected_providers=frozenset({"llama_cpp"}),
        )


def test_qualification_only_asset_cannot_satisfy_behavior_requirement(
    tmp_path: Path,
) -> None:
    asset = _build(tmp_path, behavior=False)

    result = evidence.validate_asset(
        asset,
        expected_source_commit=SOURCE_COMMIT,
        expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
    )
    assert result["behavioral_claim_count"] == 0
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="schedule-level"):
        evidence.validate_asset(
            asset,
            expected_source_commit=SOURCE_COMMIT,
            expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
            require_behavioral_claim=True,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"host_path": "/root/run"}), "GGUF qualification"),
        (lambda value: value.update({"runs": 1}), "GGUF qualification"),
        (lambda value: value.update({"status": "failed"}), "GGUF qualification"),
        (lambda value: value.update({"image_digest": "latest"}), "image digest"),
    ],
)
def test_gguf_summary_rejects_process_metadata_and_false_success(
    tmp_path: Path, mutation: object, message: str
) -> None:
    path = _gguf_summary(tmp_path / "gguf.json")
    value = json.loads(path.read_bytes())
    assert callable(mutation)
    mutation(value)
    _write_json(path, value, newline=True)

    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match=message):
        evidence.build_asset(
            output=tmp_path / "asset.tar.gz",
            source_commit=SOURCE_COMMIT,
            source_archive_sha256=SOURCE_ARCHIVE_SHA256,
            qualification_summaries={"llama_cpp": path},
            behavioral_receipts=[],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ok", False, "TensorRT-LLM qualification"),
        ("gpu_count", 1, "TensorRT-LLM qualification"),
        ("runtime_provider_receipt_sha256", "bad", "runtime_provider"),
    ],
)
def test_tensorrt_summary_rejects_degraded_or_malformed_qualification(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    path = _tensorrt_summary(tmp_path / "tensorrt.json")
    summary = json.loads(path.read_bytes())
    summary[field] = value
    _write_json(path, summary, newline=True)

    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match=message):
        evidence.build_asset(
            output=tmp_path / "asset.tar.gz",
            source_commit=SOURCE_COMMIT,
            source_archive_sha256=SOURCE_ARCHIVE_SHA256,
            qualification_summaries={"tensorrt_llm": path},
            behavioral_receipts=[],
        )


def test_behavior_receipt_rejects_path_injection_and_false_arithmetic(
    tmp_path: Path,
) -> None:
    path = _behavior_receipt(tmp_path / "behavior.json")
    receipt = json.loads(path.read_bytes())
    receipt["policy_digest"] = "/root/private-policy.json"
    _write_json(path, receipt)
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="contract"):
        evidence.build_asset(
            output=tmp_path / "path-injection.tar.gz",
            source_commit=SOURCE_COMMIT,
            source_archive_sha256=SOURCE_ARCHIVE_SHA256,
            qualification_summaries={},
            behavioral_receipts=[path],
        )

    receipt["policy_digest"] = "sha256:" + "6" * 64
    receipt["regression"] = 0.0
    _write_json(path, receipt)
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="does not match"):
        evidence.build_asset(
            output=tmp_path / "false-arithmetic.tar.gz",
            source_commit=SOURCE_COMMIT,
            source_archive_sha256=SOURCE_ARCHIVE_SHA256,
            qualification_summaries={},
            behavioral_receipts=[path],
        )


def test_validator_rejects_tampering_unindexed_files_and_noncanonical_metadata(
    tmp_path: Path,
) -> None:
    asset = _build(tmp_path / "source")
    files = evidence._read_archive(asset.read_bytes())

    tampered = dict(files)
    receipt_path = "receipts/llama_cpp-qualification.json"
    receipt = json.loads(tampered[receipt_path])
    receipt["qualification_evidence_sha256"] = "f" * 64
    tampered[receipt_path] = _canonical(receipt)
    tampered_asset = tmp_path / "tampered.tar.gz"
    evidence._write_archive(tampered_asset, tampered)
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="digest"):
        evidence.validate_asset(
            tampered_asset,
            expected_source_commit=SOURCE_COMMIT,
            expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        )

    missing_summary = dict(files)
    missing_summary.pop("summaries/llama_cpp-qualification-summary.json")
    missing_summary_asset = tmp_path / "missing-summary.tar.gz"
    evidence._write_archive(missing_summary_asset, missing_summary)
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="summary digest"):
        evidence.validate_asset(
            missing_summary_asset,
            expected_source_commit=SOURCE_COMMIT,
            expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        )

    tampered_summary = dict(files)
    summary_path = "summaries/tensorrt_llm-qualification-summary.json"
    summary = json.loads(tampered_summary[summary_path])
    summary["output_sha256"] = "f" * 64
    tampered_summary[summary_path] = _canonical(summary)
    tampered_summary_asset = tmp_path / "tampered-summary.tar.gz"
    evidence._write_archive(tampered_summary_asset, tampered_summary)
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="summary digest"):
        evidence.validate_asset(
            tampered_summary_asset,
            expected_source_commit=SOURCE_COMMIT,
            expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        )

    coordinated = dict(files)
    summary_path = "summaries/llama_cpp-qualification-summary.json"
    receipt_path = "receipts/llama_cpp-qualification.json"
    summary = json.loads(coordinated[summary_path])
    summary["host_path"] = "/root/private-run"
    coordinated[summary_path] = _canonical(summary)
    receipt = json.loads(coordinated[receipt_path])
    receipt["qualification_summary_sha256"] = evidence._sha256(
        coordinated[summary_path]
    )
    coordinated[receipt_path] = _canonical(receipt)
    index = json.loads(coordinated["index.json"])
    entry = next(
        item for item in index["qualifications"] if item["provider_name"] == "llama_cpp"
    )
    entry["summary_sha256"] = evidence._sha256(coordinated[summary_path])
    entry["receipt_sha256"] = evidence._sha256(coordinated[receipt_path])
    coordinated["index.json"] = _canonical(index)
    coordinated_asset = tmp_path / "coordinated-private-field.tar.gz"
    evidence._write_archive(coordinated_asset, coordinated)
    with pytest.raises(
        evidence.RuntimeReleaseEvidenceError, match="GGUF qualification"
    ):
        evidence.validate_asset(
            coordinated_asset,
            expected_source_commit=SOURCE_COMMIT,
            expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        )

    unindexed = dict(files)
    unindexed["raw.log"] = b"private execution output"
    unindexed_asset = tmp_path / "unindexed.tar.gz"
    evidence._write_archive(unindexed_asset, unindexed)
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="unindexed"):
        evidence.validate_asset(
            unindexed_asset,
            expected_source_commit=SOURCE_COMMIT,
            expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        )

    noncanonical_asset = tmp_path / "noncanonical.tar.gz"
    with tarfile.open(noncanonical_asset, mode="w:gz") as archive:
        info = tarfile.TarInfo("index.json")
        info.size = len(files["index.json"])
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(files["index.json"]))
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="metadata"):
        evidence.validate_asset(
            noncanonical_asset,
            expected_source_commit=SOURCE_COMMIT,
            expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        )

    trailing_asset = tmp_path / "trailing-data.tar.gz"
    trailing_asset.write_bytes(asset.read_bytes() + b"unindexed trailing bytes")
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="encoding"):
        evidence.validate_asset(
            trailing_asset,
            expected_source_commit=SOURCE_COMMIT,
            expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        )


def test_validator_rejects_coordinated_required_binding_omission(
    tmp_path: Path,
) -> None:
    asset = _build(tmp_path / "source")
    files = evidence._read_archive(asset.read_bytes())
    summary_path = "summaries/tensorrt_llm-qualification-summary.json"
    receipt_path = "receipts/tensorrt_llm-qualification.json"

    summary = json.loads(files[summary_path])
    summary.pop("tokenizer_sha256")
    files[summary_path] = _canonical(summary)
    receipt = json.loads(files[receipt_path])
    receipt["qualification_summary_sha256"] = evidence._sha256(files[summary_path])
    files[receipt_path] = _canonical(receipt)
    index = json.loads(files["index.json"])
    entry = next(
        item
        for item in index["qualifications"]
        if item["provider_name"] == "tensorrt_llm"
    )
    entry["summary_sha256"] = evidence._sha256(files[summary_path])
    entry["receipt_sha256"] = evidence._sha256(files[receipt_path])
    files["index.json"] = _canonical(index)
    coordinated_asset = tmp_path / "coordinated-omission.tar.gz"
    evidence._write_archive(coordinated_asset, files)

    with pytest.raises(
        evidence.RuntimeReleaseEvidenceError, match="TensorRT-LLM qualification"
    ):
        evidence.validate_asset(
            coordinated_asset,
            expected_source_commit=SOURCE_COMMIT,
            expected_source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        )


def test_builder_refuses_unsafe_names_existing_output_and_symlink_input(
    tmp_path: Path,
) -> None:
    summary = _gguf_summary(tmp_path / "gguf.json")
    output = tmp_path / "asset.tar.gz"
    output.write_bytes(b"owned")
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="already exists"):
        evidence.build_asset(
            output=output,
            source_commit=SOURCE_COMMIT,
            source_archive_sha256=SOURCE_ARCHIVE_SHA256,
            qualification_summaries={"llama_cpp": summary},
            behavioral_receipts=[],
        )

    link = tmp_path / "summary-link.json"
    link.symlink_to(summary)
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="safely readable"):
        evidence.build_asset(
            output=tmp_path / "linked.tar.gz",
            source_commit=SOURCE_COMMIT,
            source_archive_sha256=SOURCE_ARCHIVE_SHA256,
            qualification_summaries={"llama_cpp": link},
            behavioral_receipts=[],
        )

    receipt = _behavior_receipt(tmp_path / "b.json")
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="duplicated"):
        evidence.build_asset(
            output=tmp_path / "duplicate-behavior.tar.gz",
            source_commit=SOURCE_COMMIT,
            source_archive_sha256=SOURCE_ARCHIVE_SHA256,
            qualification_summaries={},
            behavioral_receipts=[receipt, receipt],
        )
