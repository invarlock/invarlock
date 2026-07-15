from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.release import runtime_release_evidence as evidence
from tests.scripts._runtime_release_evidence_test_support import (
    IMAGE_DIGEST,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_COMMIT,
    behavior_receipt,
    canonical,
    gguf_summary,
    tensorrt_summary,
)


@pytest.mark.parametrize(
    "payload, message",
    [
        (b"\xff", "UTF-8 JSON"),
        (b'{"a":}', "UTF-8 JSON"),
        (b"[1]", "JSON object"),
        (b'{"a": 1}', "canonical JSON"),
        (b'{"a":1,"a":2}', "duplicate object key"),
    ],
)
def test_canonical_object_parser_rejects_ambiguous_or_noncanonical_json(
    payload: bytes, message: str
) -> None:
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match=message):
        evidence._parse_canonical_object(payload, label="input")


def test_canonical_encoder_rejects_values_that_cannot_be_signed() -> None:
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="canonicalizable"):
        evidence._canonical_json({"unsupported": object()})
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="canonicalizable"):
        evidence._canonical_json({"nonfinite": float("nan")})


@pytest.mark.parametrize("payload", [b"", b"{}\n\n"])
def test_producer_summary_rejects_empty_or_multiple_json_lines(payload: bytes) -> None:
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="framing"):
        evidence._parse_producer_summary(payload, label="summary")


def test_bounded_reader_rejects_missing_symlink_directory_and_oversize(
    tmp_path: Path,
) -> None:
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="safely readable"):
        evidence._read_regular_file(tmp_path / "missing", label="input", limit=2)

    regular = tmp_path / "regular"
    regular.write_bytes(b"abc")
    link = tmp_path / "link"
    link.symlink_to(regular)
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="safely readable"):
        evidence._read_regular_file(link, label="input", limit=4)
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="bounded regular"):
        evidence._read_regular_file(tmp_path, label="input", limit=4)
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="bounded regular"):
        evidence._read_regular_file(regular, label="input", limit=2)


def test_bounded_reader_detects_growth_mutation_and_read_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"x")
    real_read = evidence.os.read

    monkeypatch.setattr(evidence.os, "read", lambda _fd, _size: b"xx")
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="byte limit"):
        evidence._read_regular_file(source, label="input", limit=1)

    def fail_read(_fd: int, _size: int) -> bytes:
        raise OSError("read failed")

    monkeypatch.setattr(evidence.os, "read", fail_read)
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="could not be read"):
        evidence._read_regular_file(source, label="input", limit=2)

    monkeypatch.setattr(evidence.os, "read", real_read)
    real_fstat = evidence.os.fstat
    calls = 0

    def changed_fstat(fd: int):
        nonlocal calls
        calls += 1
        value = real_fstat(fd)
        if calls == 1:
            return value
        return SimpleNamespace(
            st_mode=value.st_mode,
            st_size=value.st_size,
            st_dev=value.st_dev,
            st_ino=value.st_ino,
            st_mtime_ns=value.st_mtime_ns + 1,
            st_ctime_ns=value.st_ctime_ns,
        )

    monkeypatch.setattr(evidence.os, "fstat", changed_fstat)
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="changed"):
        evidence._read_regular_file(source, label="input", limit=2)


@pytest.mark.parametrize(
    "source_commit, archive_digest, message",
    [
        ("a" * 39, SOURCE_ARCHIVE_SHA256, "full lowercase commit"),
        (SOURCE_COMMIT, "B" * 64, "lowercase sha256"),
    ],
)
def test_source_bindings_must_be_full_canonical_digests(
    source_commit: str, archive_digest: str, message: str
) -> None:
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match=message):
        evidence._require_source_bindings(source_commit, archive_digest)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("fixture_revision", "short", "fixture revision"),
        ("image_digest", "c" * 64, "image digest"),
        ("evidence_sha256", "D" * 64, "evidence digest"),
    ],
)
def test_gguf_summary_rejects_malformed_identity_bindings(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    summary = json.loads(gguf_summary(tmp_path / "gguf.json").read_bytes())
    summary[field] = value
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match=message):
        evidence._validate_gguf_summary(summary)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("candidate_image_digest", "c" * 64, "image digest"),
        ("tokenizer_sha256", "Z" * 64, "tokenizer_sha256"),
    ],
)
def test_tensorrt_summary_rejects_malformed_artifact_bindings(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    summary = json.loads(tensorrt_summary(tmp_path / "trt.json").read_bytes())
    summary[field] = value
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match=message):
        evidence._validate_tensorrt_summary(summary)


def test_qualification_receipt_rejects_unknown_provider_and_device_count_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = gguf_summary(tmp_path / "gguf.json").read_bytes()
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="unsupported"):
        evidence._qualification_receipt(
            provider_name="unknown",
            qualification_name=None,
            summary_payload=payload,
            source_commit=SOURCE_COMMIT,
            source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        )

    monkeypatch.setattr(
        evidence,
        "_validate_gguf_summary",
        lambda _summary: (IMAGE_DIGEST, "d" * 64, 2),
    )
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="device count"):
        evidence._qualification_receipt(
            provider_name="llama_cpp",
            qualification_name=None,
            summary_payload=payload,
            source_commit=SOURCE_COMMIT,
            source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        )


def test_receipt_validators_defend_provider_profile_and_numeric_invariants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification = evidence._qualification_receipt(
        provider_name="llama_cpp",
        qualification_name="cpu",
        summary_payload=canonical(
            json.loads(gguf_summary(tmp_path / "gguf.json").read_bytes())
        ),
        source_commit=SOURCE_COMMIT,
        source_archive_sha256=SOURCE_ARCHIVE_SHA256,
    )
    behavior = json.loads(behavior_receipt(tmp_path / "behavior.json").read_bytes())
    monkeypatch.setattr(evidence, "_require_schema", lambda *_args, **_kwargs: None)

    for replacement, message in (
        ({**qualification, "provider_name": "unknown"}, "unsupported"),
        ({**qualification, "qualification_name": "../cpu"}, "name is invalid"),
        ({**qualification, "qualified_device_count": 2}, "profile is inconsistent"),
    ):
        with pytest.raises(evidence.RuntimeReleaseEvidenceError, match=message):
            evidence._validate_qualification_receipt(replacement)

    for replacement, message in (
        ({**behavior, "verdict": "fail"}, "scope is invalid"),
        ({**behavior, "baseline_score": True}, "scores must be numeric"),
        ({**behavior, "baseline_score": float("inf")}, "scores must be finite"),
        ({**behavior, "regression": 0.1}, "does not match"),
    ):
        with pytest.raises(evidence.RuntimeReleaseEvidenceError, match=message):
            evidence._validate_behavior_receipt(replacement)


@pytest.mark.parametrize("value", ["onnx", "unknown:name"])
def test_qualification_key_rejects_unknown_providers(value: str) -> None:
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="unsupported"):
        evidence._parse_qualification_key(value)


@pytest.mark.parametrize("value", ["llama_cpp", "=path", "llama_cpp="])
def test_qualification_path_requires_an_explicit_name_and_path(value: str) -> None:
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="PROVIDER"):
        evidence._parse_qualification_paths([value])


def test_qualification_path_rejects_duplicate_canonical_name() -> None:
    with pytest.raises(evidence.RuntimeReleaseEvidenceError, match="duplicate"):
        evidence._parse_qualification_paths(
            ["llama_cpp:cpu=first.json", "llama_cpp:cpu=second.json"]
        )


def test_cli_converts_closed_validation_errors_into_usage_errors(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as raised:
        evidence.main(
            [
                "build",
                "--source-commit",
                SOURCE_COMMIT,
                "--source-archive-sha256",
                SOURCE_ARCHIVE_SHA256,
                "--qualification",
                "unknown=summary.json",
                "--output",
                str(tmp_path / "asset.tar.gz"),
            ]
        )
    assert raised.value.code == 2


def test_cli_rejects_a_missing_validation_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(evidence, "build_asset", lambda **_kwargs: None)

    with pytest.raises(SystemExit) as raised:
        evidence.main(
            [
                "build",
                "--source-commit",
                SOURCE_COMMIT,
                "--source-archive-sha256",
                SOURCE_ARCHIVE_SHA256,
                "--behavior",
                str(tmp_path / "receipt.json"),
                "--output",
                str(tmp_path / "asset.tar.gz"),
            ]
        )

    assert raised.value.code == 2
    assert "without a validation result" in capsys.readouterr().err
