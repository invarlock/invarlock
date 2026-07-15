#!/usr/bin/env python3
"""Build and verify a compact runtime release-evidence asset.

The asset intentionally contains only closed, canonical, digest-oriented
qualification and behavioral-claim receipts. Native engines, model files,
host paths, raw logs, and environment output are outside this format.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import re
import stat
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Final

from jsonschema import Draft202012Validator

from invarlock import public_contracts

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
INDEX_FORMAT: Final = public_contracts.RUNTIME_RELEASE_EVIDENCE_INDEX_FORMAT_VERSION
QUALIFICATION_RECEIPT_FORMAT: Final = (
    public_contracts.RUNTIME_QUALIFICATION_RELEASE_RECEIPT_FORMAT_VERSION
)
VALIDATION_FORMAT: Final = "invarlock/runtime-release-evidence-validation-v1"
BEHAVIOR_RECEIPT_FORMAT: Final = "invarlock/runtime-behavioral-claim-receipt-v1"
BEHAVIOR_CLAIM_SET: Final = "invarlock-runtime-behavioral-regression-v1"
QUALIFICATION_SCOPE: Final = "runtime_qualification_only"
BEHAVIOR_SCOPE: Final = "schedule_level_behavior"
GGUF_FORMAT: Final = "invarlock/gguf-runtime-blackbox-summary-v1"
TENSORRT_FORMAT: Final = "invarlock/tensorrt-llm-dual-gpu-qualification-v1"

_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_IMAGE_DIGEST = re.compile(r"^sha256:[a-f0-9]{64}$")
_GIT_COMMIT = re.compile(r"^[a-f0-9]{40}$")
_QUALIFICATION_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?$")
_MAX_SOURCE_BYTES: Final = 256 * 1024
_MAX_ASSET_BYTES: Final = 4 * 1024 * 1024
_MAX_ASSET_FILES: Final = 64
_MAX_TOTAL_PAYLOAD_BYTES: Final = 2 * 1024 * 1024
_ARCHIVE_MODE: Final = 0o444
_PRIVATE_FILE_MODE: Final = 0o400
_GGUF_KEYS: Final = frozenset(
    {
        "evidence_sha256",
        "fixture_revision",
        "format_version",
        "image_digest",
        "runs",
        "status",
    }
)
_TENSORRT_KEYS: Final = frozenset(
    {
        "candidate_image_digest",
        "engine_bundle_tree_sha256",
        "format_version",
        "gpu_count",
        "ok",
        "output_sha256",
        "runtime_provider_receipt_sha256",
        "tokenizer_sha256",
    }
)
_PROVIDER_PROFILE: Final = {
    "llama_cpp": (GGUF_FORMAT, "linux_cpu", 1),
    "tensorrt_llm": (
        TENSORRT_FORMAT,
        "linux_nvidia_gpu_compute_capability_9_0",
        2,
    ),
}


class RuntimeReleaseEvidenceError(RuntimeError):
    """Raised when a runtime release-evidence asset is unsafe or inconsistent."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeReleaseEvidenceError("JSON value is not canonicalizable") from exc


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeReleaseEvidenceError("JSON contains a duplicate object key")
        result[key] = value
    return result


def _parse_canonical_object(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        decoded = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeReleaseEvidenceError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise RuntimeReleaseEvidenceError(f"{label} must be a JSON object")
    if _canonical_json(decoded) != payload:
        raise RuntimeReleaseEvidenceError(f"{label} must use canonical JSON")
    return decoded


def _parse_producer_summary(payload: bytes, *, label: str) -> dict[str, object]:
    """Accept the producers' canonical JSON line without normalizing its hash."""

    canonical = payload[:-1] if payload.endswith(b"\n") else payload
    if not canonical or canonical.endswith(b"\n"):
        raise RuntimeReleaseEvidenceError(f"{label} has invalid JSON framing")
    return _parse_canonical_object(canonical, label=label)


def _read_regular_file(path: Path, *, label: str, limit: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeReleaseEvidenceError(f"{label} is not safely readable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise RuntimeReleaseEvidenceError(f"{label} must be a bounded regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(64 * 1024, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                raise RuntimeReleaseEvidenceError(f"{label} exceeds the byte limit")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RuntimeReleaseEvidenceError(f"{label} changed while it was read")
        return b"".join(chunks)
    except OSError as exc:
        raise RuntimeReleaseEvidenceError(f"{label} could not be read") from exc
    finally:
        os.close(descriptor)


def _load_schema(filename: str) -> dict[str, object]:
    payload = _read_regular_file(
        REPO_ROOT / "contracts" / filename,
        label=f"contract {filename}",
        limit=_MAX_SOURCE_BYTES,
    )
    return _parse_canonical_object(_canonical_json(json.loads(payload)), label=filename)


def _require_schema(value: Mapping[str, object], *, filename: str, label: str) -> None:
    errors = sorted(
        Draft202012Validator(_load_schema(filename)).iter_errors(dict(value)),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        path = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise RuntimeReleaseEvidenceError(
            f"{label} violates its contract at {path}: {error.message}"
        )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_source_bindings(source_commit: str, source_archive_sha256: str) -> None:
    if _GIT_COMMIT.fullmatch(source_commit) is None:
        raise RuntimeReleaseEvidenceError(
            "source commit must be a full lowercase commit"
        )
    if _SHA256.fullmatch(source_archive_sha256) is None:
        raise RuntimeReleaseEvidenceError(
            "source archive digest must be lowercase sha256"
        )


def _validate_gguf_summary(summary: Mapping[str, object]) -> tuple[str, str, int]:
    if (
        set(summary) != _GGUF_KEYS
        or summary.get("format_version") != GGUF_FORMAT
        or summary.get("runs") != 2
        or summary.get("status") != "ok"
    ):
        raise RuntimeReleaseEvidenceError("GGUF qualification summary is invalid")
    revision = summary.get("fixture_revision")
    image = summary.get("image_digest")
    evidence = summary.get("evidence_sha256")
    if not isinstance(revision, str) or _GIT_COMMIT.fullmatch(revision) is None:
        raise RuntimeReleaseEvidenceError("GGUF fixture revision is invalid")
    if not isinstance(image, str) or _IMAGE_DIGEST.fullmatch(image) is None:
        raise RuntimeReleaseEvidenceError("GGUF qualification image digest is invalid")
    if not isinstance(evidence, str) or _SHA256.fullmatch(evidence) is None:
        raise RuntimeReleaseEvidenceError(
            "GGUF qualification evidence digest is invalid"
        )
    return image, evidence, 1


def _validate_tensorrt_summary(
    summary: Mapping[str, object],
) -> tuple[str, str, int]:
    if (
        set(summary) != _TENSORRT_KEYS
        or summary.get("format_version") != TENSORRT_FORMAT
        or summary.get("gpu_count") != 2
        or summary.get("ok") is not True
    ):
        raise RuntimeReleaseEvidenceError(
            "TensorRT-LLM qualification summary is invalid"
        )
    image = summary.get("candidate_image_digest")
    if not isinstance(image, str) or _IMAGE_DIGEST.fullmatch(image) is None:
        raise RuntimeReleaseEvidenceError(
            "TensorRT-LLM qualification image digest is invalid"
        )
    for field_name in (
        "engine_bundle_tree_sha256",
        "output_sha256",
        "runtime_provider_receipt_sha256",
        "tokenizer_sha256",
    ):
        value = summary.get(field_name)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise RuntimeReleaseEvidenceError(
                f"TensorRT-LLM qualification {field_name} is invalid"
            )
    evidence = summary["runtime_provider_receipt_sha256"]
    assert isinstance(evidence, str)
    return image, evidence, 2


def _qualification_receipt(
    *,
    provider_name: str,
    qualification_name: str | None,
    summary_payload: bytes,
    source_commit: str,
    source_archive_sha256: str,
) -> dict[str, object]:
    summary = _parse_producer_summary(
        summary_payload, label=f"{provider_name} qualification summary"
    )
    if provider_name == "llama_cpp":
        image, evidence, device_count = _validate_gguf_summary(summary)
    elif provider_name == "tensorrt_llm":
        image, evidence, device_count = _validate_tensorrt_summary(summary)
    else:
        raise RuntimeReleaseEvidenceError(
            f"unsupported qualification provider: {provider_name}"
        )
    qualification_format, platform_profile, expected_count = _PROVIDER_PROFILE[
        provider_name
    ]
    if device_count != expected_count:
        raise RuntimeReleaseEvidenceError("qualification device count is inconsistent")
    receipt: dict[str, object] = {
        "format_version": QUALIFICATION_RECEIPT_FORMAT,
        "claim_scope": QUALIFICATION_SCOPE,
        "provider_name": provider_name,
        "platform_profile": platform_profile,
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "runtime_image_digest": image,
        "qualification_format": qualification_format,
        "qualification_summary_sha256": _sha256(summary_payload),
        "qualification_evidence_sha256": evidence,
        "qualified_device_count": device_count,
    }
    if qualification_name is not None:
        receipt["qualification_name"] = qualification_name
    _validate_qualification_receipt(receipt)
    return receipt


def _validate_qualification_receipt(receipt: Mapping[str, object]) -> None:
    _require_schema(
        receipt,
        filename="runtime_qualification_release_receipt.schema.json",
        label="qualification receipt",
    )
    provider = receipt.get("provider_name")
    if not isinstance(provider, str) or provider not in _PROVIDER_PROFILE:
        raise RuntimeReleaseEvidenceError("qualification provider is unsupported")
    qualification_name = receipt.get("qualification_name")
    if qualification_name is not None and (
        not isinstance(qualification_name, str)
        or _QUALIFICATION_NAME.fullmatch(qualification_name) is None
    ):
        raise RuntimeReleaseEvidenceError("qualification name is invalid")
    expected_format, expected_profile, expected_count = _PROVIDER_PROFILE[provider]
    if (
        receipt.get("qualification_format") != expected_format
        or receipt.get("platform_profile") != expected_profile
        or receipt.get("qualified_device_count") != expected_count
    ):
        raise RuntimeReleaseEvidenceError(
            "qualification receipt provider profile is inconsistent"
        )


def _validate_behavior_receipt(receipt: Mapping[str, object]) -> None:
    _require_schema(
        receipt,
        filename="runtime_behavioral_claim_receipt.schema.json",
        label="behavioral claim receipt",
    )
    if (
        receipt.get("format_version") != BEHAVIOR_RECEIPT_FORMAT
        or receipt.get("claim_set") != BEHAVIOR_CLAIM_SET
        or receipt.get("metric") != "exact_match"
        or receipt.get("verdict") != "pass"
    ):
        raise RuntimeReleaseEvidenceError("behavioral claim receipt scope is invalid")
    baseline = receipt.get("baseline_score")
    subject = receipt.get("subject_score")
    regression = receipt.get("regression")
    if any(
        isinstance(value, bool) or not isinstance(value, int | float)
        for value in (
            baseline,
            subject,
            regression,
        )
    ):
        raise RuntimeReleaseEvidenceError("behavioral claim scores must be numeric")
    assert isinstance(baseline, int | float)
    assert isinstance(subject, int | float)
    assert isinstance(regression, int | float)
    if not all(
        math.isfinite(float(value)) for value in (baseline, subject, regression)
    ):
        raise RuntimeReleaseEvidenceError("behavioral claim scores must be finite")
    if not math.isclose(
        float(regression),
        float(baseline) - float(subject),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeReleaseEvidenceError(
            "behavioral claim regression does not match its scores"
        )


def _safe_member_name(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not name
        or name.startswith("/")
        or "\\" in name
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != name
    ):
        raise RuntimeReleaseEvidenceError("asset contains an unsafe member path")
    return name


def _archive_bytes(files: Mapping[str, bytes]) -> bytes:
    raw_output = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=raw_output, mtime=0
    ) as compressed:
        with tarfile.open(
            fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT
        ) as archive:
            for name in sorted(files):
                payload = files[name]
                info = tarfile.TarInfo(_safe_member_name(name))
                info.size = len(payload)
                info.mode = _ARCHIVE_MODE
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                archive.addfile(info, io.BytesIO(payload))
    return raw_output.getvalue()


def _write_archive(path: Path, files: Mapping[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeReleaseEvidenceError("output asset already exists")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(_archive_bytes(files))
        os.chmod(temporary, _PRIVATE_FILE_MODE)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RuntimeReleaseEvidenceError("output asset already exists") from exc
        except OSError as exc:
            raise RuntimeReleaseEvidenceError(
                "output asset could not be published"
            ) from exc
    except Exception:
        raise
    finally:
        temporary.unlink(missing_ok=True)


def _read_archive(payload: bytes) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    total = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) > _MAX_ASSET_FILES:
                raise RuntimeReleaseEvidenceError("asset contains too many members")
            for member in members:
                name = _safe_member_name(member.name)
                if name in files:
                    raise RuntimeReleaseEvidenceError(
                        "asset contains duplicate members"
                    )
                if (
                    not member.isfile()
                    or member.mode != _ARCHIVE_MODE
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname
                    or member.gname
                    or member.mtime != 0
                    or member.size > _MAX_SOURCE_BYTES
                ):
                    raise RuntimeReleaseEvidenceError(
                        "asset member metadata is not canonical"
                    )
                total += member.size
                if total > _MAX_TOTAL_PAYLOAD_BYTES:
                    raise RuntimeReleaseEvidenceError("asset payload exceeds the limit")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RuntimeReleaseEvidenceError("asset member cannot be read")
                member_payload = extracted.read(_MAX_SOURCE_BYTES + 1)
                if len(member_payload) != member.size:
                    raise RuntimeReleaseEvidenceError(
                        "asset member size is inconsistent"
                    )
                files[name] = member_payload
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeReleaseEvidenceError(
            "asset is not a valid gzip tar archive"
        ) from exc
    if payload != _archive_bytes(files):
        raise RuntimeReleaseEvidenceError("asset archive encoding is not canonical")
    return files


def build_asset(
    *,
    output: Path,
    source_commit: str,
    source_archive_sha256: str,
    qualification_summaries: Mapping[str, Path],
    behavioral_receipts: Sequence[Path],
) -> dict[str, object]:
    """Build, close, and independently reload one compact release asset."""

    _require_source_bindings(source_commit, source_archive_sha256)
    if not qualification_summaries and not behavioral_receipts:
        raise RuntimeReleaseEvidenceError("asset must contain at least one receipt")
    files: dict[str, bytes] = {}
    qualification_entries: list[dict[str, str]] = []
    parsed_qualifications: list[tuple[str, str | None, Path]] = []
    qualification_names_by_provider: dict[str, set[str | None]] = {}
    for key, summary_input_path in qualification_summaries.items():
        provider, qualification_name = _parse_qualification_key(key)
        parsed_qualifications.append((provider, qualification_name, summary_input_path))
        qualification_names_by_provider.setdefault(provider, set()).add(
            qualification_name
        )
    for provider, names in qualification_names_by_provider.items():
        if len(names) > 1 and None in names:
            raise RuntimeReleaseEvidenceError(
                f"repeated {provider} qualifications must all be named"
            )
    for provider, qualification_name, producer_path in sorted(
        parsed_qualifications, key=lambda value: (value[0], value[1] or "")
    ):
        qualification_key = _qualification_key(provider, qualification_name)
        producer_summary_payload = _read_regular_file(
            producer_path,
            label=f"{qualification_key} qualification summary",
            limit=_MAX_SOURCE_BYTES,
        )
        summary = _parse_producer_summary(
            producer_summary_payload,
            label=f"{qualification_key} qualification summary",
        )
        summary_payload = _canonical_json(summary)
        summary_sha256 = _sha256(summary_payload)
        receipt_payload = _canonical_json(
            _qualification_receipt(
                provider_name=provider,
                qualification_name=qualification_name,
                summary_payload=summary_payload,
                source_commit=source_commit,
                source_archive_sha256=source_archive_sha256,
            )
        )
        receipt_path, summary_path = _qualification_paths(provider, qualification_name)
        files[receipt_path] = receipt_payload
        files[summary_path] = summary_payload
        entry: dict[str, str] = {
            "provider_name": provider,
            "claim_scope": QUALIFICATION_SCOPE,
            "receipt_path": receipt_path,
            "receipt_sha256": _sha256(receipt_payload),
            "summary_path": summary_path,
            "summary_sha256": summary_sha256,
        }
        if qualification_name is not None:
            entry["qualification_name"] = qualification_name
        qualification_entries.append(entry)
    behavior_entries: list[dict[str, str]] = []
    seen_behavior_receipts: set[str] = set()
    for receipt_input in behavioral_receipts:
        receipt_payload = _read_regular_file(
            receipt_input,
            label="behavioral claim receipt",
            limit=_MAX_SOURCE_BYTES,
        )
        receipt = _parse_canonical_object(receipt_payload, label="behavioral receipt")
        _validate_behavior_receipt(receipt)
        receipt_sha256 = _sha256(receipt_payload)
        if receipt_sha256 in seen_behavior_receipts:
            raise RuntimeReleaseEvidenceError("behavioral receipt is duplicated")
        seen_behavior_receipts.add(receipt_sha256)
        claim_id = f"claim-{receipt_sha256}"
        receipt_path = f"receipts/{claim_id}-behavior.json"
        files[receipt_path] = receipt_payload
        behavior_entries.append(
            {
                "claim_id": claim_id,
                "claim_scope": BEHAVIOR_SCOPE,
                "receipt_path": receipt_path,
                "receipt_sha256": receipt_sha256,
            }
        )
    behavior_entries.sort(key=lambda entry: entry["claim_id"])
    index: dict[str, object] = {
        "format_version": INDEX_FORMAT,
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "qualifications": qualification_entries,
        "behavioral_claims": behavior_entries,
    }
    _require_schema(
        index,
        filename="runtime_release_evidence_index.schema.json",
        label="runtime release evidence index",
    )
    files["index.json"] = _canonical_json(index)
    _write_archive(output, files)
    return validate_asset(
        output,
        expected_source_commit=source_commit,
        expected_source_archive_sha256=source_archive_sha256,
        expected_providers=frozenset(qualification_names_by_provider),
        expected_qualifications=frozenset(
            _qualification_key(provider, name)
            for provider, name, _path in parsed_qualifications
        ),
        require_behavioral_claim=bool(behavioral_receipts),
    )


def validate_asset(
    asset: Path,
    *,
    expected_source_commit: str,
    expected_source_archive_sha256: str,
    expected_providers: frozenset[str] | None = None,
    expected_qualifications: frozenset[str] | None = None,
    require_behavioral_claim: bool = False,
    expected_asset_sha256: str | None = None,
) -> dict[str, object]:
    """Validate archive safety, closed schemas, and every indexed digest."""

    _require_source_bindings(expected_source_commit, expected_source_archive_sha256)
    asset_payload = _read_regular_file(
        asset, label="runtime release evidence asset", limit=_MAX_ASSET_BYTES
    )
    asset_sha256 = _sha256(asset_payload)
    if expected_asset_sha256 is not None:
        if _SHA256.fullmatch(expected_asset_sha256) is None:
            raise RuntimeReleaseEvidenceError("expected asset digest is invalid")
        if asset_sha256 != expected_asset_sha256:
            raise RuntimeReleaseEvidenceError("asset digest does not match")
    files = _read_archive(asset_payload)
    index_payload = files.get("index.json")
    if index_payload is None:
        raise RuntimeReleaseEvidenceError("asset index is missing")
    index = _parse_canonical_object(index_payload, label="asset index")
    _require_schema(
        index,
        filename="runtime_release_evidence_index.schema.json",
        label="runtime release evidence index",
    )
    if (
        index.get("format_version") != INDEX_FORMAT
        or index.get("source_commit") != expected_source_commit
        or index.get("source_archive_sha256") != expected_source_archive_sha256
    ):
        raise RuntimeReleaseEvidenceError("asset source binding does not match")
    qualifications = index.get("qualifications")
    behaviors = index.get("behavioral_claims")
    assert isinstance(qualifications, list)
    assert isinstance(behaviors, list)
    if qualifications != sorted(
        qualifications,
        key=lambda value: (
            str(value["provider_name"]),
            str(value.get("qualification_name", "")),
        ),
    ) or behaviors != sorted(behaviors, key=lambda value: str(value["claim_id"])):
        raise RuntimeReleaseEvidenceError("asset index entries are not canonical")
    indexed_files = {"index.json"}
    observed_providers: set[str] = set()
    observed_qualifications: set[str] = set()
    qualification_names_by_provider: dict[str, set[str | None]] = {}
    for entry in qualifications:
        assert isinstance(entry, dict)
        provider = entry["provider_name"]
        qualification_name = entry.get("qualification_name")
        receipt_path = entry["receipt_path"]
        summary_path = entry["summary_path"]
        assert isinstance(provider, str)
        assert qualification_name is None or isinstance(qualification_name, str)
        assert isinstance(receipt_path, str)
        assert isinstance(summary_path, str)
        expected_receipt_path, expected_summary_path = _qualification_paths(
            provider, qualification_name
        )
        if (
            receipt_path != expected_receipt_path
            or summary_path != expected_summary_path
        ):
            raise RuntimeReleaseEvidenceError(
                "qualification index paths do not match the provider and name"
            )
        qualification_key = _qualification_key(provider, qualification_name)
        if qualification_key in observed_qualifications:
            raise RuntimeReleaseEvidenceError("asset repeats a qualification name")
        observed_qualifications.add(qualification_key)
        observed_providers.add(provider)
        qualification_names_by_provider.setdefault(provider, set()).add(
            qualification_name
        )
        receipt_payload = files.get(receipt_path)
        if (
            receipt_payload is None
            or _sha256(receipt_payload) != entry["receipt_sha256"]
        ):
            raise RuntimeReleaseEvidenceError(
                "qualification receipt digest does not match"
            )
        receipt = _parse_canonical_object(
            receipt_payload, label=f"{provider} qualification receipt"
        )
        _validate_qualification_receipt(receipt)
        summary_payload = files.get(summary_path)
        if (
            summary_payload is None
            or _sha256(summary_payload) != entry["summary_sha256"]
        ):
            raise RuntimeReleaseEvidenceError(
                "qualification summary digest does not match"
            )
        summary_sha256 = entry["summary_sha256"]
        assert isinstance(summary_sha256, str)
        expected_receipt = _qualification_receipt(
            provider_name=provider,
            qualification_name=qualification_name,
            summary_payload=summary_payload,
            source_commit=expected_source_commit,
            source_archive_sha256=expected_source_archive_sha256,
        )
        if (
            receipt != expected_receipt
            or receipt.get("qualification_summary_sha256") != entry["summary_sha256"]
        ):
            raise RuntimeReleaseEvidenceError(
                "qualification receipt binding does not match the index"
            )
        indexed_files.add(receipt_path)
        indexed_files.add(summary_path)
    for provider, names in qualification_names_by_provider.items():
        if len(names) > 1 and None in names:
            raise RuntimeReleaseEvidenceError(
                f"repeated {provider} qualifications must all be named"
            )
    if expected_providers is not None and observed_providers != set(expected_providers):
        raise RuntimeReleaseEvidenceError("qualified provider set does not match")
    if expected_qualifications is not None:
        canonical_expected = {
            _qualification_key(*_parse_qualification_key(value))
            for value in expected_qualifications
        }
        if observed_qualifications != canonical_expected:
            raise RuntimeReleaseEvidenceError(
                "qualified qualification set does not match"
            )
    observed_claim_ids: set[str] = set()
    for entry in behaviors:
        assert isinstance(entry, dict)
        claim_id = entry["claim_id"]
        receipt_path = entry["receipt_path"]
        assert isinstance(claim_id, str)
        assert isinstance(receipt_path, str)
        if claim_id in observed_claim_ids:
            raise RuntimeReleaseEvidenceError("asset repeats a behavior claim ID")
        observed_claim_ids.add(claim_id)
        receipt_payload = files.get(receipt_path)
        if (
            receipt_payload is None
            or _sha256(receipt_payload) != entry["receipt_sha256"]
        ):
            raise RuntimeReleaseEvidenceError(
                "behavioral receipt digest does not match"
            )
        receipt = _parse_canonical_object(
            receipt_payload, label=f"{claim_id} behavioral receipt"
        )
        _validate_behavior_receipt(receipt)
        indexed_files.add(receipt_path)
    if require_behavioral_claim and not behaviors:
        raise RuntimeReleaseEvidenceError("asset has no schedule-level behavior claim")
    if set(files) != indexed_files:
        raise RuntimeReleaseEvidenceError("asset contains an unindexed file")
    return {
        "format_version": VALIDATION_FORMAT,
        "status": "ok",
        "asset_sha256": asset_sha256,
        "source_commit": expected_source_commit,
        "source_archive_sha256": expected_source_archive_sha256,
        "qualification_count": len(qualifications),
        "qualified_provider_count": len(observed_providers),
        "behavioral_claim_count": len(behaviors),
    }


def _qualification_key(provider: str, qualification_name: str | None) -> str:
    return (
        provider if qualification_name is None else f"{provider}:{qualification_name}"
    )


def _parse_qualification_key(value: str) -> tuple[str, str | None]:
    provider, separator, qualification_name = value.partition(":")
    if provider not in _PROVIDER_PROFILE:
        raise RuntimeReleaseEvidenceError(
            f"unsupported qualification provider: {provider}"
        )
    if not separator:
        return provider, None
    if (
        not qualification_name
        or ":" in qualification_name
        or _QUALIFICATION_NAME.fullmatch(qualification_name) is None
    ):
        raise RuntimeReleaseEvidenceError(
            "qualification name must be a lowercase path-free slug"
        )
    return provider, qualification_name


def _qualification_paths(
    provider: str, qualification_name: str | None
) -> tuple[str, str]:
    key = provider if qualification_name is None else f"{provider}-{qualification_name}"
    return (
        f"receipts/{key}-qualification.json",
        f"summaries/{key}-qualification-summary.json",
    )


def _parse_qualification_paths(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise RuntimeReleaseEvidenceError(
                "qualification must use PROVIDER[:NAME]=SUMMARY"
            )
        provider, qualification_name = _parse_qualification_key(name)
        name = _qualification_key(provider, qualification_name)
        if name in result:
            raise RuntimeReleaseEvidenceError(f"duplicate qualification name: {name}")
        result[name] = Path(raw_path)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Build and reload a closed asset")
    build.add_argument("--source-commit", required=True)
    build.add_argument("--source-archive-sha256", required=True)
    build.add_argument(
        "--qualification",
        action="append",
        default=[],
        metavar="PROVIDER[:NAME]=SUMMARY",
    )
    build.add_argument(
        "--behavior", action="append", default=[], type=Path, metavar="RECEIPT"
    )
    build.add_argument("--output", required=True, type=Path)

    validate = commands.add_parser("validate", help="Validate an existing asset")
    validate.add_argument("--asset", required=True, type=Path)
    validate.add_argument("--expected-source-commit", required=True)
    validate.add_argument("--expected-source-archive-sha256", required=True)
    validate.add_argument("--expected-asset-sha256")
    validate.add_argument(
        "--expected-provider",
        action="append",
        choices=sorted(_PROVIDER_PROFILE),
        default=None,
    )
    validate.add_argument(
        "--expected-qualification",
        action="append",
        default=None,
        metavar="PROVIDER[:NAME]",
    )
    validate.add_argument("--require-behavioral-claim", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    result: dict[str, object] | None = None
    try:
        if args.command == "build":
            qualification_paths = _parse_qualification_paths(args.qualification)
            result = build_asset(
                output=args.output,
                source_commit=args.source_commit,
                source_archive_sha256=args.source_archive_sha256,
                qualification_summaries=qualification_paths,
                behavioral_receipts=args.behavior,
            )
        else:
            providers = (
                frozenset(args.expected_provider)
                if args.expected_provider is not None
                else None
            )
            expected_qualifications = (
                frozenset(
                    _qualification_key(*_parse_qualification_key(value))
                    for value in args.expected_qualification
                )
                if args.expected_qualification is not None
                else None
            )
            result = validate_asset(
                args.asset,
                expected_source_commit=args.expected_source_commit,
                expected_source_archive_sha256=(args.expected_source_archive_sha256),
                expected_providers=providers,
                expected_qualifications=expected_qualifications,
                require_behavioral_claim=args.require_behavioral_claim,
                expected_asset_sha256=args.expected_asset_sha256,
            )
    except RuntimeReleaseEvidenceError as exc:
        parser.error(str(exc))
    if result is None:
        parser.error("command completed without a validation result")
    print(_canonical_json(result).decode("utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI tests
    raise SystemExit(main())
