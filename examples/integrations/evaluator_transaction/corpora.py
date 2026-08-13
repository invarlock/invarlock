"""Closed corpus profiles shared by maintained evaluator transactions."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

_ROOT = Path(__file__).resolve().parent
_REPOSITORY = _ROOT.parents[2]
_QUICK_RECORDS = _ROOT.parent / "lm-evaluation-harness" / "records.json"
_FLAGSHIP_RECORDS = _ROOT / "mmlu_pro_qwen_instruct_400.jsonl"
_FLAGSHIP_MANIFEST = _ROOT / "flagship_corpus.json"
_QUALIFICATION_MANIFEST = (
    _REPOSITORY / "docs/reference/qualification-suites.manifest.json"
)


@dataclass(frozen=True, slots=True)
class CorpusProfile:
    key: str
    profile_id: str
    dataset_name: str
    split: str
    record_count: int
    dataset_sha256: str
    context_length: int
    minimum_side_accuracy: float
    maximum_interval_width_pp: float
    delta_min_pp: float

    def dataset_descriptor(self, path: str = "inputs/records.jsonl") -> dict[str, Any]:
        return {
            "path": path,
            "sha256": self.dataset_sha256,
            "format": "jsonl",
            "name": self.dataset_name,
            "split": self.split,
            "input_field": "prompt",
            "expected_output_field": "expected",
            "id_field": "id",
        }

    def acceptance_policy(self) -> dict[str, Any]:
        return {
            "resolved_policy": {
                "metrics": {
                    "exact_match": {
                        "delta_min_pp": self.delta_min_pp,
                        "maximum_interval_width_pp": self.maximum_interval_width_pp,
                        "minimum_record_count": self.record_count,
                        "minimum_side_accuracy": self.minimum_side_accuracy,
                    }
                }
            }
        }


def _json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unavailable or invalid") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value


def _manifest() -> dict[str, Any]:
    return _json(_FLAGSHIP_MANIFEST, label="flagship corpus manifest")


def _qualification_manifest() -> dict[str, Any]:
    manifest = _manifest()
    payload = _QUALIFICATION_MANIFEST.read_bytes()
    expected = manifest["qualification_suite"]["manifest_sha256"]
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(
            "qualification suite manifest does not match its pinned digest"
        )
    return _json(_QUALIFICATION_MANIFEST, label="qualification suite manifest")


def records_jsonl(records: Iterable[dict[str, str]], *, compact: bool = False) -> bytes:
    options: dict[str, Any] = {"sort_keys": True}
    if compact:
        options.update(separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return b"".join(
        (json.dumps(record, **options) + "\n").encode("utf-8") for record in records
    )


def _quick_profile() -> CorpusProfile:
    return CorpusProfile(
        key="quick",
        profile_id="causal-cloze-102-v1",
        dataset_name="qwen3-0.6b-base-to-post-trained",
        split="validation",
        record_count=102,
        dataset_sha256="d80e81ba17fb93b9b8a46f9817f9841f5f9c2858c9d703b3ce28847b2eaeb57c",
        context_length=64,
        minimum_side_accuracy=0.20,
        maximum_interval_width_pp=20.0,
        delta_min_pp=-20.0,
    )


def _flagship_profile() -> CorpusProfile:
    manifest = _manifest()
    dataset = manifest["derived_dataset"]
    policy = manifest["acceptance_policy"]
    return CorpusProfile(
        key="flagship",
        profile_id=manifest["profile_id"],
        dataset_name=dataset["name"],
        split=dataset["split"],
        record_count=dataset["record_count"],
        dataset_sha256=dataset["sha256"],
        context_length=dataset["maximum_input_tokens"],
        minimum_side_accuracy=policy["minimum_side_accuracy"],
        maximum_interval_width_pp=policy["maximum_interval_width_pp"],
        delta_min_pp=policy["delta_min_pp"],
    )


def corpus_profile(key: str) -> CorpusProfile:
    profiles = {"quick": _quick_profile(), "flagship": _flagship_profile()}
    try:
        return profiles[key]
    except KeyError as exc:
        raise ValueError(f"unknown corpus profile: {key}") from exc


def _canonical_payload(values: list[dict[str, str]], profile: CorpusProfile) -> bytes:
    return records_jsonl(values, compact=profile.key == "flagship")


def validate_dataset_records(payload: bytes, profile: CorpusProfile) -> None:
    try:
        values = [json.loads(line) for line in payload.splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("evaluator corpus is not valid JSONL") from exc
    if len(values) != profile.record_count or any(
        not isinstance(value, dict)
        or set(value) != {"expected", "id", "prompt"}
        or any(not isinstance(value[key], str) or not value[key] for key in value)
        for value in values
    ):
        raise ValueError(
            f"evaluator corpus must contain {profile.record_count} complete records"
        )
    records = cast(list[dict[str, str]], values)
    if len({value["id"] for value in records}) != len(records):
        raise ValueError("evaluator corpus IDs are not unique")
    if _canonical_payload(records, profile) != payload:
        raise ValueError("evaluator corpus JSONL is not canonical")


def profile_for_dataset(payload: bytes) -> CorpusProfile:
    observed = hashlib.sha256(payload).hexdigest()
    for key in ("quick", "flagship"):
        profile = corpus_profile(key)
        if observed == profile.dataset_sha256:
            validate_dataset_records(payload, profile)
            return profile
    raise ValueError("dataset is not a pinned evaluator corpus")


def profile_for_descriptor(value: object) -> CorpusProfile:
    for key in ("quick", "flagship"):
        profile = corpus_profile(key)
        if value == profile.dataset_descriptor():
            return profile
    raise ValueError("dataset descriptor is not a pinned evaluator corpus")


def corpus_provenance(profile: CorpusProfile) -> dict[str, Any]:
    value: dict[str, Any] = {
        "profile_id": profile.profile_id,
        "dataset_name": profile.dataset_name,
        "dataset_sha256": profile.dataset_sha256,
        "record_count": profile.record_count,
    }
    if profile.key == "flagship":
        manifest = _manifest()
        qualification = _qualification_manifest()
        artifact_name = manifest["qualification_suite"]["artifact"]
        artifact = qualification["artifacts"][artifact_name]
        records = flagship_records()
        if (
            artifact["sha256"] != manifest["qualification_suite"]["artifact_sha256"]
            or qualification["record_count"] != profile.record_count
            or qualification["selection_algorithm"]
            != manifest["qualification_suite"]["selection_algorithm"]
            or qualification["selected_ids"]["text"]
            != [record["id"] for record in records]
        ):
            raise ValueError("flagship corpus disagrees with its qualification suite")
        value.update(
            {
                "source": manifest["source"],
                "qualification_suite": manifest["qualification_suite"],
            }
        )
    return value


def _read_regular_file(path: Path, *, expected_bytes: int) -> bytes:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nofollow, int):
        raise RuntimeError("secure bundled corpus loading is unavailable")
    try:
        descriptor = os.open(path, os.O_RDONLY | nofollow)
    except OSError as exc:
        raise RuntimeError(
            "bundled evaluator corpus could not be opened safely"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_bytes:
            raise RuntimeError("bundled evaluator corpus does not have its pinned size")
        chunks: list[bytes] = []
        remaining = expected_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        identity = lambda item: (  # noqa: E731 - stable file projection
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        if identity(before) != identity(after):
            raise RuntimeError("bundled evaluator corpus changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def flagship_records() -> list[dict[str, str]]:
    profile = corpus_profile("flagship")
    expected_bytes = _manifest()["derived_dataset"]["byte_length"]
    payload = _read_regular_file(_FLAGSHIP_RECORDS, expected_bytes=expected_bytes)
    if hashlib.sha256(payload).hexdigest() != profile.dataset_sha256:
        raise RuntimeError(
            "bundled evaluator corpus does not match its pinned identity"
        )
    validate_dataset_records(payload, profile)
    return cast(
        list[dict[str, str]], [json.loads(line) for line in payload.splitlines()]
    )


def quick_records() -> list[dict[str, str]]:
    values = json.loads(_QUICK_RECORDS.read_text(encoding="utf-8"))
    payload = records_jsonl(values)
    validate_dataset_records(payload, corpus_profile("quick"))
    return cast(list[dict[str, str]], values)


__all__ = [
    "CorpusProfile",
    "corpus_profile",
    "corpus_provenance",
    "flagship_records",
    "profile_for_dataset",
    "profile_for_descriptor",
    "quick_records",
    "records_jsonl",
    "validate_dataset_records",
]
