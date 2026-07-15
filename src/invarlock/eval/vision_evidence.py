"""Materialization-evidence loading for the ``vision_text`` provider."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from invarlock.evidence_pack_json import (
    StrictJsonError,
    read_json_object_snapshot,
    read_jsonl_snapshot,
    sha256_prefixed,
)
from invarlock.vision_dataset_evidence import (
    canonical_json_bytes,
    dataset_record_digest,
    materialized_record_digest,
    validate_dataset_evidence,
)


@dataclass(frozen=True)
class VisionMaterializationSnapshot:
    manifest_bytes: bytes
    records: tuple[dict[str, Any], ...]
    manifest_sha256: str
    materialization_digest: str
    bindings: dict[str, dict[str, Any]]
    dataset: dict[str, str | None]


def load_materialization_snapshot(
    manifest_path: Path,
) -> VisionMaterializationSnapshot:
    """Strictly snapshot and cross-bind evidence plus its exact manifest bytes."""

    evidence_path = manifest_path.parent / "dataset_evidence.json"
    try:
        _, payload = read_json_object_snapshot(
            evidence_path, label="vision_text dataset evidence"
        )
    except (OSError, StrictJsonError) as exc:
        raise ValueError("vision_text dataset_evidence.json is invalid") from exc
    try:
        manifest_bytes, raw_records = read_jsonl_snapshot(
            manifest_path, label="vision_text manifest"
        )
    except (OSError, StrictJsonError) as exc:
        raise ValueError("vision_text manifest is invalid") from exc
    errors = validate_dataset_evidence(
        payload,
        strict_counts=False,
        require_runtime_identity=False,
    )
    if errors:
        raise ValueError(
            "vision_text dataset evidence is invalid: " + "; ".join(errors)
        )
    evidence_records = payload.get("records")
    assert isinstance(evidence_records, list)
    if not all(isinstance(record, dict) for record in raw_records):
        raise ValueError("vision_text manifest records must be JSON objects")
    records = tuple(dict(record) for record in raw_records)
    manifest_sha256 = sha256_prefixed(manifest_bytes)
    if payload.get("manifest_sha256") != manifest_sha256:
        raise ValueError("vision_text manifest bytes do not match dataset evidence")
    manifest_ids = [record.get("id") for record in records]
    evidence_ids = [record.get("id") for record in evidence_records]
    if manifest_ids != evidence_ids:
        raise ValueError(
            "vision_text manifest record order does not match dataset evidence"
        )
    bindings = {
        str(record["id"]): dict(record)
        for record in evidence_records
        if isinstance(record, Mapping) and isinstance(record.get("id"), str)
    }
    dataset = payload.get("dataset")
    assert isinstance(dataset, dict)
    return VisionMaterializationSnapshot(
        manifest_bytes=manifest_bytes,
        records=records,
        manifest_sha256=manifest_sha256,
        materialization_digest=str(payload["semantic_digest"]),
        bindings=bindings,
        dataset={
            "id": str(dataset["id"]),
            "config_name": (
                str(dataset["config_name"])
                if dataset.get("config_name") is not None
                else None
            ),
            "revision": (
                str(dataset["revision"])
                if dataset.get("revision") is not None
                else None
            ),
            "split": str(dataset["split"]),
        },
    )


def bind_loaded_record(
    *,
    record_id: str,
    raw_record: Mapping[str, Any],
    observed_image_sha256: str,
    materialization_digest: str | None,
    manifest_sha256: str,
    bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if materialization_digest is None:
        return {}
    binding = bindings.get(record_id)
    if not isinstance(binding, Mapping):
        raise ValueError(
            f"vision_text record {record_id!r} is absent from dataset evidence"
        )
    source = raw_record.get("source")
    source_map = source if isinstance(source, Mapping) else {}
    answers = raw_record.get("answers")
    if not isinstance(answers, list) or not all(
        isinstance(item, str) for item in answers
    ):
        raise ValueError("vision_text manifest answers are not canonically bound")
    if not answers or raw_record.get("answer") != answers[0]:
        raise ValueError("vision_text manifest primary answer is not canonically bound")
    prompt = raw_record.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("vision_text manifest prompt is not canonically bound")
    declared_hashes = {
        "answer_sha256": hashlib.sha256(canonical_json_bytes(answers)).hexdigest(),
        "image_sha256": observed_image_sha256,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }
    for key, expected_digest in declared_hashes.items():
        if source_map.get(key) != expected_digest:
            raise ValueError(f"vision_text manifest {key} is invalid")
    row_index = source_map.get("row_index")
    if isinstance(row_index, bool) or not isinstance(row_index, int):
        raise ValueError("vision_text manifest row_index is not canonically bound")
    observed_dataset_digest = dataset_record_digest(
        dataset=str(source_map.get("dataset") or ""),
        revision=(
            str(source_map["revision"])
            if source_map.get("revision") is not None
            else None
        ),
        split=str(source_map.get("split") or ""),
        row_index=row_index,
        record_id=record_id,
        question=str(source_map.get("question") or ""),
        answers=answers,
    )
    if source_map.get("dataset_record_sha256") != observed_dataset_digest:
        raise ValueError("vision_text manifest dataset record digest is invalid")
    if source_map.get("record_sha256") != materialized_record_digest(raw_record):
        raise ValueError("vision_text manifest materialized record digest is invalid")
    expected_image = binding.get("image_sha256")
    if expected_image != observed_image_sha256:
        raise ValueError(
            "vision_text materialized image bytes changed after attestation"
        )
    for key in ("dataset_record_sha256", "record_sha256"):
        if source_map.get(key) != binding.get(key):
            raise ValueError(
                f"vision_text manifest {key} is not bound to dataset evidence"
            )
    return {
        "dataset_record_sha256": binding["dataset_record_sha256"],
        "materialization_digest": materialization_digest,
        "manifest_sha256": manifest_sha256,
        "record_sha256": binding["record_sha256"],
    }


__all__ = [
    "VisionMaterializationSnapshot",
    "bind_loaded_record",
    "load_materialization_snapshot",
]
