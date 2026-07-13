"""Canonical semantic evidence for strict ``vision_text`` evaluation lanes."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

DATASET_EVIDENCE_SCHEMA = "dataset_evidence.v1"
STRICT_TOTAL = 800
STRICT_PREVIEW = 400
STRICT_FINAL = 400
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PRIVATE_PATH_RE = re.compile(r"(?:^|\s)/(?:Users|home|root|private|tmp|var/folders)/")
_MATERIALIZATION_FIELDS = frozenset(
    {
        "schema",
        "kind",
        "attestation_scope",
        "dataset",
        "sampling",
        "prompt_template_sha256",
        "manifest_sha256",
        "records",
        "semantic_digest",
    }
)
_EVALUATION_FIELDS = frozenset(
    {
        "schema",
        "kind",
        "attestation_scope",
        "materialization_digest",
        "manifest_sha256",
        "sampling",
        "runtime_identity",
        "records",
        "semantic_digest",
    }
)
_RECORD_FIELDS = frozenset(
    {"arm", "dataset_record_sha256", "id", "image_sha256", "record_sha256"}
)
_DATASET_FIELDS = frozenset({"config_name", "id", "revision", "split"})
_RUNTIME_IDENTITY_FIELDS = frozenset(
    {"tokenizer_sha256", "processor_sha256", "chat_template_sha256"}
)
_MATERIALIZATION_SCOPE = {
    "covers": [
        "dataset coordinates",
        "selected record identities",
        "materialized image bytes",
        "prompt and answer bindings",
    ],
    "excludes": [
        "model quality",
        "processor runtime identity",
        "runtime image identity",
    ],
}
_EVALUATION_SCOPE = {
    "covers": [
        "materialization semantic identity",
        "preview/final schedule",
        "evaluation-record input bindings",
        "tokenizer, processor, and chat-template identity",
    ],
    "excludes": ["runtime image trust", "model quality beyond recorded metrics"],
}
_MATERIALIZATION_SUMMARY_FIELDS = frozenset(
    {
        "generated_at",
        "selected_count",
        "record_count",
        "skipped_count",
        "max_samples",
        "seed",
        "shuffle",
        "image_format",
        "manifest",
        "hashes",
        "fields",
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON canonically; arrays make concatenated/variable-length IDs safe."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def semantic_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def dataset_record_digest(
    *,
    dataset: str,
    revision: str | None,
    split: str,
    row_index: int,
    record_id: str,
    question: str,
    answers: Sequence[str],
) -> str:
    return semantic_digest(
        {
            "answers": list(answers),
            "dataset": dataset,
            "id": record_id,
            "question": question,
            "revision": revision,
            "row_index": row_index,
            "split": split,
        }
    )


def materialized_record_digest(record: Mapping[str, Any]) -> str:
    source = record.get("source")
    source_map = source if isinstance(source, Mapping) else {}
    return semantic_digest(
        {
            "answer_sha256": source_map.get("answer_sha256"),
            "dataset_record_sha256": source_map.get("dataset_record_sha256"),
            "id": record.get("id"),
            "image_sha256": source_map.get("image_sha256"),
            "prompt_sha256": source_map.get("prompt_sha256"),
        }
    )


def build_materialization_evidence(
    *,
    dataset: str,
    revision: str | None,
    config_name: str | None,
    split: str,
    seed: int,
    shuffle: bool,
    prompt_template_sha256: str,
    manifest_sha256: str,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    record_entries: list[dict[str, Any]] = []
    midpoint = len(records) // 2
    for index, record in enumerate(records):
        source = record.get("source")
        source_map = source if isinstance(source, Mapping) else {}
        record_entries.append(
            {
                "arm": "preview" if index < midpoint else "final",
                "dataset_record_sha256": source_map.get("dataset_record_sha256"),
                "id": str(record.get("id") or ""),
                "image_sha256": source_map.get("image_sha256"),
                "record_sha256": source_map.get("record_sha256"),
            }
        )
    payload: dict[str, Any] = {
        "schema": DATASET_EVIDENCE_SCHEMA,
        "kind": "materialization",
        "attestation_scope": copy.deepcopy(_MATERIALIZATION_SCOPE),
        "dataset": {
            "config_name": config_name,
            "id": dataset,
            "revision": revision,
            "split": split,
        },
        "sampling": {
            "final": len(records) - midpoint,
            "preview": midpoint,
            "seed": seed,
            "shuffle": shuffle,
            "total": len(records),
        },
        "prompt_template_sha256": prompt_template_sha256,
        "manifest_sha256": manifest_sha256,
        "records": record_entries,
    }
    payload["semantic_digest"] = semantic_digest(payload)
    return payload


def _semantic_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    semantic_keys = (
        "schema",
        "kind",
        "attestation_scope",
        "dataset",
        "sampling",
        "prompt_template_sha256",
        "manifest_sha256",
        "materialization_digest",
        "runtime_identity",
        "records",
    )
    return {key: copy.deepcopy(payload[key]) for key in semantic_keys if key in payload}


def _contains_private_path(value: Any) -> bool:
    if isinstance(value, str):
        if _PRIVATE_PATH_RE.search(value) or value.startswith("~") or "\\" in value:
            return True
        candidate = PurePosixPath(value)
        return candidate.is_absolute() and not value.startswith("sha256:")
    if isinstance(value, Mapping):
        return any(_contains_private_path(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_private_path(item) for item in value)
    return False


def _contains_non_finite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, Mapping):
        return any(_contains_non_finite(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_non_finite(item) for item in value)
    return False


def _record_errors(records: Any, *, strict_counts: bool) -> list[str]:
    errors: list[str] = []
    if not isinstance(records, list):
        return ["dataset_evidence.records must be a list"]
    ids: list[str] = []
    arms: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"dataset_evidence.records[{index}] must be an object")
            continue
        if set(record) != _RECORD_FIELDS:
            errors.append(
                f"dataset_evidence.records[{index}] must contain exact v1 fields"
            )
        record_id = record.get("id")
        if (
            not isinstance(record_id, str)
            or not record_id
            or record_id != record_id.strip()
            or len(record_id.encode("utf-8")) > 4096
        ):
            errors.append(f"dataset_evidence.records[{index}].id is invalid")
        else:
            ids.append(record_id)
        arm = record.get("arm")
        if arm not in {"preview", "final"}:
            errors.append(f"dataset_evidence.records[{index}].arm is invalid")
        else:
            arms.append(str(arm))
        for key in ("dataset_record_sha256", "record_sha256"):
            if not isinstance(record.get(key), str) or not _DIGEST_RE.fullmatch(
                str(record.get(key))
            ):
                errors.append(f"dataset_evidence.records[{index}].{key} is invalid")
        image_digest = record.get("image_sha256")
        if not isinstance(image_digest, str) or not _HEX_RE.fullmatch(image_digest):
            errors.append(f"dataset_evidence.records[{index}].image_sha256 is invalid")
    if len(ids) != len(set(ids)):
        errors.append("dataset_evidence record IDs must be unique")
    if strict_counts:
        if len(records) != STRICT_TOTAL:
            errors.append("strict vision dataset_evidence requires exactly 800 records")
        if (
            arms.count("preview") != STRICT_PREVIEW
            or arms.count("final") != STRICT_FINAL
        ):
            errors.append(
                "strict vision dataset_evidence requires 400 preview and 400 final records"
            )
        expected_arms = (["preview"] * STRICT_PREVIEW) + (["final"] * STRICT_FINAL)
        if arms != expected_arms:
            errors.append(
                "strict vision dataset_evidence requires the canonical 400/400 arm schedule"
            )
    return errors


def _sampling_errors(sampling: object, *, strict_counts: bool) -> list[str]:
    if not isinstance(sampling, dict):
        return ["dataset_evidence.sampling must be an object"]
    errors: list[str] = []
    if set(sampling) != {"final", "preview", "seed", "shuffle", "total"}:
        errors.append("dataset_evidence.sampling has non-canonical fields")
    if isinstance(sampling.get("seed"), bool) or not isinstance(
        sampling.get("seed"), int
    ):
        errors.append("dataset_evidence.sampling.seed must be an integer")
    if not isinstance(sampling.get("shuffle"), bool):
        errors.append("dataset_evidence.sampling.shuffle must be a boolean")
    for field in ("final", "preview", "total"):
        value = sampling.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(
                f"dataset_evidence.sampling.{field} must be a non-negative integer"
            )
    preview_count = sampling.get("preview")
    final_count = sampling.get("final")
    if (
        isinstance(preview_count, int)
        and not isinstance(preview_count, bool)
        and isinstance(final_count, int)
        and not isinstance(final_count, bool)
        and sampling.get("total") != preview_count + final_count
    ):
        errors.append("dataset_evidence.sampling total must equal preview plus final")
    if strict_counts and (
        sampling.get("final") != STRICT_FINAL
        or sampling.get("preview") != STRICT_PREVIEW
        or sampling.get("total") != STRICT_TOTAL
    ):
        errors.append(
            "strict vision dataset_evidence sampling must be exactly 800/400/400"
        )
    return errors


def _runtime_identity_errors(identity: object, *, required: bool) -> list[str]:
    errors: list[str] = []
    if required and not isinstance(identity, dict):
        errors.append("strict vision dataset_evidence requires runtime_identity")
    if identity is None:
        return errors
    if not isinstance(identity, dict) or set(identity) != _RUNTIME_IDENTITY_FIELDS:
        return errors + ["strict vision dataset_evidence requires runtime_identity"]
    for key in ("tokenizer_sha256", "processor_sha256", "chat_template_sha256"):
        if not isinstance(identity.get(key), str) or not _DIGEST_RE.fullmatch(
            str(identity.get(key))
        ):
            errors.append(f"strict vision dataset_evidence requires {key}")
    return errors


def _kind_specific_errors(payload: dict[str, Any], *, kind: object) -> list[str]:
    errors: list[str] = []
    if kind == "materialization":
        dataset = payload.get("dataset")
        if (
            not isinstance(dataset, dict)
            or set(dataset) != _DATASET_FIELDS
            or not isinstance(dataset.get("id"), str)
            or not dataset.get("id")
            or not isinstance(dataset.get("split"), str)
            or not dataset.get("split")
        ):
            errors.append(
                "materialization dataset_evidence requires dataset coordinates"
            )
        elif any(
            dataset.get(field) is not None
            and (not isinstance(dataset.get(field), str) or not dataset.get(field))
            for field in ("config_name", "revision")
        ):
            errors.append("materialization dataset optional coordinates are invalid")
        prompt_digest = payload.get("prompt_template_sha256")
        if not isinstance(prompt_digest, str) or not _HEX_RE.fullmatch(prompt_digest):
            errors.append(
                "materialization dataset_evidence requires prompt template identity"
            )
    manifest_digest = payload.get("manifest_sha256")
    if not isinstance(manifest_digest, str) or not _DIGEST_RE.fullmatch(
        manifest_digest
    ):
        errors.append("dataset_evidence requires manifest_sha256")
    if kind == "evaluation":
        materialization_digest = payload.get("materialization_digest")
        if not isinstance(materialization_digest, str) or not _DIGEST_RE.fullmatch(
            materialization_digest
        ):
            errors.append("evaluation dataset_evidence requires materialization_digest")
    return errors


def validate_dataset_evidence(
    payload: Any,
    *,
    strict_counts: bool,
    require_runtime_identity: bool,
    allow_materialization_summary_fields: bool = False,
) -> list[str]:
    if not isinstance(payload, dict):
        return ["strict vision evidence requires dataset_evidence as an object"]
    errors: list[str] = []
    if payload.get("schema") != DATASET_EVIDENCE_SCHEMA:
        errors.append("strict vision evidence requires dataset_evidence.v1")
    kind = payload.get("kind")
    if kind not in {"materialization", "evaluation"}:
        errors.append("dataset_evidence.kind must be materialization or evaluation")
    allowed_fields = (
        _MATERIALIZATION_FIELDS
        | (
            _MATERIALIZATION_SUMMARY_FIELDS
            if allow_materialization_summary_fields
            else set()
        )
        if kind == "materialization"
        else _EVALUATION_FIELDS
    )
    unsupported_fields = sorted(set(payload) - allowed_fields)
    if unsupported_fields:
        errors.append(
            "dataset_evidence contains unsupported fields: "
            + ", ".join(unsupported_fields)
        )
    missing_fields = sorted(allowed_fields - set(payload))
    if missing_fields:
        errors.append(
            "dataset_evidence is missing required fields: " + ", ".join(missing_fields)
        )
    if _contains_private_path(payload):
        errors.append("dataset_evidence must not contain local or private paths")
    observed = payload.get("semantic_digest")
    expected = (
        None
        if _contains_non_finite(payload)
        else semantic_digest(_semantic_payload(payload))
    )
    if expected is None:
        errors.append("dataset_evidence must not contain non-finite values")
    elif observed != expected:
        errors.append("dataset_evidence semantic digest mismatch")
    scope = payload.get("attestation_scope")
    expected_scope = (
        _MATERIALIZATION_SCOPE if kind == "materialization" else _EVALUATION_SCOPE
    )
    if scope != expected_scope:
        errors.append("dataset_evidence requires the exact v1 attestation_scope")
    errors.extend(
        _sampling_errors(payload.get("sampling"), strict_counts=strict_counts)
    )
    errors.extend(_record_errors(payload.get("records"), strict_counts=strict_counts))
    errors.extend(_kind_specific_errors(payload, kind=kind))
    errors.extend(
        _runtime_identity_errors(
            payload.get("runtime_identity"), required=require_runtime_identity
        )
    )
    return errors


def validate_evaluation_materialization_binding(
    materialization: Any,
    evaluation: Any,
    *,
    strict_counts: bool,
) -> list[str]:
    """Require evaluation evidence to name the exact materialized record schedule."""

    errors = validate_dataset_evidence(
        materialization,
        strict_counts=strict_counts,
        require_runtime_identity=False,
    )
    errors.extend(
        validate_dataset_evidence(
            evaluation,
            strict_counts=strict_counts,
            require_runtime_identity=True,
        )
    )
    if not isinstance(materialization, Mapping) or not isinstance(evaluation, Mapping):
        return errors
    expected_digest = materialization.get("semantic_digest")
    if evaluation.get("materialization_digest") != expected_digest:
        errors.append(
            "evaluation dataset_evidence materialization_digest does not match"
        )
    if evaluation.get("manifest_sha256") != materialization.get("manifest_sha256"):
        errors.append("evaluation dataset_evidence manifest_sha256 does not match")
    if evaluation.get("sampling") != materialization.get("sampling"):
        errors.append(
            "evaluation dataset_evidence sampling does not match materialization"
        )
    if evaluation.get("records") != materialization.get("records"):
        errors.append(
            "evaluation dataset_evidence records do not match materialization"
        )
    return errors


def build_report_evidence(
    *,
    materialization_digest: str,
    manifest_sha256: str,
    sampling: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": DATASET_EVIDENCE_SCHEMA,
        "kind": "evaluation",
        "attestation_scope": copy.deepcopy(_EVALUATION_SCOPE),
        "materialization_digest": materialization_digest,
        "manifest_sha256": manifest_sha256,
        "sampling": dict(sampling),
        "runtime_identity": dict(runtime_identity),
        "records": [dict(record) for record in records],
    }
    payload["semantic_digest"] = semantic_digest(payload)
    return payload


def build_report_evidence_from_run_report(
    report: Mapping[str, Any],
) -> dict[str, Any] | None:
    windows = report.get("evaluation_windows")
    if not isinstance(windows, Mapping):
        return None
    records: list[dict[str, Any]] = []
    materialization_digests: set[str] = set()
    manifest_digests: set[str] = set()
    runtime_identities: list[dict[str, Any]] = []
    arm_counts: dict[str, int] = {"preview": 0, "final": 0}
    for arm in ("preview", "final"):
        section = windows.get(arm)
        if not isinstance(section, Mapping):
            return None
        identity = section.get("processor_identity")
        if isinstance(identity, Mapping):
            runtime_identities.append(dict(identity))
        input_records = section.get("input_records")
        if not isinstance(input_records, list):
            return None
        arm_counts[arm] = len(input_records)
        for record in input_records:
            if not isinstance(record, Mapping):
                return None
            record_id = record.get("id")
            if not isinstance(record_id, str) or not record_id:
                return None
            materialization_digest = record.get("materialization_digest")
            if isinstance(materialization_digest, str):
                materialization_digests.add(materialization_digest)
            manifest_digest = record.get("manifest_sha256")
            if isinstance(manifest_digest, str):
                manifest_digests.add(manifest_digest)
            records.append(
                {
                    "arm": arm,
                    "dataset_record_sha256": record.get("dataset_record_sha256"),
                    "id": record_id,
                    "image_sha256": record.get("image_sha256"),
                    "record_sha256": record.get("record_sha256"),
                }
            )
    if (
        len(materialization_digests) != 1
        or len(manifest_digests) != 1
        or not runtime_identities
    ):
        return None
    if any(identity != runtime_identities[0] for identity in runtime_identities[1:]):
        return None
    meta = report.get("meta")
    seed = meta.get("seed") if isinstance(meta, Mapping) else None
    return build_report_evidence(
        materialization_digest=next(iter(materialization_digests)),
        manifest_sha256=next(iter(manifest_digests)),
        sampling={
            "final": arm_counts["final"],
            "preview": arm_counts["preview"],
            "seed": seed,
            "shuffle": False,
            "total": arm_counts["preview"] + arm_counts["final"],
        },
        runtime_identity=runtime_identities[0],
        records=records,
    )


__all__ = [
    "DATASET_EVIDENCE_SCHEMA",
    "STRICT_FINAL",
    "STRICT_PREVIEW",
    "STRICT_TOTAL",
    "build_materialization_evidence",
    "build_report_evidence",
    "build_report_evidence_from_run_report",
    "canonical_json_bytes",
    "dataset_record_digest",
    "materialized_record_digest",
    "semantic_digest",
    "validate_evaluation_materialization_binding",
    "validate_dataset_evidence",
]
