"""Strict replay checks for canonical multimodal dataset evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from invarlock.vision_dataset_evidence import validate_dataset_evidence


def _provider(report: Mapping[str, Any]) -> str:
    dataset = report.get("dataset")
    if not isinstance(dataset, Mapping):
        return ""
    return str(dataset.get("provider") or "").strip().lower()


def _binding_record(record: Mapping[str, Any], *, arm: str) -> dict[str, Any]:
    return {
        "arm": arm,
        "dataset_record_sha256": record.get("dataset_record_sha256"),
        "id": str(record.get("id") or record.get("example_id") or ""),
        "image_sha256": record.get("image_sha256"),
        "record_sha256": record.get("record_sha256"),
    }


def append_strict_vision_evidence_errors(
    errors: list[str], report: Mapping[str, Any]
) -> None:
    if _provider(report) != "vision_text":
        return
    evidence = report.get("dataset_evidence")
    evidence_errors = validate_dataset_evidence(
        evidence,
        strict_counts=True,
        require_runtime_identity=True,
    )
    errors.extend(evidence_errors)
    if not isinstance(evidence, dict) or evidence_errors:
        return

    provenance = report.get("provenance")
    provider_digest = (
        provenance.get("provider_digest") if isinstance(provenance, Mapping) else None
    )
    nested_evidence = (
        provider_digest.get("dataset_evidence")
        if isinstance(provider_digest, Mapping)
        else None
    )
    if nested_evidence != evidence:
        errors.append(
            "strict vision dataset_evidence must exactly match provenance.provider_digest"
        )

    expected_records = evidence.get("records")
    assert isinstance(expected_records, list)
    expected_by_arm = {
        arm: [record for record in expected_records if record.get("arm") == arm]
        for arm in ("preview", "final")
    }
    windows = report.get("evaluation_windows")
    if not isinstance(windows, Mapping):
        errors.append("strict vision evidence requires evaluation_windows")
        return
    runtime_identity = evidence.get("runtime_identity")
    manifest_sha256 = evidence.get("manifest_sha256")
    for arm in ("preview", "final"):
        section = windows.get(arm)
        if not isinstance(section, Mapping):
            errors.append(f"strict vision evidence requires evaluation_windows.{arm}")
            continue
        if section.get("processor_identity") != runtime_identity:
            errors.append(
                f"evaluation_windows.{arm}.processor_identity must match dataset_evidence"
            )
        input_records = section.get("input_records")
        output_records = section.get("records")
        if not isinstance(input_records, list) or not isinstance(output_records, list):
            errors.append(
                f"strict vision evidence requires bound input/output records for {arm}"
            )
            continue
        expected_arm = expected_by_arm[arm]
        if len(input_records) != len(expected_arm) or len(output_records) != len(
            expected_arm
        ):
            errors.append(
                f"strict vision {arm} record count does not match dataset_evidence"
            )
            continue
        for index, expected in enumerate(expected_arm):
            input_record = input_records[index]
            output_record = output_records[index]
            if not isinstance(input_record, Mapping) or not isinstance(
                output_record, Mapping
            ):
                errors.append(f"strict vision {arm} record {index} is malformed")
                continue
            if _binding_record(input_record, arm=arm) != expected:
                errors.append(
                    f"strict vision {arm} input record {index} is not materialization-bound"
                )
            if input_record.get("manifest_sha256") != manifest_sha256:
                errors.append(
                    f"strict vision {arm} input record {index} does not bind manifest bytes"
                )
            if _binding_record(output_record, arm=arm) != expected:
                errors.append(
                    f"strict vision {arm} output record {index} is not materialization-bound"
                )
            if output_record.get("manifest_sha256") != manifest_sha256:
                errors.append(
                    f"strict vision {arm} output record {index} does not bind manifest bytes"
                )


__all__ = ["append_strict_vision_evidence_errors"]
