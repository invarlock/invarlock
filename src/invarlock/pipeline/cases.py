"""Exact planned membership, independent of recorded outputs and execution context."""

from __future__ import annotations

import json
import re
from typing import Any, cast

from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.pipeline.contracts import PipelineError, digest, validate

CASE_SET_FORMAT = "invarlock/pipeline-case-set-v1"
_CASE_FIELDS = ("id", "input", "expected", "metadata")


def canonical_case_set(value: dict[str, Any]) -> dict[str, Any]:
    """Validate and detach a planned artifact, sorting only its outer case list."""
    validate(value, "case_set")
    ids = [row["id"] for row in value["cases"]]
    if len(ids) != len(set(ids)):
        raise PipelineError("planned case set has duplicate case IDs")
    ordered = {**value, "cases": sorted(value["cases"], key=lambda row: row["id"])}
    return cast(dict[str, Any], json.loads(canonical_json_bytes(ordered)))


def case_set_digest(value: dict[str, Any]) -> str:
    """Hash a validated, ID-sorted case set using the pipeline canonical rules."""
    return digest(canonical_case_set(value))


def validate_run_case_set(run: dict[str, Any], expected_digest: str) -> None:
    """Require every planned ID/input/reference/metadata entry, without filtering."""
    if (
        not isinstance(expected_digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", expected_digest) is None
    ):
        raise PipelineError(
            "expected_case_set_digest must be a lowercase sha256 digest"
        )
    validate(run, "run")
    planned = {
        "format": CASE_SET_FORMAT,
        "cases": [{key: row[key] for key in _CASE_FIELDS} for row in run["records"]],
    }
    if case_set_digest(planned) != expected_digest:
        raise PipelineError(
            f"run {run['run_id']!r} does not match the planned case set"
        )
