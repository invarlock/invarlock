"""Pure, portable identities for independently approved captured inputs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from jsonschema import Draft202012Validator

from invarlock.core.scoring import MetricError, validate_configuration
from invarlock.evaluation_record_contracts.contracts import (
    EvaluationRecordsError,
    _canonical_chunks,
    digest,
    validate,
)
from invarlock.evaluation_records.cases import validate_run_case_set
from invarlock.evaluation_records.io import run_digest
from invarlock.public_contracts import (
    load_captured_evaluation_request_schema,
    load_normalized_captured_request_schema,
)


def _validate_request(value: dict[str, Any], *, normalized: bool) -> None:
    size = 0
    for chunk in _canonical_chunks(value):
        size += len(chunk)
        if size > 1024 * 1024:
            raise EvaluationRecordsError(
                "captured request exceeds the 1 MiB byte limit"
            )
    pending: list[Any] = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, str) and any(ord(c) < 32 or ord(c) == 127 for c in item):
            raise EvaluationRecordsError(
                "captured request strings cannot contain control characters"
            )
    schema = (
        load_normalized_captured_request_schema()
        if normalized
        else load_captured_evaluation_request_schema()
    )
    error = next(Draft202012Validator(schema).iter_errors(value), None)
    if error is not None:
        raise EvaluationRecordsError(f"invalid captured request: {error.message}")


def comparison_policy_digest(policy: Mapping[str, Any]) -> str:
    """Hash canonical policy JSON after portable validation, without scoring."""
    value = dict(policy)
    validate(value, "policy")
    names = [m["name"] for m in value["metrics"]]
    slices = [s["name"] for s in value["slices"]]
    if (
        len(names) != len(set(names))
        or len(slices) != len(set(slices))
        or "overall" in slices
    ):
        raise EvaluationRecordsError(
            "metric and slice names must be unique; overall is reserved"
        )
    for metric in value["metrics"]:
        kind, config = metric["kind"], metric["configuration"]
        if kind == "recorded":
            provenance = metric.get("accepted_provenance")
            if not provenance or not metric.get("score_key") or config:
                raise EvaluationRecordsError(
                    "recorded metric requires provenance and score_key, without configuration"
                )
            if provenance["unit"] != metric["unit"] or (
                provenance["kind"] in {"judge", "human"}
                and provenance["rubric_digest"] is None
            ):
                raise EvaluationRecordsError(
                    "recorded metric provenance is inconsistent"
                )
        else:
            if "score_key" in metric or "accepted_provenance" in metric:
                raise EvaluationRecordsError(
                    "recomputed metrics cannot accept recorded provenance"
                )
            if metric["direction"] != "higher" or metric["unit"] != "score":
                raise EvaluationRecordsError(
                    "deterministic scorers require higher-is-better score units"
                )
            try:
                validate_configuration(kind, config, portable=True)
            except MetricError as exc:
                raise EvaluationRecordsError(str(exc)) from exc
        if metric.get("subject_minimum", -math.inf) > metric.get(
            "subject_maximum", math.inf
        ):
            raise EvaluationRecordsError("subject minimum exceeds maximum")
    return digest(value)


def captured_comparison_id(
    *,
    request_digest: str,
    baseline_run_digest: str,
    subject_run_digest: str,
    policy_digest: str,
) -> str:
    """Identify captured input intent, not its report or publication environment."""
    return digest(
        {
            "kind": "captured",
            "request_digest": request_digest,
            "baseline_run_digest": baseline_run_digest,
            "subject_run_digest": subject_run_digest,
            "policy_digest": policy_digest,
        }
    )


def normalize_captured_request(
    request: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any],
    subject: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind effective source intent and approved runs; never access a host path."""
    authored = dict(request)
    _validate_request(authored, normalized=False)
    policy_digest = comparison_policy_digest(policy)
    comparison: dict[str, Any] = {"policy_digest": policy_digest}
    for side, run in (("baseline", baseline), ("subject", subject)):
        source = authored["comparison"][side]
        actual = run_digest(run)
        if "expected_case_set_digest" in policy:
            validate_run_case_set(dict(run), policy["expected_case_set_digest"])
        if "expected_run_digest" in source and source["expected_run_digest"] != actual:
            raise EvaluationRecordsError(f"{side} run digest differs from expected pin")
        identities = ("source", "run_id", "artifact_digest", "score_provenance")
        if source["adapter"] == "invarlock":
            if any(key in source for key in identities):
                raise EvaluationRecordsError(
                    "canonical run identities cannot be overridden at import"
                )
        else:
            if (
                any(key not in source for key in identities[:3])
                or run.get("source_digest") is None
            ):
                raise EvaluationRecordsError(
                    "native import requires source, run and artifact identities and physical source digest"
                )
            for key in identities:
                if source.get(key, {}) != run.get(key):
                    raise EvaluationRecordsError(
                        f"{side} {key} differs from declared source intent"
                    )
        comparison[side] = {
            key: value for key, value in source.items() if key != "path"
        }
        comparison[side]["run_digest"] = actual
    normalized = {
        "format": authored["format_version"],
        "execution": {"mode": "captured"},
        "comparison": comparison,
    }
    _validate_request(normalized, normalized=True)
    # Detach nested authored source/provenance maps without changing array order.
    from invarlock.evidence_pack_json import parse_json_bytes

    result = parse_json_bytes(
        b"".join(_canonical_chunks(normalized)), label="normalized request"
    )
    assert isinstance(result, dict)
    return result


def captured_request_digest(normalized_request: Mapping[str, Any]) -> str:
    """Hash only the closed path-free normalized request, never authored YAML."""
    value = dict(normalized_request)
    _validate_request(value, normalized=True)
    for side in ("baseline", "subject"):
        source = value["comparison"][side]
        if (
            source.get("expected_run_digest", source["run_digest"])
            != source["run_digest"]
        ):
            raise EvaluationRecordsError(f"{side} normalized run pin is inconsistent")
    return digest(value)
