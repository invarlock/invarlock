"""Evaluator-neutral case capture and explicit, replayable text preparation.

Capturing records does not execute or qualify the named evaluator. Callers own
the mapping from their evaluator's cases; aggregate results are not case rows.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from invarlock.evaluation_record_contracts.contracts import (
    MAX_RECORDS,
    EvaluationRecordsError,
    digest,
)


def _projection_tokens(configuration: Any) -> list[str]:
    if (
        not isinstance(configuration, Mapping)
        or set(configuration) != {"kind", "pointer"}
        or configuration["kind"] != "json-pointer"
    ):
        raise EvaluationRecordsError("input projection requires kind and pointer only")
    pointer = configuration["pointer"]
    if (
        not isinstance(pointer, str)
        or len(pointer) > 4096
        or not pointer.startswith("/")
        or re.search(r"~(?![01])", pointer)
    ):
        raise EvaluationRecordsError(
            "input projection requires an RFC6901 JSON pointer"
        )
    tokens = [
        part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")
    ]
    if tokens[0] not in ("input", "context"):
        raise EvaluationRecordsError("input projection must select input or context")
    return tokens


def _project_text(source: dict[str, Any], tokens: list[str]) -> str:
    value: Any = source
    for token in tokens:
        if isinstance(value, dict) and token in value:
            value = value[token]
        elif (
            isinstance(value, list)
            and re.fullmatch(r"0|[1-9][0-9]*", token)
            and len(token) <= len(str(len(value)))
            and int(token) < len(value)
        ):
            value = value[int(token)]
        else:
            raise EvaluationRecordsError("input projection pointer does not resolve")
    if not isinstance(value, str):
        raise EvaluationRecordsError("input projection must select one text string")
    return value


def _project_record(
    row: dict[str, Any], configuration: Mapping[str, Any]
) -> dict[str, Any]:
    tokens = _projection_tokens(configuration)
    context = row.get("context", {})
    if isinstance(context, dict) and "input_projection" in context:
        raise EvaluationRecordsError("input projection context key is reserved")
    source = {"input": row["input"], "context": context}
    return {
        **row,
        "input": _project_text(source, tokens),
        "context": {
            **(context if isinstance(context, dict) else {}),
            "input_projection": {
                "configuration": dict(configuration),
                "configuration_digest": digest(dict(configuration)),
                "source": source,
                "source_digest": digest(source),
            },
        },
    }


def verify_input_projection(record: dict[str, Any]) -> None:
    """Authenticate and replay a retained projection without external inputs."""
    context = record.get("context")
    if not isinstance(context, dict) or "input_projection" not in context:
        return
    binding = context["input_projection"]
    if not isinstance(binding, dict) or set(binding) != {
        "configuration",
        "configuration_digest",
        "source",
        "source_digest",
    }:
        raise EvaluationRecordsError("input projection binding has invalid fields")
    source = binding["source"]
    if not isinstance(source, dict) or set(source) != {"input", "context"}:
        raise EvaluationRecordsError("input projection source has invalid fields")
    configuration = binding["configuration"]
    tokens = _projection_tokens(configuration)
    if (
        digest(configuration) != binding["configuration_digest"]
        or digest(source) != binding["source_digest"]
        or _project_text(source, tokens) != record["input"]
    ):
        raise EvaluationRecordsError(
            "input projection does not replay its retained binding"
        )
    original_context = source["context"]
    expected_context = original_context if isinstance(original_context, dict) else {}
    if "input_projection" in expected_context or digest(
        {key: value for key, value in context.items() if key != "input_projection"}
    ) != digest(expected_context):
        raise EvaluationRecordsError("input projection original context does not match")


def verify_input_pair(left: dict[str, Any], right: dict[str, Any]) -> None:
    """Keep source case identity and the chosen mapping consistent across a pair."""
    left_context, right_context = left["context"], right["context"]
    left_binding = (
        left_context.get("input_projection") if isinstance(left_context, dict) else None
    )
    right_binding = (
        right_context.get("input_projection")
        if isinstance(right_context, dict)
        else None
    )
    if left_binding is None and right_binding is None:
        return
    if (
        left_binding is None
        or right_binding is None
        or left_binding["configuration_digest"] != right_binding["configuration_digest"]
        or digest(left_binding["source"]["input"])
        != digest(right_binding["source"]["input"])
    ):
        raise EvaluationRecordsError(
            "paired input projection or original input changed"
        )


def capture_evaluator_run(
    records: list[dict[str, Any]],
    *,
    source: Mapping[str, str],
    run_id: str,
    artifact_digest: str | None,
    service_identity: Mapping[str, Any] | None = None,
    source_digest: str | None = None,
    score_provenance: Mapping[str, Any] | None = None,
    input_projection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture explicit case rows from any evaluator with its real name/version.

    Records must carry stable paired IDs, original inputs, references and outputs.
    Optional likelihoods and scores remain captured observations. No runtime,
    qualification, or scorer authority follows from the source's name.
    """
    from invarlock.evaluation_comparison.comparison import _check_run, make_run

    if (
        not isinstance(records, list)
        or not 1 <= len(records) <= MAX_RECORDS
        or any(not isinstance(row, dict) for row in records)
    ):
        raise EvaluationRecordsError(
            "capture requires a non-empty bounded list of case records"
        )
    if not isinstance(source, Mapping):
        raise EvaluationRecordsError("capture requires explicit source name/version")
    if score_provenance is not None and not isinstance(score_provenance, Mapping):
        raise EvaluationRecordsError("capture score provenance must be an object")
    run = make_run(
        records,
        source=dict(source),
        run_id=run_id,
        artifact_digest=artifact_digest,
        service_identity=service_identity,
        source_digest=source_digest,
        score_provenance=dict(score_provenance)
        if score_provenance is not None
        else None,
    )
    if input_projection is not None:
        run["records"] = [
            _project_record(row, input_projection) for row in run["records"]
        ]
        _check_run(run)
    return run


def evaluator_input_capabilities(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Count usable captured inputs; availability does not establish assurance."""
    from invarlock.evaluation_comparison.comparison import _check_run

    _check_run(run)
    required_facts = {
        "exact_match": ["text_reference", "text_output", "no_row_error"],
        "normalized_nll_per_utf8_byte": ["bound_reference_likelihood", "no_row_error"],
        "judge": [
            "text_input",
            "text_output",
            "text_or_absent_reference",
            "no_row_error",
        ],
    }
    result: dict[str, dict[str, Any]] = {
        name: {"usable_count": 0, "unavailable_ids": [], "required_facts": facts}
        for name, facts in required_facts.items()
    }
    for row in run["records"]:
        text_reference = isinstance(row["expected"], str)
        text_output = isinstance(row["output"], str)
        available = {
            "exact_match": text_reference and text_output,
            "normalized_nll_per_utf8_byte": "likelihood" in row,
            "judge": isinstance(row["input"], str)
            and text_output
            and (row["expected"] is None or text_reference),
        }
        for name, usable in available.items():
            if usable and row["error"] is None:
                result[name]["usable_count"] += 1
            else:
                result[name]["unavailable_ids"].append(row["id"])
    return result


__all__ = ["capture_evaluator_run", "evaluator_input_capabilities"]
