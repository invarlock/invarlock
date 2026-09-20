"""Installed parsers for existing exports; no evaluator SDK or inference required."""

from __future__ import annotations

import hashlib
import io
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from invarlock.evaluation_record_contracts.contracts import (
    MAX_INPUT_BYTES,
    MAX_RECORDS,
    EvaluationRecordsError,
)
from invarlock.evaluation_records.capture_facts import merge_capture_scores
from invarlock.evaluator_capture import capture_evaluator_run
from invarlock.evidence_pack_json import parse_json_bytes

ADAPTERS = (
    "invarlock",
    "jsonl",
    "inspect-json",
    "lm-eval-samples",
    "promptfoo-jsonl",
    "langfuse-json",
    "evaluator-json",
    "evaluator-native-json",
)


def _rows(value: Any, label: str) -> list[dict[str, Any]]:
    if (
        not isinstance(value, list)
        or not value
        or len(value) > MAX_RECORDS
        or any(not isinstance(row, dict) for row in value)
    ):
        raise EvaluationRecordsError(
            f"{label} requires a non-empty array of record objects"
        )
    return value


def _jsonl_rows(raw: bytes, label: str) -> list[dict[str, Any]]:
    """Parse non-empty JSONL rows without expanding work past the record limit."""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(io.BytesIO(raw), 1):
        if line.isspace():
            continue
        if len(rows) == MAX_RECORDS:
            raise EvaluationRecordsError(
                f"{label} exceeds the {MAX_RECORDS}-record limit"
            )
        row = parse_json_bytes(line, label=f"export line {line_number}")
        if not isinstance(row, dict):
            raise EvaluationRecordsError(
                f"{label} requires a non-empty array of record objects"
            )
        rows.append(row)
    return _rows(rows, label)


def _scalar_output(value: Any) -> Any:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, str):
        raise EvaluationRecordsError(
            "ambiguous or non-text native completion; select one response in your evaluator"
        )
    return value


def _scores(
    values: dict[str, Any], *, allow_native_values: bool = False
) -> dict[str, float]:
    if not isinstance(values, dict):
        raise EvaluationRecordsError("native scores must be an object")
    result = {}
    for key, value in values.items():
        if isinstance(value, dict):
            value = value.get("value")
        if isinstance(value, str) and value in ("C", "I"):
            value = 1.0 if value == "C" else 0.0
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            if allow_native_values:
                # Categorical/structured grades remain recorded source data;
                # they do not need a numeric mapping for InvarLock to rescore.
                continue
            raise EvaluationRecordsError(
                f"score {key!r} is not numeric; provide an explicit SDK mapping"
            )
        if not isinstance(key, str) or not key:
            raise EvaluationRecordsError("native scores require nonempty names")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise EvaluationRecordsError("native scores must be finite")
        result[key] = float(value)
    return result


def _native_object(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise EvaluationRecordsError(f"native {label} must be an object or null")
    return value


def _task_error(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, dict)) and value:
        return "upstream_error"
    raise EvaluationRecordsError("native task error must be nonempty text or an object")


def _tags(row: dict[str, Any]) -> dict[str, str]:
    # Only string metadata is exposed as slice tags; the original file digest
    # binds other upstream fields, which are not decision inputs.
    metadata = _native_object(row.get("metadata"), "metadata")
    if not isinstance(metadata, dict):
        raise EvaluationRecordsError("native metadata must be an object")
    return {
        k: v
        for k, v in metadata.items()
        if isinstance(v, str) and not k.startswith("invarlock_")
    }


def _with_capture_facts(
    record: dict[str, Any], row: dict[str, Any], metadata: Any
) -> dict[str, Any]:
    """Retain original fields and explicitly supplied continuation measurements."""
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise EvaluationRecordsError("native metadata must be an object")
    record["context"] = {**record.get("context", {}), "upstream_record": row}
    record["scores"] = merge_capture_scores(record.get("scores", {}), metadata)
    if "invarlock_likelihood" in metadata:
        record["likelihood"] = metadata["invarlock_likelihood"]
    return record


def _inspect(
    value: Any, *, allow_native_score_values: bool = False
) -> list[dict[str, Any]]:
    if (
        not isinstance(value, dict)
        or type(value.get("version")) is not int
        or value["version"] not in (1, 2)
    ):
        raise EvaluationRecordsError(
            "inspect-json requires a version 1 or 2 JSON EvalLog (export .eval logs as JSON)"
        )
    if value.get("status") not in ("success", "error", "cancelled"):
        raise EvaluationRecordsError("Inspect log lacks a recognized completion status")
    if value["status"] != "success":
        raise EvaluationRecordsError(
            "Inspect run did not complete successfully; supply a complete run"
        )
    result = []
    for row in _rows(value.get("samples"), "Inspect samples"):
        output = _native_object(row.get("output"), "output")
        choices = output.get("choices", [])
        if not isinstance(choices, list):
            raise EvaluationRecordsError("Inspect choices must be an array")
        task_error = _task_error(row.get("error"))
        metadata = _native_object(row.get("metadata"), "metadata")
        likelihood_only = (
            isinstance(metadata, dict) and "invarlock_likelihood" in metadata
        )
        if len(choices) > 1 or (
            not choices and task_error is None and not likelihood_only
        ):
            raise EvaluationRecordsError(
                "Inspect import requires one completion per sample"
            )
        content = choices[0]["message"]["content"] if choices else None
        if isinstance(content, list):
            if any(part.get("type") != "text" for part in content):
                raise EvaluationRecordsError(
                    "Inspect completion contains unsupported non-text content"
                )
            content = "".join(part["text"] for part in content)
        target = row.get("target")
        if isinstance(target, list):
            if len(target) != 1:
                raise EvaluationRecordsError(
                    "multiple Inspect targets require an explicit scoring adapter"
                )
            target = target[0]
        record_id = _identifier(row["id"])
        if type(row.get("epoch", 1)) is not int or row.get("epoch", 1) != 1:
            raise EvaluationRecordsError(
                "multiple Inspect epochs require an explicit paired trial mapping"
            )
        result.append(
            _with_capture_facts(
                {
                    "id": record_id,
                    "input": row["input"],
                    "expected": target,
                    "output": content,
                    "scores": _scores(
                        _native_object(row.get("scores"), "scores"),
                        allow_native_values=allow_native_score_values,
                    ),
                    "metadata": _tags(row),
                    "error": task_error,
                },
                row,
                metadata,
            )
        )
    return result


def _identifier(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise EvaluationRecordsError("native record IDs must be strings or integers")
    return str(value)


def _harness(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if any(
            key not in row for key in ("doc", "target", "arguments", "filtered_resps")
        ):
            raise EvaluationRecordsError(
                "LM Eval requires --log_samples generation records with doc, target, arguments and filtered_resps"
            )
        scores = {
            key: value
            for key, value in row.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and key not in ("doc_id",)
        }
        metadata = _native_object(row.get("metadata"), "metadata")
        facts = (
            metadata.get("invarlock_likelihood") if isinstance(metadata, dict) else None
        )
        response = row["filtered_resps"]
        task_error = _task_error(row.get("error"))
        if task_error is not None and response in (None, [], [None]):
            output = None
        elif facts is not None and response == [None]:
            output = None
        elif (
            isinstance(facts, dict)
            and isinstance(response, list)
            and len(response) == 1
            and isinstance(response[0], list)
            and len(response[0]) == 2
        ):
            measured, greedy = response[0]
            arguments = row["arguments"]
            # EvaluationTracker.save_results_samples names request/argument
            # slots when writing JSONL. Only one exact two-argument request
            # establishes the same continuation binding as the in-memory list.
            if isinstance(arguments, dict) and set(arguments) == {"gen_args_0"}:
                request = arguments["gen_args_0"]
                if isinstance(request, dict) and set(request) == {"arg_0", "arg_1"}:
                    arguments = [[request["arg_0"], request["arg_1"]]]
                    # The same logger's sanitize_list encodes tuple values as
                    # strings. Accept only its exact spelling of independently
                    # supplied numeric likelihood and boolean greedy flag.
                    supplied = facts.get("logprob_sum")
                    if (
                        type(supplied) in (int, float)
                        and isinstance(measured, str)
                        and measured == str(supplied)
                        and greedy in ("True", "False")
                    ):
                        measured, greedy = supplied, greedy == "True"
            if (
                type(measured) not in (int, float)
                or type(greedy) is not bool
                or measured != facts.get("logprob_sum")
                or not isinstance(arguments, list)
                or len(arguments) != 1
                or not isinstance(arguments[0], list)
                or len(arguments[0]) != 2
                or not isinstance(arguments[0][0], str)
                or arguments[0][1] != row["target"]
            ):
                raise EvaluationRecordsError(
                    "Harness likelihood facts differ from the native result or continuation"
                )
            output = None
        else:
            output = _scalar_output(response)
        result.append(
            _with_capture_facts(
                {
                    "id": _identifier(row["doc_id"]),
                    "input": row["doc"],
                    "context": {"arguments": row["arguments"]},
                    "expected": row["target"],
                    "output": output,
                    "scores": scores,
                    "metadata": _tags(row),
                    "error": task_error,
                },
                row,
                metadata,
            )
        )
    return result


def _promptfoo_error(row: dict[str, Any]) -> str | None:
    """Distinguish Promptfoo's ASSERT failure from provider or grading ERROR."""
    response = _native_object(row.get("response"), "response")
    error = row.get("error") or response.get("error")
    if "failureReason" not in row:
        # Older exports without typed reasons retain conservative error handling.
        return "upstream_error" if error else None
    reason = row["failureReason"]
    if type(reason) is not int or reason not in (0, 1, 2):
        raise EvaluationRecordsError(
            "Promptfoo failureReason is unsupported or ambiguous"
        )
    if reason == 2:
        if row.get("success") is not False:
            raise EvaluationRecordsError(
                "Promptfoo runtime failure contradicts success"
            )
        return "upstream_error"
    grading = row.get("gradingResult")
    if reason == 0:
        if (
            error
            or row.get("success") is False
            or (isinstance(grading, dict) and grading.get("pass") is False)
        ):
            raise EvaluationRecordsError(
                "Promptfoo failureReason contradicts failure fields"
            )
        return None
    if (
        row.get("success") is not False
        or not isinstance(grading, dict)
        or grading.get("pass") is not False
        or "output" not in response
        or response.get("error")
        or type(row.get("score")) not in (int, float)
        or type(grading.get("score")) not in (int, float)
        or row["score"] != grading["score"]
        or not isinstance(grading.get("reason"), str)
        or row.get("error") != grading["reason"]
    ):
        raise EvaluationRecordsError(
            "Promptfoo assertion failure fields are inconsistent"
        )
    # Native applyGradingResult uses row.error for an ordinary wrong answer.
    # Preserve that answer so deterministic metrics can still evaluate it.
    return None


def _promptfoo(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        case = row.get("testCase")
        if not isinstance(case, dict) or "vars" not in case or "prompt" not in row:
            raise EvaluationRecordsError(
                "Promptfoo rows need testCase and prompt for pairing; use the full export or SDK capture"
            )
        # Never infer a reference answer from an assertion or successful judgment.
        metadata = _native_object(case.get("metadata"), "metadata")
        response = _native_object(row.get("response"), "response")
        expected = metadata.get("invarlock_expected")
        stable_id = metadata.get("invarlock_id")
        if stable_id is not None and (
            not isinstance(stable_id, str) or not stable_id.strip()
        ):
            raise EvaluationRecordsError(
                "Promptfoo invarlock_id must be a nonempty string"
            )
        prompt = row["prompt"]
        prompt = prompt.get("raw") if isinstance(prompt, dict) else prompt
        if not isinstance(prompt, str):
            raise EvaluationRecordsError(
                "Promptfoo row requires the actual rendered prompt"
            )
        result.append(
            _with_capture_facts(
                {
                    "id": stable_id
                    if stable_id is not None
                    else _identifier(row["testIdx"])
                    + ":"
                    + _identifier(row["promptIdx"]),
                    "input": case["vars"],
                    "context": {"prompt": prompt},
                    "expected": expected,
                    "output": response.get("output"),
                    "scores": _scores(
                        {
                            k: row[k]
                            for k in ("score", "latencyMs", "cost")
                            if row.get(k) is not None
                        }
                    ),
                    "metadata": _tags(case),
                    "error": _promptfoo_error(row),
                },
                row,
                case.get("metadata"),
            )
        )
    return result


def load_run(
    path: str | Path,
    *,
    adapter: str = "invarlock",
    source: Mapping[str, str] | None = None,
    run_id: str | None = None,
    artifact_digest: str | None = None,
    service_identity: Mapping[str, Any] | None = None,
    score_provenance: Mapping[str, Any] | None = None,
    input_projection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Import a native export or a canonical run with explicit source identities."""
    from invarlock.captured_contracts import read_file

    try:
        raw = read_file(Path(path), MAX_INPUT_BYTES)
    except (OSError, ValueError) as exc:
        raise EvaluationRecordsError(f"cannot import {adapter}: {exc}") from exc
    return _parse_run_bytes(
        raw,
        adapter=adapter,
        source=source,
        run_id=run_id,
        artifact_digest=artifact_digest,
        service_identity=service_identity,
        score_provenance=score_provenance,
        input_projection=input_projection,
    )


def write_evaluator_export(
    raw: bytes,
    destination: str | Path,
    *,
    expected_ids: list[str],
    adapter: str = "invarlock",
    source: Mapping[str, str] | None = None,
    run_id: str | None = None,
    artifact_digest: str | None = None,
    service_identity: Mapping[str, Any] | None = None,
    score_provenance: Mapping[str, Any] | None = None,
    input_projection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a complete export, then write its exact bytes to a new file.

    Obtain expected IDs independently before execution. This checks capture
    completeness, not model execution or qualification of an upstream scorer.
    The returned canonical run has the same identity as a subsequent load_run.
    """
    if (
        not isinstance(expected_ids, list)
        or not 1 <= len(expected_ids) <= MAX_RECORDS
        or any(
            not isinstance(value, str) or not value.strip() for value in expected_ids
        )
        or len(set(expected_ids)) != len(expected_ids)
    ):
        raise EvaluationRecordsError(
            "expected_ids must be a bounded nonempty list of unique string IDs"
        )
    run = _parse_run_bytes(
        raw,
        adapter=adapter,
        source=source,
        run_id=run_id,
        artifact_digest=artifact_digest,
        service_identity=service_identity,
        score_provenance=score_provenance,
        input_projection=input_projection,
    )
    if {row["id"] for row in run["records"]} != set(expected_ids):
        raise EvaluationRecordsError(
            "export does not contain exactly the expected case IDs"
        )
    with Path(destination).open("xb") as stream:
        stream.write(raw)
    return run


def _parse_run_bytes(
    raw: bytes,
    *,
    adapter: str = "invarlock",
    source: Mapping[str, str] | None = None,
    run_id: str | None = None,
    artifact_digest: str | None = None,
    service_identity: Mapping[str, Any] | None = None,
    score_provenance: Mapping[str, Any] | None = None,
    input_projection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse secured immutable bytes without reopening the source path."""
    if adapter not in ADAPTERS:
        raise EvaluationRecordsError(
            f"unsupported adapter {adapter!r}; choose {', '.join(ADAPTERS)}"
        )
    try:
        if not isinstance(raw, bytes) or len(raw) > MAX_INPUT_BYTES:
            raise EvaluationRecordsError("evaluation export exceeds its byte limit")
        if adapter in ("jsonl", "lm-eval-samples", "promptfoo-jsonl"):
            rows = _jsonl_rows(raw, adapter)
            records = (
                _harness(rows)
                if adapter == "lm-eval-samples"
                else _promptfoo(rows)
                if adapter == "promptfoo-jsonl"
                else rows
            )
        else:
            value = parse_json_bytes(raw, label="evaluation export")
            if adapter == "invarlock":
                from invarlock.evaluation_comparison.comparison import _check_run

                _check_run(value)
                if any(
                    v is not None
                    for v in (
                        source,
                        run_id,
                        artifact_digest,
                        service_identity,
                        score_provenance,
                        input_projection,
                    )
                ):
                    raise EvaluationRecordsError(
                        "canonical run identities cannot be overridden at import"
                    )
                return cast(dict[str, Any], value)
            if adapter in ("evaluator-json", "evaluator-native-json"):
                from invarlock.evaluation_records.integrations import (
                    FORMAT,
                    parse_evaluator_export,
                )

                if adapter == "evaluator-native-json":
                    if source is None or run_id is None:
                        raise EvaluationRecordsError(
                            "native evaluator JSON requires source name/version and run_id"
                        )
                    if source["name"] == "langfuse":
                        value = {
                            "format": "invarlock/langfuse-export-v1",
                            "sdk_version": source["version"],
                            "result": value,
                        }
                    value = {
                        "format": FORMAT,
                        "evaluator": source["name"],
                        "source_version": source["version"],
                        "run_id": run_id,
                        "payload": value,
                    }
                records = parse_evaluator_export(value, source=source, run_id=run_id)
            elif adapter == "langfuse-json":
                from invarlock.evaluation_records.langfuse import parse_langfuse_export

                records = parse_langfuse_export(value, source=source, run_id=run_id)
            else:
                records = _inspect(value)
        if (
            source is None
            or run_id is None
            or (artifact_digest is None and service_identity is None)
        ):
            raise EvaluationRecordsError(
                "native import requires source name/version, run_id and artifact_digest or service_identity from your pipeline"
            )
        return capture_evaluator_run(
            records,
            source=dict(source),
            run_id=run_id,
            artifact_digest=artifact_digest,
            service_identity=service_identity,
            score_provenance=dict(score_provenance)
            if score_provenance is not None
            else None,
            source_digest="sha256:" + hashlib.sha256(raw).hexdigest(),
            input_projection=input_projection,
        )
    except (
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        AttributeError,
        OSError,
        OverflowError,
        RecursionError,
    ) as exc:
        if isinstance(exc, EvaluationRecordsError):
            raise
        raise EvaluationRecordsError(f"cannot import {adapter}: {exc}") from exc
