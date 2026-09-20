"""Dedicated captures of evaluator reports, prediction tables, and attempt logs.

These adapters import recorded execution data only. Aggregate metrics remain
provenance and are never replicated into per-case scores. Optional ``columns``
and ``score_columns`` name actual table columns, without aligning rows by order.
SDK imports are unnecessary: supported model/dataclass/table serializers are
used explicitly and every resulting value must have a lossless JSON form.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, cast
from uuid import UUID

from invarlock.evaluation_record_contracts.contracts import (
    MAX_RECORDS,
    EvaluationRecordsError,
)
from invarlock.evaluation_records.capture_facts import merge_capture_scores

EVALUATORS = frozenset(
    {
        "pydantic-evals",
        "azure-ai-evaluation",
        "evidently",
        "mlflow",
        "garak",
        "openai-evals",
        "trulens",
    }
)


def _json(value: Any, depth: int = 0) -> Any:
    if depth > 64:
        raise EvaluationRecordsError("evaluator value exceeds maximum nesting")

    def convert(item: Any) -> Any:
        return _json(item, depth + 1)

    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise EvaluationRecordsError("evaluator values must be finite")
        return value
    if isinstance(value, Enum):
        return convert(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise EvaluationRecordsError("evaluator objects require string keys")
        return {key: convert(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_RECORDS:
            raise EvaluationRecordsError("evaluator array exceeds record limit")
        return [convert(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: convert(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if type(value).__module__.startswith("trulens.") and isinstance(
        getattr(type(value), "model_fields", None), dict
    ):
        # TruLens overrides model_dump with a JSON utility whose kwargs differ
        # from Pydantic's. Read its declared serializable fields directly; its
        # excluded runtime futures are deliberately not evidence.
        return {
            name: convert(getattr(value, name))
            for name, field in type(value).model_fields.items()
            if field.exclude is not True
        }
    if callable(getattr(value, "model_dump", None)):
        return convert(value.model_dump(mode="python"))
    # Scalar numpy values occur in pandas tables; preserve their scalar type.
    if type(value).__module__.startswith("numpy") and callable(
        getattr(value, "item", None)
    ):
        return convert(value.item())
    raise EvaluationRecordsError(
        f"unsupported evaluator value type: {type(value).__name__}"
    )


def _obj(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationRecordsError(f"{label} requires an object")
    return value


def _array(value: Any, label: str, *, empty: bool = False) -> list[Any]:
    if (
        not isinstance(value, list)
        or len(value) > MAX_RECORDS
        or (not value and not empty)
    ):
        raise EvaluationRecordsError(
            f"{label} requires a bounded {'nonempty ' if not empty else ''}array"
        )
    return value


def _id(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise EvaluationRecordsError(
            "evaluator row requires an independent nonempty string ID"
        )
    return value


def _number(value: Any, label: str) -> float:
    try:
        valid = type(value) in (int, float, bool) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise EvaluationRecordsError(
            f"{label} requires a finite numeric or boolean score"
        )
    return float(value)


def _scores(value: Any) -> dict[str, float]:
    return {_id(key): _number(item, key) for key, item in _obj(value, "scores").items()}


def _error(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        for key in ("message", "error_message", "msg"):
            if isinstance(value.get(key), str) and value[key]:
                return cast(str, value[key])
    raise EvaluationRecordsError("native error requires a nonempty message")


def _record(
    native: dict[str, Any],
    *,
    ident: Any,
    input_value: Any,
    output: Any,
    expected: Any = None,
    scores: dict[str, float] | None = None,
    error: Any = None,
    summary: Any = None,
) -> dict[str, Any]:
    metadata = native.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise EvaluationRecordsError("native metadata must be an object or null")
    record = {
        "id": _id(ident),
        "input": input_value,
        "expected": expected,
        "output": output,
        "scores": dict(scores) if scores is not None else {},
        "error": _error(error),
        "metadata": {
            key: value
            for key, value in metadata.items()
            if isinstance(value, str) and not key.startswith("invarlock_")
        }
        if isinstance(metadata, dict)
        else {},
        "context": {"upstream_record": native},
    }
    record["scores"] = merge_capture_scores(
        record["scores"], metadata if isinstance(metadata, dict) else {}
    )
    if summary:
        record["context"]["upstream_summary"] = summary
    for owner in (native, metadata if isinstance(metadata, dict) else {}):
        for key in ("likelihood", "invarlock_likelihood"):
            if key in owner:
                if "likelihood" in record and record["likelihood"] != owner[key]:
                    raise EvaluationRecordsError(
                        "conflicting explicit likelihood evidence"
                    )
                record["likelihood"] = owner[key]
    return record


def _table(value: Any) -> list[dict[str, Any]]:
    if callable(getattr(value, "as_dataframe", None)):
        value = value.as_dataframe()
    if callable(getattr(value, "to_dict", None)) and not isinstance(value, Mapping):
        value = value.to_dict(orient="records")
    value = _json(value)
    if isinstance(value, dict) and set(value) == {"columns", "data"}:
        columns = _array(value["columns"], "table columns")
        if any(not isinstance(key, str) for key in columns) or len(columns) != len(
            set(columns)
        ):
            raise EvaluationRecordsError("table columns require unique names")
        rows = _array(value["data"], "table data")
        if any(not isinstance(row, list) or len(row) != len(columns) for row in rows):
            raise EvaluationRecordsError("table row and column lengths differ")
        value = [dict(zip(columns, row, strict=True)) for row in rows]
    return [_obj(row, "table row") for row in _array(value, "table rows")]


def _pydantic(value: Any) -> list[dict[str, Any]]:
    report = _obj(_json(value), "Pydantic Evals report")
    cases = _array(report.get("cases"), "Pydantic Evals cases", empty=True)
    failures = _array(report.get("failures", []), "Pydantic Evals failures", empty=True)
    summary = {
        key: item for key, item in report.items() if key not in {"cases", "failures"}
    }
    records = []
    for raw, failed in [(row, False) for row in cases] + [
        (row, True) for row in failures
    ]:
        row = _obj(raw, "Pydantic Evals case")
        if "inputs" not in row or (not failed and "output" not in row):
            raise EvaluationRecordsError(
                "Pydantic Evals case lacks inputs or actual output"
            )
        scores = {}
        for group in ("assertions", "scores"):
            for name, metric in _obj(row.get(group, {}), group).items():
                if name in scores:
                    raise EvaluationRecordsError("Pydantic Evals score names collide")
                scores[_id(name)] = _number(
                    metric.get("value") if isinstance(metric, dict) else metric, name
                )
        error = row.get("error_message") if failed else row.get("error")
        evaluator_failures = row.get("evaluator_failures", [])
        _array(evaluator_failures, "Pydantic Evals evaluator failures", empty=True)
        if failed and error is None:
            raise EvaluationRecordsError(
                "Pydantic Evals failed case lacks error_message"
            )
        records.append(
            _record(
                row,
                ident=row.get("name"),
                input_value=row["inputs"],
                expected=row.get("expected_output"),
                output=row.get("output"),
                scores=scores,
                error=error,
                summary=summary,
            )
        )
    return records


def _azure(value: Any) -> list[dict[str, Any]]:
    report = _obj(_json(value), "Azure AI Evaluation result")
    summary = {key: item for key, item in report.items() if key != "rows"}
    records = []
    for row in _table(report.get("rows")):
        if "inputs.metadata" in row:
            if "metadata" in row and row["metadata"] != row["inputs.metadata"]:
                raise EvaluationRecordsError(
                    "Azure capture metadata conflicts with inputs.metadata"
                )
            row = {**row, "metadata": row["inputs.metadata"]}
        if (
            "inputs.response" not in row
            and "outputs.response" not in row
            and not any(
                item is not None
                for key, item in row.items()
                if key == "error" or key.endswith(".error")
            )
        ):
            raise EvaluationRecordsError("Azure row lacks captured response")
        scores = {
            key.removeprefix("outputs."): _number(item, key)
            for key, item in row.items()
            if key.startswith("outputs.")
            and key != "outputs.response"
            and type(item) in (int, float, bool)
        }
        # Evaluator errors do not establish a failed model invocation. They
        # remain in upstream_record; only the captured task error is promoted.
        errors = [row["error"]] if row.get("error") is not None else []
        error = (
            "; ".join(cast(str, _error(item)) for item in errors) if errors else None
        )
        records.append(
            _record(
                row,
                ident=row.get("inputs.record_id", row.get("inputs.id")),
                input_value=row.get("inputs.query", row.get("inputs.input")),
                expected=row.get("inputs.ground_truth"),
                output=row.get("outputs.response", row.get("inputs.response")),
                scores=scores,
                error=error,
                summary=summary,
            )
        )
    return records


def _columns(report: dict[str, Any], defaults: dict[str, str]) -> dict[str, str]:
    columns = _obj(report.get("columns", {}), "column mapping")
    if columns.keys() - defaults.keys() or any(
        not isinstance(value, str) or not value for value in columns.values()
    ):
        raise EvaluationRecordsError("unsupported table column mapping")
    return {**defaults, **columns}


def _scored_table(value: Any, evaluator: str) -> list[dict[str, Any]]:
    report = _obj(value, "scored table export")
    if evaluator == "mlflow":
        source = report.get("prediction_table", report.get("rows"))
        if source is None:
            source = _obj(report.get("tables", {}), "MLflow tables").get(
                "eval_results_table"
            )
        defaults = {
            "id": "record_id",
            "input": "input",
            "expected": "target",
            "output": "prediction",
        }
    else:
        source = report.get("dataset", report.get("rows"))
        defaults = {
            "id": "record_id",
            "input": "input",
            "expected": "reference",
            "output": "output",
        }
    rows = _table(source)
    columns = _columns(report, defaults)
    names = report.get("score_columns")
    if names is not None:
        _array(names, "score_columns")
        if any(not isinstance(name, str) or not name for name in names) or len(
            names
        ) != len(set(names)):
            raise EvaluationRecordsError("score_columns requires unique names")
    summary = _json(
        {
            key: item
            for key, item in report.items()
            if key not in {"rows", "dataset", "tables", "prediction_table"}
        }
    )
    records = []
    for row in rows:
        output_key = columns["output"]
        # MLflow's standard LLM table uses plural predictions and targets.
        if "output" not in report.get("columns", {}) and evaluator == "mlflow":
            output_key = "predictions" if "predictions" in row else output_key
        expected_key = columns["expected"]
        if "expected" not in report.get("columns", {}) and evaluator == "mlflow":
            expected_key = "targets" if "targets" in row else expected_key
        if output_key not in row and not row.get("error"):
            raise EvaluationRecordsError(
                f"{evaluator} table lacks actual model output column"
            )
        score_names = (
            names
            if names is not None
            else [
                key
                for key in row
                if (evaluator == "mlflow" and key.endswith("/score"))
                or (evaluator == "evidently" and key == "exact_match")
            ]
        )
        scores = {}
        for name in score_names:
            if name not in row:
                raise EvaluationRecordsError(f"missing score column: {name}")
            if row[name] is not None:
                scores[name] = _number(row[name], name)
        records.append(
            _record(
                row,
                ident=row.get(columns["id"]),
                input_value=row.get(columns["input"]),
                expected=row.get(expected_key),
                output=row.get(output_key),
                scores=scores,
                error=row.get("error"),
                summary=summary,
            )
        )
    return records


def _garak_text(value: Any, *, prompt: bool = False) -> Any:
    """Select the SDK's unambiguous text field without flattening conversations."""
    if prompt and isinstance(value, dict) and set(value) <= {"turns", "notes"}:
        turns = value.get("turns")
        if (
            isinstance(turns, list)
            and len(turns) == 1
            and isinstance(turns[0], dict)
            and turns[0].get("role") == "user"
            and set(turns[0]) <= {"role", "content"}
        ):
            selected = _garak_text(turns[0].get("content"))
            if isinstance(selected, str):
                return selected
        return value
    if (
        isinstance(value, dict)
        and "text" in value
        and set(value)
        <= {"text", "lang", "data_path", "data_type", "data_checksum", "notes"}
        and isinstance(value["text"], str)
        and all(
            value.get(key) is None
            for key in ("data_path", "data_type", "data_checksum")
        )
    ):
        return value["text"]
    return value


def _garak(value: Any) -> list[dict[str, Any]]:
    value = _obj(value, "Garak export")
    entries = value.get("attempts", value.get("entries"))
    summary = {
        key: item
        for key, item in value.items()
        if key not in {"attempts", "entries", "source_cases"}
    }
    entries = _array(entries, "Garak attempts")
    histories: dict[str, list[dict[str, Any]]] = {}
    other = []
    for entry in entries:
        row = _obj(_json(entry), "Garak entry")
        if row.get("entry_type", "attempt") != "attempt":
            if row.get("entry_type") not in {
                "eval",
                "init",
                "start_run",
                "config",
                "completion",
            }:
                raise EvaluationRecordsError("unsupported Garak report entry")
            other.append(row)
            continue
        ident = _id(row.get("uuid"))
        status = row.get("status")
        if type(status) is not int or status not in (0, 1, 2):
            raise EvaluationRecordsError("Garak attempt has invalid status")
        history = histories.setdefault(ident, [])
        if history and (
            history[-1]["status"] >= status
            or history[-1].get("prompt") != row.get("prompt")
        ):
            raise EvaluationRecordsError(
                "Garak attempt history is ambiguous or conflicts"
            )
        history.append(row)
    if other:
        summary["report_entries"] = other
    records = []
    for ident, history in histories.items():
        row = history[-1]
        if "prompt" not in row:
            raise EvaluationRecordsError("Garak attempt lacks prompt")
        outputs = _array(row.get("outputs"), "Garak outputs", empty=True)
        detectors = _obj(row.get("detector_results", {}), "Garak detector results")
        for name, values in detectors.items():
            if len(_array(values, name, empty=True)) != len(outputs):
                raise EvaluationRecordsError(
                    "Garak detector scores and outputs are misaligned"
                )
        for index in range(max(1, len(outputs))):
            output = _garak_text(outputs[index]) if outputs else None
            error = row.get("error")
            likelihood_only = any(
                key in row for key in ("likelihood", "invarlock_likelihood")
            ) or (
                isinstance(row.get("metadata"), dict)
                and any(
                    key in row["metadata"]
                    for key in ("likelihood", "invarlock_likelihood")
                )
            )
            if error is None and (
                row["status"] != 2 or (output is None and not likelihood_only)
            ):
                error = (
                    "Garak attempt incomplete"
                    if row["status"] != 2
                    else "Garak generation returned no output"
                )
            scores = {
                name: _number(values[index], name)
                for name, values in detectors.items()
                if outputs and values[index] is not None
            }
            native = {**row, "attempt_history": history, "generation_index": index}
            records.append(
                _record(
                    native,
                    ident=f"{ident}:{index}",
                    input_value=_garak_text(row["prompt"], prompt=True),
                    output=output,
                    scores=scores,
                    error=error,
                    summary=summary,
                )
            )
    if isinstance(value, dict) and "source_cases" in value:
        sources = _array(value["source_cases"], "Garak source_cases")
        by_native = {}
        for raw in sources:
            case = _obj(raw, "Garak source case")
            native_id = _id(case.get("native_id"))
            _id(case.get("id"))
            if "input" not in case or "expected" not in case or native_id in by_native:
                raise EvaluationRecordsError(
                    "Garak source cases require unique native_id, input and expected"
                )
            by_native[native_id] = case
        if set(by_native) != {row["id"] for row in records}:
            raise EvaluationRecordsError(
                "Garak source case membership differs from attempt generations"
            )
        for record in records:
            case = by_native[record["id"]]
            if case["input"] != record["input"] or (
                "output" in case and case["output"] != record["output"]
            ):
                raise EvaluationRecordsError(
                    "Garak source case conflicts with actual attempt input/output"
                )
            captured = _record(
                case,
                ident=case["id"],
                input_value=record["input"],
                output=record["output"],
                expected=case["expected"],
                scores=record["scores"],
                error=record["error"],
            )
            native_attempt = record["context"]["upstream_record"]
            if (
                "likelihood" in captured
                and record["output"] is None
                and native_attempt["status"] == 2
                and native_attempt.get("error") is None
            ):
                captured["error"] = None
            if (
                "likelihood" in captured
                and "likelihood" in record
                and captured["likelihood"] != record["likelihood"]
            ):
                raise EvaluationRecordsError(
                    "Garak source likelihood conflicts with attempt"
                )
            record.update(
                {key: item for key, item in captured.items() if key != "context"}
            )
            record["context"]["source_case"] = case
    return records


def _openai(value: Any) -> list[dict[str, Any]]:
    report = value if isinstance(value, dict) else {"events": value}
    report = _obj(_json(report), "OpenAI Evals log")
    events = _array(report.get("events"), "OpenAI Evals events")
    summary = {key: item for key, item in report.items() if key != "events"}
    grouped: dict[str, list[dict[str, Any]]] = {}
    run_ids = set()
    event_ids = set()
    for raw in events:
        event = _obj(raw, "OpenAI Evals event")
        if (
            ("spec" in event or "final_report" in event)
            and "type" not in event
            and "sample_id" not in event
        ):
            summary.setdefault("log_entries", []).append(event)
            continue
        if event.get("type") == "final_report" and event.get("sample_id") is None:
            summary.setdefault("log_entries", []).append(event)
            continue
        ident = _id(event.get("sample_id"))
        _obj(event.get("data"), "OpenAI Evals event data")
        if "run_id" in event:
            run_ids.add(_id(event["run_id"]))
        if "event_id" in event:
            event_id = event["event_id"]
            if type(event_id) is not int or event_id < 0 or event_id in event_ids:
                raise EvaluationRecordsError(
                    "OpenAI Evals duplicate or invalid event ID"
                )
            event_ids.add(event_id)
        grouped.setdefault(ident, []).append(event)
    if len(run_ids) > 1:
        raise EvaluationRecordsError("OpenAI Evals log contains multiple runs")
    records = []
    for ident, case_events in grouped.items():
        samplings = [
            event["data"] for event in case_events if event.get("type") == "sampling"
        ]
        matches = [
            event["data"] for event in case_events if event.get("type") == "match"
        ]
        errors = [
            event["data"] for event in case_events if event.get("type") == "error"
        ]
        if len(samplings) > 1 or len(matches) > 1:
            raise EvaluationRecordsError(
                "OpenAI Evals sample has ambiguous multiple sampling or match events"
            )
        sampling = samplings[0] if samplings else {}
        match = matches[0] if matches else {}
        if (
            "sampled" in sampling
            and "sampled" in match
            and sampling["sampled"] != match["sampled"]
        ):
            raise EvaluationRecordsError("OpenAI Evals sampled outputs conflict")
        if "sampled" not in sampling and "sampled" not in match and not errors:
            raise EvaluationRecordsError(
                "OpenAI Evals sample lacks actual sampled output"
            )
        scores = {}
        if matches:
            if type(match.get("correct")) is not bool:
                raise EvaluationRecordsError(
                    "OpenAI Evals match requires boolean correct"
                )
            scores["match"] = float(match["correct"])
        for event in case_events:
            if event.get("type") == "metrics":
                for name, metric in _scores(event["data"]).items():
                    if name in scores:
                        raise EvaluationRecordsError(
                            "OpenAI Evals duplicate metric name"
                        )
                    scores[name] = metric
        native: dict[str, Any] = {"sample_id": ident, "events": case_events}
        for event in case_events:
            if "metadata" in event["data"]:
                metadata = event["data"]["metadata"]
                if metadata is not None:
                    _obj(metadata, "OpenAI Evals capture metadata")
                if "metadata" in native and native["metadata"] != metadata:
                    raise EvaluationRecordsError(
                        "OpenAI Evals conflicting capture metadata"
                    )
                native["metadata"] = metadata
        # Only explicitly complete likelihood evidence is promoted; cond_logp remains context.
        for event in case_events:
            if "likelihood" in event["data"]:
                if (
                    "likelihood" in native
                    and native["likelihood"] != event["data"]["likelihood"]
                ):
                    raise EvaluationRecordsError(
                        "OpenAI Evals conflicting likelihood evidence"
                    )
                native["likelihood"] = event["data"]["likelihood"]
        records.append(
            _record(
                native,
                ident=ident,
                input_value=sampling.get("prompt", match.get("prompt")),
                expected=match.get("expected"),
                output=sampling.get("sampled", match.get("sampled")),
                scores=scores,
                error="; ".join(cast(str, _error(item)) for item in errors)
                if errors
                else None,
                summary=summary,
            )
        )
    return records


def _trulens(value: Any) -> list[dict[str, Any]]:
    report = _obj(value, "TruLens record export")
    rows = _table(report.get("records"))
    feedback_names = _json(report.get("feedback_columns", []))
    _array(feedback_names, "TruLens feedback_columns", empty=True)
    if any(not isinstance(name, str) or not name for name in feedback_names) or len(
        set(feedback_names)
    ) != len(feedback_names):
        raise EvaluationRecordsError("TruLens feedback columns require unique names")
    external_feedback: dict[str, list[dict[str, Any]]] = {}
    if "feedback_results" in report:
        for raw in _array(
            _json(report["feedback_results"]), "TruLens supplied feedback", empty=True
        ):
            feedback = _obj(raw, "TruLens supplied feedback result")
            external_feedback.setdefault(_id(feedback.get("record_id")), []).append(
                feedback
            )
        if external_feedback.keys() - {row.get("record_id") for row in rows}:
            raise EvaluationRecordsError(
                "TruLens supplied feedback has unknown record IDs"
            )
    records = []
    for row in rows:
        row = dict(row)
        if "meta" in row:
            if "metadata" in row and row["metadata"] != row["meta"]:
                raise EvaluationRecordsError(
                    "TruLens meta conflicts with supplied metadata"
                )
            row["metadata"] = row["meta"]
        if row.get("record_id") in external_feedback:
            existing = row.get("feedback_results")
            row["feedback_results"] = (
                _array(
                    existing if existing is not None else [],
                    "TruLens feedback results",
                    empty=True,
                )
                + external_feedback[row["record_id"]]
            )
        model_record = "main_input" in row or "main_output" in row
        input_key, output_key = (
            ("main_input", "main_output") if model_record else ("input", "output")
        )
        error = row.get("main_error", row.get("error"))
        if input_key not in row or (output_key not in row and error is None):
            raise EvaluationRecordsError("TruLens record lacks captured input/output")
        scores = {}
        for name in feedback_names:
            if name not in row:
                raise EvaluationRecordsError(f"TruLens feedback column missing: {name}")
            if row[name] is not None:
                scores[name] = _number(row[name], name)
        feedbacks = _array(
            row["feedback_results"] if row.get("feedback_results") is not None else [],
            "TruLens feedback results",
            empty=True,
        )
        for raw in feedbacks:
            feedback = _obj(raw, "TruLens feedback result")
            if feedback.get("record_id", row.get("record_id")) != row.get("record_id"):
                raise EvaluationRecordsError(
                    "TruLens feedback belongs to another record"
                )
            name = _id(feedback.get("name"))
            if name in scores:
                raise EvaluationRecordsError("TruLens duplicate feedback name")
            if feedback.get("result") is not None:
                scores[name] = _number(feedback["result"], name)
            if feedback.get("error") is not None:
                _error(feedback["error"])
        records.append(
            _record(
                row,
                ident=row.get("record_id"),
                input_value=row.get(input_key),
                output=row.get(output_key),
                expected=row.get("ground_truth"),
                scores=scores,
                error=error,
            )
        )
    return records


def serialize_results(evaluator: str, results: object) -> Any:
    """Detach supported SDK reports into the native JSON payload accepted below."""
    if evaluator not in EVALUATORS:
        raise EvaluationRecordsError(f"unsupported batch evaluator: {evaluator}")
    if evaluator in {"evidently", "mlflow"}:
        if evaluator == "mlflow" and not isinstance(results, (dict, list)):
            results = {
                "metrics": getattr(results, "metrics", None),
                "tables": getattr(results, "tables", None),
            }
        if not isinstance(results, dict):
            results = {"rows": results}
        report = dict(results)
        for key in ("rows", "dataset", "prediction_table"):
            if key in report:
                report[key] = _table(report[key])
        if "tables" in report:
            report["tables"] = {
                key: _table(table)
                for key, table in _obj(report["tables"], "MLflow tables").items()
            }
        return _json(report)
    if evaluator == "garak":
        report = dict(results) if isinstance(results, dict) else {"attempts": results}
        key = "attempts" if "attempts" in report else "entries"
        report[key] = [
            entry.as_dict() if callable(getattr(entry, "as_dict", None)) else entry
            for entry in _array(report.get(key), "Garak attempts")
        ]
        return _json(report)
    if evaluator == "trulens":
        if isinstance(results, tuple) and len(results) == 2:
            results = {"records": results[0], "feedback_columns": results[1]}
        report = dict(_obj(results, "TruLens export"))
        report["records"] = _table(report.get("records"))
        return _json(report)
    return _json(results)


def export_records(evaluator: str, results: object) -> list[dict[str, Any]]:
    """Capture complete per-case native data for one supported batch evaluator.

    Pydantic Evals accepts EvaluationReport; Azure accepts evaluate() results;
    Evidently accepts Dataset/dataframe or {rows, score_columns}; MLflow accepts
    EvaluationResult or {metrics, prediction_table, columns, score_columns};
    Garak accepts Attempt objects/report entries; OpenAI Evals accepts recorder
    events; TruLens accepts (records_dataframe, feedback_columns) or a records
    envelope with model Record/FeedbackResult serializations.
    """
    results = serialize_results(evaluator, results)
    functions = {
        "pydantic-evals": _pydantic,
        "azure-ai-evaluation": _azure,
        "garak": _garak,
        "openai-evals": _openai,
        "trulens": _trulens,
    }
    if evaluator in {"evidently", "mlflow"}:
        records = _scored_table(results, evaluator)
    else:
        records = functions[evaluator](results)
    _array(records, f"{evaluator} captured records")
    ids = [row["id"] for row in records]
    if len(ids) != len(set(ids)):
        raise EvaluationRecordsError(f"{evaluator} duplicate independent record IDs")
    return cast(list[dict[str, Any]], _json(records))
