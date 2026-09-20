"""Reconstruct retained facts in pinned SDK containers, then serialize them.

This exercises SDK serialization, not model execution. References absent from
SDK result models remain explicit source columns. No model output, likelihood,
grade, or failed task is inferred by this helper.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from invarlock.evaluation_records.batch_integrations import (
    export_records,
    serialize_results,
)


def roundtrip(evaluator: str, payload: Any, *, tmp_path: Path) -> Any:
    """Return real SDK serialization of a supported complete native payload.

    The caller supplies a pinned SDK environment and network isolation. The
    returned JSON includes SDK context and is suitable for either raw import or
    the public export envelope. All original case facts are checked before return.
    """
    original = export_records(evaluator, payload)
    payload = deepcopy(payload)
    tmp_path.mkdir(parents=True, exist_ok=True)
    if evaluator == "pydantic-evals":
        from pydantic_evals.reporting import (
            EvaluationReport,
            ReportCase,
            ReportCaseFailure,
        )

        cases, failures = [], []
        for row in original:
            common = {
                "name": row["id"],
                "inputs": row["input"],
                "expected_output": row["expected"],
                "metadata": _capture_metadata(row),
            }
            if row["error"] is not None:
                if row["output"] is not None:
                    raise ValueError(
                        "Pydantic failed case cannot retain a partial output"
                    )
                failures.append(
                    ReportCaseFailure(
                        **common,
                        error_message=row["error"],
                        error_stacktrace="",
                    )
                )
            else:
                cases.append(
                    ReportCase(
                        **common,
                        output=row["output"],
                        metrics={},
                        attributes={
                            "capture_scope": "retained SDK serialization; task timing unavailable"
                        },
                        scores={},
                        labels={},
                        assertions={},
                        task_duration=0.0,
                        total_duration=0.0,
                    )
                )
        value = EvaluationReport(
            name="retained-capture", cases=cases, failures=failures
        )
    elif evaluator == "evidently":
        import pandas as pd
        from evidently import DataDefinition, Dataset

        value = Dataset.from_pandas(
            pd.DataFrame(payload["rows"]),
            data_definition=DataDefinition(),
        )
    elif evaluator == "mlflow":
        from mlflow.models import EvaluationResult
        from mlflow.models.evaluation.artifacts import JsonEvaluationArtifact

        artifact = JsonEvaluationArtifact(
            uri=(tmp_path / "prediction-table.json").as_uri(),
            content=payload["prediction_table"],
        )
        result = EvaluationResult(
            metrics=payload.get("metrics", {}),
            artifacts={"eval_results_table": artifact},
        )
        result_path = tmp_path / "evaluation-result"
        result.save(result_path)
        restored = EvaluationResult.load(result_path)
        value = {
            "metrics": restored.metrics,
            "prediction_table": restored.artifacts["eval_results_table"].content,
        }
    elif evaluator == "garak":
        from garak.attempt import Attempt, Message

        attempts, source_cases = [], []
        for row in original:
            attempt = Attempt(prompt=Message(text=row["input"]), status=2)
            # An absent answer remains absent: the SDK permits no generated turns.
            if row["output"] is not None:
                attempt.outputs = [row["output"]]
            attempts.append(attempt)
            source_cases.append(
                {
                    "native_id": f"{attempt.uuid}:0",
                    "id": row["id"],
                    "input": row["input"],
                    "output": row["output"],
                    "expected": row["expected"],
                    "metadata": _capture_metadata(row),
                }
            )
        value = serialize_results(evaluator, {"attempts": attempts})
        for attempt, row in zip(value["attempts"], original, strict=True):
            if row["error"] is not None:
                attempt["error"] = row["error"]
        value["source_cases"] = source_cases
    elif evaluator == "trulens":
        from trulens.core.schema.record import Record

        records = [
            Record(
                record_id=row["id"],
                app_id="retained-capture",
                calls=[],
                main_input=row["input"],
                main_output=row["output"],
                main_error=row["error"],
                meta=_capture_metadata(row),
            )
            for row in original
        ]
        value = serialize_results(evaluator, {"records": records})
        # Record has no reference field. Join the independently retained source
        # reference after serializing, exactly as a prediction-table capture does.
        references = {row["id"]: row["expected"] for row in original}
        for row in value["records"]:
            row["ground_truth"] = references[row["record_id"]]
    elif evaluator == "openai-evals":
        from evals.record import Event

        value = {
            "events": [
                Event(
                    run_id="retained-capture",
                    event_id=index,
                    sample_id=event["sample_id"],
                    type=event["type"],
                    data=event["data"],
                    created_by="retained-serialization",
                    created_at="",
                )
                for index, event in enumerate(payload["events"])
            ]
        }
    elif evaluator == "azure-ai-evaluation":
        import json

        from azure.ai.evaluation import evaluate

        def capture_marker(*, record_id: str) -> dict[str, str]:
            return {"record_id": record_id}

        data = tmp_path / "retained-inputs.jsonl"
        with data.open("x", encoding="utf-8") as stream:
            for row in original:
                stream.write(
                    json.dumps(
                        {
                            "record_id": row["id"],
                            "input": row["input"],
                            "response": row["output"],
                            "ground_truth": row["expected"],
                            "metadata": _capture_metadata(row),
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
        value = evaluate(
            data=data,
            evaluators={"capture": capture_marker},
            output_path=tmp_path / "sdk-result.json",
            fail_on_evaluator_errors=True,
        )
        errors = {row["id"]: row["error"] for row in original}
        for row in value["rows"]:
            row["error"] = errors[row["inputs.record_id"]]
    else:
        raise ValueError(f"unsupported batch SDK roundtrip: {evaluator}")
    serialized = serialize_results(evaluator, value)
    actual = export_records(evaluator, serialized)
    actual_by_id = {row["id"]: row for row in actual}
    if set(actual_by_id) != {row["id"] for row in original}:
        raise AssertionError("SDK serialization changed complete case membership")
    for row in original:
        captured = actual_by_id[row["id"]]
        for key in (
            "id",
            "input",
            "output",
            "expected",
            "metadata",
            "error",
            "likelihood",
        ):
            if row.get(key) != captured.get(key):
                raise AssertionError(f"{evaluator} SDK changed {row['id']} field {key}")
    return serialized


def _capture_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Keep source metadata extensions, including exact recorded likelihood facts."""
    native = row["context"].get("source_case", row["context"]["upstream_record"])
    metadata = native.get("metadata", native.get("meta", row["metadata"]))
    result = deepcopy(metadata or {})
    if "likelihood" in row:
        result["invarlock_likelihood"] = deepcopy(row["likelihood"])
    return result
