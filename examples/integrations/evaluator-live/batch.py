"""Execute fresh application cases with native batch evaluation frameworks.

The caller owns the model task and its authorization. This module neither
chooses a model nor estimates likelihoods. Each task result must retain its
actual output/error and any measured facts in metadata.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from uuid import UUID

VERSIONS = {
    "pydantic-evals": "2.18.0",
    "azure-ai-evaluation": "1.18.1",
    "evidently": "0.7.21",
    "mlflow": "3.14.0",
    "trulens": "2.9.0",
}


def _plain(value):
    """Serialize declared SDK data without importing the recipient package."""
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("capture values must be finite")
        return value
    if isinstance(value, Enum):
        return _plain(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("capture object keys must be strings")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if dataclasses.is_dataclass(value):
        return {
            field.name: _plain(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if type(value).__module__.startswith("trulens.") and isinstance(
        getattr(type(value), "model_fields", None), dict
    ):
        return {
            name: _plain(getattr(value, name))
            for name, field in type(value).model_fields.items()
            if field.exclude is not True
        }
    if type(value).__module__.startswith("numpy") and callable(
        getattr(value, "item", None)
    ):
        return _plain(value.item())
    if type(value).__module__.startswith(
        ("pydantic_evals.", "pydantic_ai.")
    ) and callable(getattr(value, "model_dump", None)):
        return _plain(value.model_dump(mode="python"))
    raise ValueError(f"unsupported captured SDK value: {type(value).__name__}")


class TaskFailure(RuntimeError):
    """A task returned a captured execution failure."""


class Capture:
    def __init__(self, cases, task):
        if not isinstance(cases, list) or not cases or len(cases) > 50000:
            raise ValueError("cases must be a bounded nonempty list")
        self.cases = copy.deepcopy(cases)
        self.task = task
        self.results = {}
        self.by_id = {}
        for case in self.cases:
            if (
                not isinstance(case, dict)
                or not isinstance(case.get("id"), str)
                or not case["id"].strip()
                or case["id"] in self.by_id
                or not isinstance(case.get("input"), str)
                or "expected" not in case
                or not isinstance(case.get("metadata"), dict)
            ):
                raise ValueError(
                    "cases require unique IDs, text input, expected, and metadata"
                )
            if "invarlock_task_capture" in case["metadata"]:
                raise ValueError(
                    "invarlock_task_capture is reserved for the actual task result"
                )
            self.by_id[case["id"]] = case
        _plain(self.cases)

    def invoke(self, ident, prompt):
        if ident not in self.by_id or self.by_id[ident]["input"] != prompt:
            raise ValueError(
                "framework task identity or input differs from planned case"
            )
        if ident in self.results:
            raise ValueError("framework attempted the same planned task more than once")
        case = self.by_id[ident]
        try:
            result = self.task(copy.deepcopy(case))
        except Exception as exc:
            result = {"output": None, "error": f"{type(exc).__name__}: {exc}"}
        if (
            not isinstance(result, dict)
            or "output" not in result
            or (result["output"] is not None and not isinstance(result["output"], str))
            or (
                result.get("error") is not None
                and (not isinstance(result["error"], str) or not result["error"])
            )
            or not isinstance(result.get("metadata", {}), dict)
        ):
            raise ValueError(
                "task result requires text/null output, optional error, and metadata"
            )
        _plain(result)
        metadata = copy.deepcopy(case["metadata"])
        for key, value in result.get("metadata", {}).items():
            if key in metadata and metadata[key] != value:
                raise ValueError("task metadata conflicts with planned metadata")
            metadata[key] = copy.deepcopy(value)
        if "invarlock_task_capture" in metadata:
            raise ValueError(
                "invarlock_task_capture is reserved for the actual task result"
            )
        metadata["invarlock_task_capture"] = copy.deepcopy(result)
        captured = {
            "output": result["output"],
            "error": result.get("error"),
            "metadata": metadata,
        }
        self.results[ident] = captured
        return captured

    def rows(self, *, prediction="output", expected="reference"):
        if set(self.results) != set(self.by_id):
            raise ValueError("framework did not execute the complete planned schedule")
        return [
            {
                "record_id": case["id"],
                "input": case["input"],
                prediction: self.results[case["id"]]["output"],
                expected: case["expected"],
                "metadata": self.results[case["id"]]["metadata"],
                "error": self.results[case["id"]]["error"],
            }
            for case in self.cases
        ]


def _pydantic(capture, workdir):
    import asyncio
    from contextvars import ContextVar

    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import EqualsExpected
    from pydantic_evals.lifecycle import CaseLifecycle

    current = ContextVar("live_pydantic_case")

    class Lifecycle(CaseLifecycle):
        async def setup(self):
            current.set(self.case.name)

        async def teardown(self, result):
            if self.case.name in capture.results and result is not None:
                result.metadata = capture.results[self.case.name]["metadata"]

    async def execute(prompt):
        result = capture.invoke(current.get(), prompt)
        if result["error"] is not None:
            raise TaskFailure(result["error"])
        return result["output"]

    dataset = Dataset(
        name="live-qualification",
        cases=[
            Case(
                name=case["id"],
                inputs=case["input"],
                expected_output=case["expected"],
                metadata=case["metadata"],
            )
            for case in capture.cases
        ],
        evaluators=[EqualsExpected()],
    )
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        report = dataset.evaluate_sync(
            execute, max_concurrency=1, progress=False, lifecycle=Lifecycle
        )
    finally:
        loop.close()
        asyncio.set_event_loop(None)
    native = _plain(report)
    _write(workdir / "sdk-report.json", native)
    for row in native["failures"]:
        captured = capture.results[row["name"]]
        row["sdk_error_message"] = row["error_message"]
        row["error_message"] = captured["error"]
        row["output"] = captured["output"]
    return native


def _azure(capture, workdir):
    from azure.ai.evaluation import evaluate

    def target(*, record_id, input):
        result = capture.invoke(record_id, input)
        return {"response": result["output"], "task_error": result["error"]}

    def exact_match(*, response, ground_truth):
        return {
            "exact_match": float(isinstance(response, str) and response == ground_truth)
        }

    data = workdir / "cases.jsonl"
    with data.open("x", encoding="utf-8") as stream:
        for case in capture.cases:
            stream.write(
                json.dumps(
                    {
                        "record_id": case["id"],
                        "input": case["input"],
                        "ground_truth": case["expected"],
                    },
                    allow_nan=False,
                )
                + "\n"
            )
    native = evaluate(
        data=data,
        target=target,
        evaluators={"exact_match": exact_match},
        output_path=workdir / "sdk-report.json",
        fail_on_evaluator_errors=False,
    )
    for row in native["rows"]:
        row["metadata"] = capture.results[row["inputs.record_id"]]["metadata"]
        row["error"] = row["outputs.task_error"]
    return native


def _evidently(capture, workdir):
    import pandas as pd
    from evidently import DataDefinition, Dataset
    from evidently.descriptors import ExactMatch

    for case in capture.cases:
        capture.invoke(case["id"], case["input"])
    dataset = Dataset.from_pandas(
        pd.DataFrame(capture.rows()),
        data_definition=DataDefinition(),
        descriptors=[ExactMatch(columns=["output", "reference"], alias="exact_match")],
    )
    rows = dataset.as_dataframe().to_dict(orient="records")
    originals = {row["record_id"]: row for row in capture.rows()}
    for row in rows:
        for key in ("output", "reference", "error"):
            # Dataset descriptor joins use pandas missing-value sentinels. Only
            # restore a null when the independent application capture proves it.
            if (
                isinstance(row[key], float)
                and math.isnan(row[key])
                and originals[row["record_id"]][key] is None
            ):
                row[key] = None
    native = {"rows": _plain(rows)}
    _write(workdir / "sdk-scored-dataset.json", native)
    return native


def _mlflow(capture, workdir):
    import os

    import mlflow
    import pandas as pd
    from mlflow.metrics.base import MetricValue
    from mlflow.models import EvaluationResult, make_metric

    if mlflow.active_run() is not None:
        raise ValueError("live MLflow capture requires its own local run")
    for case in capture.cases:
        capture.invoke(case["id"], case["input"])
    rows = capture.rows(prediction="prediction", expected="target")

    def exact_match(predictions, targets, metrics):
        scores = [
            float(isinstance(output, str) and output == target)
            for output, target in zip(predictions, targets, strict=True)
        ]
        return MetricValue(
            scores=scores, aggregate_results={"mean": sum(scores) / len(scores)}
        )

    previous = mlflow.get_tracking_uri()
    previous_directory = Path.cwd()
    os.chdir(workdir)
    mlflow.set_tracking_uri(f"sqlite:///{workdir / 'mlflow.sqlite'}")
    try:
        experiment = mlflow.create_experiment(
            "live-qualification", artifact_location=(workdir / "artifacts").as_uri()
        )
        with mlflow.start_run(experiment_id=experiment):
            result = mlflow.models.evaluate(
                data=pd.DataFrame(
                    [
                        {
                            key: value
                            for key, value in row.items()
                            if key not in {"metadata", "error"}
                        }
                        for row in rows
                    ]
                ),
                predictions="prediction",
                targets="target",
                model_type=None,
                extra_metrics=[
                    make_metric(
                        eval_fn=exact_match, greater_is_better=True, name="exact_match"
                    )
                ],
                evaluators=["default"],
            )
            result.save(workdir / "sdk-evaluation")
        restored = EvaluationResult.load(workdir / "sdk-evaluation")
        table = restored.artifacts["eval_results_table"].content
        columns, data = table["columns"], table["data"]
        native_rows = [dict(zip(columns, row, strict=True)) for row in data]
        by_id = {row["record_id"]: row for row in rows}
        for row in native_rows:
            source = by_id[row["record_id"]]
            row["metadata"] = source["metadata"]
            row["error"] = source["error"]
        return {"prediction_table": native_rows, "metrics": restored.metrics}
    finally:
        mlflow.set_tracking_uri(previous)
        os.chdir(previous_directory)


def _trulens(capture, workdir):
    import os

    # TruLens 2.9 retains its public local Record API under this documented mode.
    previous = os.environ.get("TRULENS_OTEL_TRACING")
    os.environ["TRULENS_OTEL_TRACING"] = "0"
    try:
        from trulens.apps.basic import TruBasicApp
        from trulens.core import TruSession
        from trulens.core.schema.feedback import FeedbackMode

        class LosslessBasicApp(TruBasicApp):
            def main_output(self, func, sig, bindings, ret):
                # This public hook preserves nullable answers instead of the
                # generic SDK output selection's implicit string conversion.
                return ret

        # This pinned SDK ignores constructor arguments for an existing default
        # singleton. Reject it before any task call or implicit database reuse.
        from trulens.core.utils.python import SingletonPerNameMeta

        key = (f"{TruSession.__module__}.{TruSession.__name__}", None)
        if key in SingletonPerNameMeta._singleton_instances:
            raise ValueError("live TruLens capture requires a fresh SDK worker process")
        TruSession(database_url=f"sqlite:///{workdir / 'trulens.sqlite'}")
        records = []

        def application_for(ident):
            def application(prompt):
                result = capture.invoke(ident, prompt)
                if result["error"] is not None:
                    raise TaskFailure(result["error"])
                return result["output"]

            return application

        for case in capture.cases:
            recorder = LosslessBasicApp(
                application_for(case["id"]),
                app_name="live-qualification",
                app_version=case["id"],
                feedback_mode=FeedbackMode.NONE,
            )
            with recorder as recording:
                try:
                    recorder.app(case["input"])
                except TaskFailure:
                    pass
            record = recording.get()
            native = _plain(record)
            native["sdk_main_output"] = native["main_output"]
            native["main_output"] = capture.results[case["id"]]["output"]
            native["sdk_main_error"] = native["main_error"]
            native["main_error"] = capture.results[case["id"]]["error"]
            native["source_record_id"] = native["record_id"]
            native["record_id"] = case["id"]
            native["ground_truth"] = case["expected"]
            native["meta"] = capture.results[case["id"]]["metadata"]
            records.append(native)
        value = {"records": records}
        _write(workdir / "sdk-records.json", value)
        return value
    finally:
        if previous is None:
            os.environ.pop("TRULENS_OTEL_TRACING", None)
        else:
            os.environ["TRULENS_OTEL_TRACING"] = previous


def _write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)


def _observations(evaluator, native):
    """Check the documented native fields before handing bytes to the recipient."""
    if evaluator == "pydantic-evals":
        rows = native["cases"] + native["failures"]
        roles = (
            "name",
            "inputs",
            "output",
            "expected_output",
            "metadata",
            "error_message",
        )
    elif evaluator == "azure-ai-evaluation":
        rows = native["rows"]
        roles = (
            "inputs.record_id",
            "inputs.input",
            "outputs.response",
            "inputs.ground_truth",
            "metadata",
            "error",
        )
    elif evaluator == "trulens":
        rows = native["records"]
        roles = (
            "record_id",
            "main_input",
            "main_output",
            "ground_truth",
            "meta",
            "main_error",
        )
    elif evaluator == "mlflow":
        rows = native["prediction_table"]
        roles = ("record_id", "input", "prediction", "target", "metadata", "error")
        if rows and "predictions" in rows[0]:
            roles = ("record_id", "input", "predictions", "target", "metadata", "error")
        if rows and "targets" in rows[0]:
            roles = (*roles[:3], "targets", *roles[4:])
    else:
        rows = native["rows"]
        roles = ("record_id", "input", "output", "reference", "metadata", "error")
    return [
        dict(
            zip(
                ("id", "input", "output", "expected", "metadata", "error"),
                (row.get(key) for key in roles),
                strict=True,
            )
        )
        for row in rows
    ]


def run(evaluator, cases, task, workdir):
    """Execute every supplied case once and return its actual native capture."""
    functions = {
        "pydantic-evals": _pydantic,
        "azure-ai-evaluation": _azure,
        "evidently": _evidently,
        "mlflow": _mlflow,
        "trulens": _trulens,
    }
    if evaluator not in functions:
        raise ValueError(f"unsupported live batch evaluator: {evaluator}")
    capture = Capture(cases, task)
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    if any(workdir.iterdir()):
        raise ValueError("live capture workdir must be empty")
    native = _plain(functions[evaluator](capture, workdir))
    capture.rows()
    records = _observations(evaluator, native)
    if len(records) != len(capture.by_id) or {row["id"] for row in records} != set(
        capture.by_id
    ):
        raise ValueError("SDK export differs from complete planned case membership")
    for row in records:
        case, result = capture.by_id[row["id"]], capture.results[row["id"]]
        if (row["input"], row["expected"], row["output"]) != (
            case["input"],
            case["expected"],
            result["output"],
        ):
            raise ValueError(
                "SDK export changed original task input, reference, or output"
            )
        if bool(row["error"]) != bool(result["error"]):
            raise ValueError("SDK export changed task failure status")
        if row["metadata"] != result["metadata"]:
            raise ValueError(
                "SDK export changed case metadata or measured likelihood facts"
            )
    _write(workdir / "application-captures.json", capture.results)
    _write(workdir / "native.json", native)
    return native
