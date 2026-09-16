"""Execute pinned local batch scorers and preserve their per-record source fields."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def evidently(cases: list[dict[str, str]]) -> dict[str, Any]:
    import pandas as pd
    from evidently import DataDefinition, Dataset
    from evidently.descriptors import ExactMatch

    dataset = Dataset.from_pandas(
        pd.DataFrame(
            [
                {key: case[key] for key in ("record_id", "output", "reference")}
                for case in cases
            ]
        ),
        data_definition=DataDefinition(),
        descriptors=[ExactMatch(columns=["output", "reference"], alias="exact_match")],
    )
    return {"rows": dataset.as_dataframe().to_dict("records")}


def langfuse(cases: list[dict[str, str]]) -> dict[str, Any]:
    from langfuse import Evaluation, Langfuse

    def evaluate(*, output: str, expected_output: str, **_: object) -> Evaluation:
        return Evaluation(
            name="exact_match", value=output == expected_output, data_type="BOOLEAN"
        )

    client = Langfuse(
        public_key="offline-public", secret_key="offline-secret", tracing_enabled=False
    )
    result = client.run_experiment(
        name="invarlock-local-qualified-batch",
        data=[
            {
                "input": case["input"],
                "expected_output": case["reference"],
                "metadata": {"record_id": case["record_id"], "output": case["output"]},
            }
            for case in cases
        ],
        task=lambda *, item, **_: item["metadata"]["output"],
        evaluators=[evaluate],
        max_concurrency=1,
    )
    return {
        "item_results": [
            {
                "item": row.item,
                "output": row.output,
                "evaluations": [
                    {
                        "name": metric.name,
                        "value": metric.value,
                        "data_type": metric.data_type,
                    }
                    for metric in row.evaluations
                ],
            }
            for row in result.item_results
        ],
        "run_evaluations": [
            {"name": metric.name, "value": metric.value, "data_type": metric.data_type}
            for metric in result.run_evaluations
        ],
    }


def pydantic(cases: list[dict[str, str]]) -> dict[str, Any]:
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import EqualsExpected

    dataset = Dataset(
        name="invarlock-local-qualified-batch",
        cases=[
            Case(
                name=case["record_id"],
                inputs={key: case[key] for key in ("record_id", "input", "output")},
                expected_output=case["reference"],
            )
            for case in cases
        ],
        evaluators=[EqualsExpected(evaluation_name="exact_match")],
    )
    # Use each case's captured output. Equal prompts must not overwrite one another.
    report = dataset.evaluate_sync(lambda item: item["output"], max_concurrency=1)
    return {
        "cases": [
            {
                "name": row.name,
                "inputs": row.inputs,
                "expected_output": row.expected_output,
                "output": row.output,
                "assertions": {
                    name: {"value": metric.value}
                    for name, metric in row.assertions.items()
                },
                "evaluator_failures": [str(item) for item in row.evaluator_failures],
                "scores": {name: metric.value for name, metric in row.scores.items()},
                "labels": {name: metric.value for name, metric in row.labels.items()},
            }
            for row in report.cases
        ],
        "failures": [str(item) for item in report.failures],
        "report_evaluator_failures": [
            str(item) for item in report.report_evaluator_failures
        ],
    }


def azure(cases: list[dict[str, str]]) -> dict[str, Any]:
    from azure.ai.evaluation import evaluate

    def exact_match(*, response: str, ground_truth: str) -> dict[str, float]:
        return {"exact_match": float(response == ground_truth)}

    with tempfile.TemporaryDirectory(prefix="invarlock-native-azure-") as temporary:
        root = Path(temporary)
        data_path = root / "cases.jsonl"
        data_path.write_text(
            "".join(
                json.dumps(
                    {
                        "record_id": case["record_id"],
                        "response": case["output"],
                        "ground_truth": case["reference"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for case in cases
            ),
            encoding="utf-8",
        )
        result = evaluate(
            data=data_path,
            evaluators={"exact_match": exact_match},
            evaluator_config={
                "exact_match": {
                    "column_mapping": {
                        "ground_truth": "${data.ground_truth}",
                        "response": "${data.response}",
                    }
                }
            },
            output_path=root / "results.jsonl",
            fail_on_evaluator_errors=True,
        )
    return {"rows": result.get("rows")}


def promptfoo(
    cases: list[dict[str, str]], *, version: str, dependency_lock: Path
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    lock = dict(line.split("=", 1) for line in dependency_lock.read_text().splitlines())
    spec = f"promptfoo@{version}"
    if lock.get("package") != spec:
        raise ValueError("Promptfoo dependency declaration does not match the profile")
    metadata = json.loads(
        subprocess.run(
            ["npm", "view", spec, "dist.integrity", "dist.shasum", "--json"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    if metadata.get("dist.integrity") != lock.get("integrity") or metadata.get(
        "dist.shasum"
    ) != lock.get("shasum"):
        raise ValueError("Promptfoo registry integrity does not match the declaration")
    environment = os.environ.copy()
    environment["PROMPTFOO_DISABLE_TELEMETRY"] = "1"
    with tempfile.TemporaryDirectory(prefix="invarlock-native-promptfoo-") as temporary:
        root = Path(temporary)
        config = root / "promptfoo.json"
        config.write_text(
            json.dumps(
                {
                    "prompts": ["{{output}}"],
                    "providers": ["echo"],
                    "evaluateOptions": {"maxConcurrency": 1},
                    "tests": [
                        {
                            "description": case["record_id"],
                            "vars": {"output": case["output"]},
                            "assert": [{"type": "equals", "value": case["reference"]}],
                        }
                        for case in cases
                    ],
                },
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        output = root / "result.json"
        completed = subprocess.run(
            [
                "npx",
                "--yes",
                spec,
                "eval",
                "--config",
                str(config),
                "--output",
                str(output),
                "--no-cache",
                "--no-progress-bar",
                "--no-share",
                "--no-table",
            ],
            env=environment,
            check=False,
        )
        if completed.returncode not in (0, 100):
            raise RuntimeError(f"Promptfoo exited with {completed.returncode}")
        document = json.loads(output.read_bytes())
    return document, [
        {
            "name": "promptfoo",
            "version": version,
            "integrity": metadata["dist.integrity"],
            "shasum": metadata["dist.shasum"],
        }
    ]


def execute(
    provider: str, cases: list[dict[str, str]], *, version: str, dependency_lock: Path
) -> tuple[dict[str, Any], list[dict[str, str]] | None]:
    if provider == "promptfoo":
        return promptfoo(cases, version=version, dependency_lock=dependency_lock)
    functions = {
        "evidently": evidently,
        "langfuse": langfuse,
        "pydantic-evals": pydantic,
        "azure-ai-evaluation": azure,
    }
    return functions[provider](cases), None
