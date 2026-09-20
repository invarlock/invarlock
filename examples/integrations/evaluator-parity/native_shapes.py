"""Native field dictionaries for contract replay, never claimed SDK execution."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _metadata(evaluator, row, version):
    metadata = deepcopy(row["metadata"])
    if "likelihood" in row:
        metadata["invarlock_likelihood"] = {
            **row["likelihood"],
            "source": {"name": evaluator, "version": version},
        }
        metadata["invarlock_original_likelihood"] = deepcopy(row["likelihood"])
        if evaluator in ("lm-evaluation-harness", "promptfoo"):
            from invarlock.evaluation_record_contracts.contracts import digest

            native_input = {
                "text" if evaluator == "lm-evaluation-harness" else "prompt": row[
                    "input"
                ]
            }
            metadata["invarlock_likelihood"]["input_digest"] = digest(native_input)
    return metadata


def payload(
    evaluator: str,
    rows: list[dict[str, Any]],
    version: str,
    *,
    run_id: str = "retained-contract-replay",
) -> Any:
    """Express the same retained cases in each explicitly supported native shape."""
    results = []
    for row in rows:
        identifier, prompt, expected, output = (
            row[k] for k in ("id", "input", "expected", "output")
        )
        metadata = _metadata(evaluator, row, version)
        score = float(output == expected) if output is not None else 0.0
        common = {"id": identifier, "metadata": metadata}
        if evaluator == "inspect-ai":
            native = {
                "id": identifier,
                "input": prompt,
                "target": expected,
                "metadata": metadata,
                "output": {
                    "choices": []
                    if output is None
                    else [{"message": {"content": output}}]
                },
                "scores": {},
            }
        elif evaluator == "lm-evaluation-harness":
            native = {
                "doc_id": identifier,
                "doc": {"text": prompt},
                "target": expected,
                "arguments": [[prompt, expected]],
                "filtered_resps": [output],
                "metadata": metadata,
            }
        elif evaluator == "promptfoo":
            native = {
                "testIdx": identifier,
                "promptIdx": 0,
                "prompt": {"raw": prompt},
                "testCase": {
                    "vars": {"prompt": prompt},
                    "metadata": {
                        **metadata,
                        "invarlock_id": identifier,
                        "invarlock_expected": expected,
                    },
                },
                "response": {"output": output},
                "success": True,
                "failureReason": 0,
            }
        elif evaluator == "langfuse":
            native = {
                "item": {
                    "id": identifier,
                    "input": prompt,
                    "expected_output": expected,
                    "metadata": metadata,
                },
                "output": output,
                "evaluations": [],
                "trace_id": None,
                "dataset_run_id": None,
            }
        elif evaluator == "deepeval":
            native = {
                **common,
                "test_case": {
                    "input": prompt,
                    "actual_output": output,
                    "expected_output": expected,
                },
                "metric_result": {"score": score},
            }
        elif evaluator == "ragas":
            native = {
                **common,
                "sample": {
                    "user_input": prompt,
                    "response": output,
                    "reference": expected,
                },
                "metric_result": {"value": score},
            }
        elif evaluator == "lighteval":
            native = {
                **common,
                "doc": {"query": prompt, "choices": [expected], "gold_index": 0},
                "model_response": {"text": [output]},
                "metric_result": {"exact_match": score},
            }
        elif evaluator == "hugging-face-evaluate":
            native = {
                **common,
                "input": prompt,
                "predictions": [output],
                "references": [expected],
                "metric_result": {"exact_match": score},
            }
        elif evaluator == "autoevals":
            native = {
                **common,
                "input": prompt,
                "output": output,
                "expected": expected,
                "metric_result": {"name": "exact_match", "score": score},
            }
        elif evaluator == "openevals":
            native = {
                **common,
                "inputs": prompt,
                "outputs": output,
                "reference_outputs": expected,
                "metric_result": {"key": "exact_match", "score": score},
            }
        elif evaluator == "arize-phoenix-evals":
            native = {
                **common,
                "record": {"input": prompt, "output": output, "expected": expected},
                "metric_result": {"name": "exact_match", "score": score},
            }
        elif evaluator == "opik":
            native = {
                **common,
                "dataset_item": {
                    "input": prompt,
                    "output": output,
                    "reference": expected,
                },
                "metric_result": {"name": "exact_match", "value": score},
            }
        elif evaluator == "pydantic-evals":
            native = {
                "name": identifier,
                "inputs": prompt,
                "expected_output": expected,
                "output": output,
                "scores": {"exact_match": {"value": score}},
                "metadata": metadata,
            }
        elif evaluator == "azure-ai-evaluation":
            native = {
                "inputs.record_id": identifier,
                "inputs.input": prompt,
                "inputs.response": output,
                "inputs.ground_truth": expected,
                "outputs.exact_match.score": score,
                "metadata": metadata,
            }
        elif evaluator == "evidently":
            native = {
                "record_id": identifier,
                "input": prompt,
                "output": output,
                "reference": expected,
                "exact_match": score,
                "metadata": metadata,
            }
        elif evaluator == "mlflow":
            native = {
                "record_id": identifier,
                "input": prompt,
                "prediction": output,
                "target": expected,
                "exact_match/score": score,
                "metadata": metadata,
            }
        elif evaluator == "garak":
            native = {
                "uuid": identifier,
                "status": 2,
                "prompt": prompt,
                "outputs": [output],
                "detector_results": {"exact_match": [score]},
                "metadata": {
                    **metadata,
                    "invarlock_id": identifier,
                    "invarlock_expected": expected,
                },
            }
        elif evaluator == "trulens":
            native = {
                "record_id": identifier,
                "main_input": prompt,
                "main_output": output,
                "ground_truth": expected,
                "feedback_results": [{"name": "exact_match", "result": score}],
                "metadata": metadata,
            }
        elif evaluator == "openai-evals":
            results.extend(
                [
                    {
                        "sample_id": identifier,
                        "type": "sampling",
                        "data": {
                            "prompt": prompt,
                            "sampled": output,
                            "metadata": metadata,
                        },
                    },
                    {
                        "sample_id": identifier,
                        "type": "match",
                        "data": {"correct": bool(score), "expected": expected},
                    },
                ]
            )
            continue
        else:
            raise ValueError(f"unsupported evaluator profile: {evaluator}")
        results.append(native)
    if evaluator == "inspect-ai":
        return {"version": 2, "status": "success", "samples": results}
    if evaluator == "langfuse":
        return {
            "format": "invarlock/langfuse-export-v1",
            "sdk_version": version,
            "result": {
                "name": "retained-contract-replay",
                "run_name": run_id,
                "experiment_id": "contract-replay",
                "item_results": results,
                "run_evaluations": [],
            },
        }
    if evaluator == "pydantic-evals":
        return {"cases": results, "failures": []}
    if evaluator == "garak":
        cases = []
        for row in rows:
            metadata = _metadata(evaluator, row, version)
            cases.append(
                {
                    "native_id": row["id"] + ":0",
                    "id": row["id"],
                    "input": row["input"],
                    "expected": row["expected"],
                    "metadata": metadata,
                }
            )
        return {"attempts": results, "source_cases": cases}
    if evaluator == "mlflow":
        return {"prediction_table": results, "metrics": {}}
    if evaluator == "evidently":
        return {"rows": results, "score_columns": ["exact_match"]}
    wrapper = {
        "azure-ai-evaluation": "rows",
        "trulens": "records",
        "openai-evals": "events",
    }.get(evaluator)
    return {wrapper: results} if wrapper else results
