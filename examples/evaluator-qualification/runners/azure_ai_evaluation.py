"""Execute Azure AI Evaluation's upstream batch orchestration."""

import json
import tempfile
from pathlib import Path

from azure.ai.evaluation import evaluate
from runner_support import arguments, finish_deterministic, load_inputs


def exact_match(
    *,
    response: str,
    ground_truth: str,
) -> dict[str, float]:
    return {"exact_match": float(response == ground_truth)}


def main() -> None:
    args = arguments()
    _, _, cases = load_inputs(args)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        data_path = root / "cases.jsonl"
        output_path = root / "results.jsonl"
        data_path.write_text(
            "".join(
                json.dumps(
                    {
                        "ground_truth": case["reference"],
                        "record_id": case["record_id"],
                        "response": case["output"],
                    },
                    separators=(",", ":"),
                    sort_keys=True,
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
            output_path=output_path,
            fail_on_evaluator_errors=True,
        )
    rows = result.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Azure AI Evaluation did not return per-row results")
    scores = [float(row["outputs.exact_match.exact_match"]) for row in rows]
    details = [{"exact_match": score} for score in scores]
    finish_deterministic(
        args=args,
        entrypoint="azure.ai.evaluation.evaluate",
        scores=scores,
        details=details,
    )


if __name__ == "__main__":
    main()
