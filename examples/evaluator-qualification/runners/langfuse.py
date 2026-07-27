"""Execute Langfuse's upstream local experiment evaluator."""

from langfuse import Evaluation, Langfuse
from runner_support import arguments, finish_deterministic, load_inputs


def evaluate_exact_match(
    *,
    output: str,
    expected_output: str,
    **_: object,
) -> Evaluation:
    return Evaluation(
        name="exact_match",
        value=output == expected_output,
        data_type="BOOLEAN",
    )


def main() -> None:
    args = arguments()
    _, _, cases = load_inputs(args)
    client = Langfuse(
        public_key="offline-public",
        secret_key="offline-secret",
        tracing_enabled=False,
    )
    data = [
        {
            "expected_output": case["reference"],
            "input": case["input"],
            "metadata": {
                "output": case["output"],
                "record_id": case["record_id"],
            },
        }
        for case in cases
    ]
    result = client.run_experiment(
        name="invarlock-offline-qualification",
        data=data,
        task=lambda *, item, **_: item["metadata"]["output"],
        evaluators=[evaluate_exact_match],
        max_concurrency=1,
    )
    values: dict[str, bool] = {}
    for item_result in result.item_results:
        record_id = str(item_result.item["metadata"]["record_id"])
        values[record_id] = bool(item_result.evaluations[0].value)
    scores = [1.0 if values[case["record_id"]] else 0.0 for case in cases]
    details = [{"exact_match": bool(score)} for score in scores]
    finish_deterministic(
        args=args,
        entrypoint="langfuse.Langfuse.run_experiment",
        scores=scores,
        details=details,
    )


if __name__ == "__main__":
    main()
