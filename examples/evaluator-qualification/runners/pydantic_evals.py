"""Execute Pydantic Evals with its EqualsExpected evaluator."""

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected
from runner_support import arguments, finish_deterministic, load_inputs


def main() -> None:
    args = arguments()
    _, _, cases = load_inputs(args)
    outputs = {case["input"]: case["output"] for case in cases}
    dataset = Dataset(
        name="invarlock-evaluator-qualification",
        cases=[
            Case(
                name=case["record_id"],
                inputs=case["input"],
                expected_output=case["reference"],
            )
            for case in cases
        ],
        evaluators=[EqualsExpected()],
    )
    report = dataset.evaluate_sync(lambda input_value: outputs[input_value])
    scores: list[float] = []
    details = []
    for evaluated in report.cases:
        assertion = next(iter(evaluated.assertions.values()))
        score = 1.0 if assertion.value else 0.0
        scores.append(score)
        details.append({"assertion": bool(assertion.value)})
    finish_deterministic(
        args=args,
        entrypoint="pydantic_evals.Dataset.evaluate_sync/EqualsExpected",
        scores=scores,
        details=details,
    )


if __name__ == "__main__":
    main()
