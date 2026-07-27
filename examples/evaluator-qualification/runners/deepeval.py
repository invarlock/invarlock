"""Execute DeepEval's upstream exact-match metric."""

from deepeval.metrics import ExactMatchMetric
from deepeval.test_case import LLMTestCase
from runner_support import arguments, finish_deterministic, load_inputs


def main() -> None:
    args = arguments()
    _, _, cases = load_inputs(args)
    scores: list[float] = []
    details = []
    for case in cases:
        metric = ExactMatchMetric()
        score = float(
            metric.measure(
                LLMTestCase(
                    input=case["input"],
                    actual_output=case["output"],
                    expected_output=case["reference"],
                )
            )
        )
        scores.append(score)
        details.append(
            {"metric_score": float(metric.score), "successful": metric.is_successful()}
        )
    finish_deterministic(
        args=args,
        entrypoint="deepeval.metrics.ExactMatchMetric.measure",
        scores=scores,
        details=details,
    )


if __name__ == "__main__":
    main()
