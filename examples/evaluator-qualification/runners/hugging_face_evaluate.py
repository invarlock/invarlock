"""Execute Hugging Face Evaluate's upstream exact-match module."""

import evaluate
from runner_support import arguments, finish_deterministic, load_inputs


def main() -> None:
    args = arguments()
    _, _, cases = load_inputs(args)
    metric = evaluate.load("exact_match")
    scores: list[float] = []
    details = []
    for case in cases:
        result = metric.compute(
            predictions=[case["output"]],
            references=[case["reference"]],
        )
        score = float(result["exact_match"])
        scores.append(score)
        details.append({"metric_result": {"exact_match": score}})
    finish_deterministic(
        args=args,
        entrypoint="evaluate.load('exact_match').compute",
        scores=scores,
        details=details,
    )


if __name__ == "__main__":
    main()
