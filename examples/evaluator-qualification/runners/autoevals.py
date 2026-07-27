"""Execute Braintrust AutoEvals' upstream ExactMatch evaluator."""

from autoevals import ExactMatch
from runner_support import arguments, finish_deterministic, load_inputs


def main() -> None:
    args = arguments()
    _, _, cases = load_inputs(args)
    metric = ExactMatch()
    scores: list[float] = []
    details = []
    for case in cases:
        result = metric(output=case["output"], expected=case["reference"])
        score = float(result.score)
        scores.append(score)
        details.append({"name": result.name, "score": score})
    finish_deterministic(
        args=args,
        entrypoint="autoevals.ExactMatch.__call__",
        scores=scores,
        details=details,
    )


if __name__ == "__main__":
    main()
