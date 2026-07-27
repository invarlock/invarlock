"""Execute Phoenix Evals' upstream exact-match metric."""

from phoenix.evals.metrics import exact_match
from runner_support import arguments, finish_deterministic, load_inputs


def main() -> None:
    args = arguments()
    _, _, cases = load_inputs(args)
    scores: list[float] = []
    details = []
    for case in cases:
        result = exact_match(case["output"], case["reference"])
        score = float(result.score)
        scores.append(score)
        details.append({"score": score})
    finish_deterministic(
        args=args,
        entrypoint="phoenix.evals.metrics.exact_match",
        scores=scores,
        details=details,
    )


if __name__ == "__main__":
    main()
