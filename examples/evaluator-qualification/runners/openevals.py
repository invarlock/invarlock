"""Execute OpenEvals' upstream exact-match evaluator."""

from openevals.exact import exact_match
from runner_support import arguments, finish_deterministic, load_inputs


def main() -> None:
    args = arguments()
    _, _, cases = load_inputs(args)
    scores: list[float] = []
    details = []
    for case in cases:
        result = exact_match(
            outputs=case["output"],
            reference_outputs=case["reference"],
        )
        score = 1.0 if result["score"] else 0.0
        scores.append(score)
        details.append({"key": result["key"], "score": bool(result["score"])})
    finish_deterministic(
        args=args,
        entrypoint="openevals.exact.exact_match",
        scores=scores,
        details=details,
    )


if __name__ == "__main__":
    main()
