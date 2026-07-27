"""Execute Opik's upstream Equals metric."""

from opik.evaluation.metrics import Equals
from runner_support import arguments, finish_deterministic, load_inputs


def main() -> None:
    args = arguments()
    _, _, cases = load_inputs(args)
    metric = Equals()
    scores: list[float] = []
    details = []
    for case in cases:
        result = metric.score(
            output=case["output"],
            reference=case["reference"],
        )
        score = float(result.value)
        scores.append(score)
        details.append({"name": result.name, "value": score})
    finish_deterministic(
        args=args,
        entrypoint="opik.evaluation.metrics.Equals.score",
        scores=scores,
        details=details,
    )


if __name__ == "__main__":
    main()
