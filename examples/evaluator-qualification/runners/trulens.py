"""Execute a deterministic metric through TruLens' upstream Metric API."""

from runner_support import arguments, finish_deterministic, load_inputs
from runners.trulens_metric import exact_match
from trulens.core import Metric


def main() -> None:
    args = arguments()
    _, _, cases = load_inputs(args)
    metric = Metric(implementation=exact_match, name="exact_match")
    scores = [float(metric(case["output"], case["reference"])) for case in cases]
    details = [{"score": score} for score in scores]
    finish_deterministic(
        args=args,
        entrypoint="trulens.core.Metric.__call__",
        scores=scores,
        details=details,
    )


if __name__ == "__main__":
    main()
