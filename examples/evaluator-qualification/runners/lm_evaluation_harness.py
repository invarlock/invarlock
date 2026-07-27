"""Execute LM Evaluation Harness's upstream exact-match metric."""

from lm_eval.api.metrics import exact_match_hf_evaluate
from runner_support import arguments, finish_deterministic, load_inputs


def main() -> None:
    args = arguments()
    _, _, cases = load_inputs(args)
    scores: list[float] = []
    details = []
    for case in cases:
        result = exact_match_hf_evaluate([case["output"]], [case["reference"]])
        score = float(result["exact_match"])
        scores.append(score)
        details.append({"metric_result": {"exact_match": score}})
    finish_deterministic(
        args=args,
        entrypoint="lm_eval.api.metrics.exact_match_hf_evaluate",
        scores=scores,
        details=details,
    )


if __name__ == "__main__":
    main()
