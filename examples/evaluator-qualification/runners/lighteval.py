"""Execute LightEval's upstream exact-match metric."""

from lighteval.metrics.metrics_sample import ExactMatches
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from runner_support import arguments, finish_deterministic, load_inputs


def main() -> None:
    args = arguments()
    _, _, cases = load_inputs(args)
    metric = ExactMatches(strip_strings=False, normalize_pred=None)
    scores: list[float] = []
    details = []
    for case in cases:
        document = Doc(
            task_name="invarlock-qualification",
            query=case["input"],
            choices=[case["reference"]],
            gold_index=0,
        )
        response = ModelResponse(text=[case["output"]])
        score = float(metric.compute(doc=document, model_response=response))
        scores.append(score)
        details.append({"score": score})
    finish_deterministic(
        args=args,
        entrypoint="lighteval.metrics.metrics_sample.ExactMatches.compute",
        scores=scores,
        details=details,
    )


if __name__ == "__main__":
    main()
