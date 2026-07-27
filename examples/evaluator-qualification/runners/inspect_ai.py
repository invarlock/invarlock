"""Execute Inspect AI's upstream exact-match scorer."""

import asyncio

from inspect_ai.model import ChatMessageUser, ModelOutput
from inspect_ai.scorer import Target, match
from inspect_ai.solver import TaskState
from runner_support import arguments, finish_deterministic, load_inputs


async def run() -> None:
    args = arguments()
    _, _, cases = load_inputs(args)
    scorer = match(location="exact", ignore_case=False)
    scores: list[float] = []
    details = []
    for case in cases:
        state = TaskState(
            model="offline",
            sample_id=case["record_id"],
            epoch=1,
            input=case["input"],
            messages=[ChatMessageUser(content=case["input"])],
            output=ModelOutput.from_content(model="offline", content=case["output"]),
        )
        result = await scorer(state, Target(case["reference"]))
        value = str(result.value)
        score = 1.0 if value == "C" else 0.0
        scores.append(score)
        details.append({"answer": result.answer, "score_value": value})
    finish_deterministic(
        args=args,
        entrypoint="inspect_ai.scorer.match",
        scores=scores,
        details=details,
    )


if __name__ == "__main__":
    asyncio.run(run())
