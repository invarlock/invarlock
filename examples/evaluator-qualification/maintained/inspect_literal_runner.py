"""Execute Inspect's single-target scorer only on its literal-agreement domain."""

import asyncio

from inspect_ai.model import ChatMessageUser, ModelOutput
from inspect_ai.scorer import Target, match
from inspect_ai.solver import TaskState
from maintained.inspect_semantics import (
    PROFILE_ID,
    SCORER_CONFIGURATION,
    project_result,
    validate_cases,
)
from runner_support import (
    arguments,
    finish_deterministic,
    load_inputs,
    require_profile_package,
)


async def run() -> None:
    args = arguments()
    profile, _, cases = load_inputs(args)
    if profile["profile_id"] != PROFILE_ID:
        raise ValueError(
            "Inspect literal runner requires its separate versioned profile"
        )
    require_profile_package(profile)
    validate_cases(cases)
    scorer = match(**SCORER_CONFIGURATION)
    scores = []
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
        score, detail = project_result(case, result)
        scores.append(score)
        details.append(detail)
    finish_deterministic(
        args=args,
        entrypoint="inspect_ai.scorer.match",
        scores=scores,
        details=details,
    )


if __name__ == "__main__":
    asyncio.run(run())
