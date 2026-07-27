"""Execute Ragas's upstream exact-match metric."""

import asyncio

from ragas.metrics.collections import ExactMatch
from runner_support import arguments, finish_deterministic, load_inputs


async def run() -> None:
    args = arguments()
    _, _, cases = load_inputs(args)
    metric = ExactMatch()
    scores: list[float] = []
    details = []
    for case in cases:
        score = float(
            await metric.ascore(
                reference=case["reference"],
                response=case["output"],
            )
        )
        scores.append(score)
        details.append({"score": score})
    finish_deterministic(
        args=args,
        entrypoint="ragas.metrics.collections.ExactMatch.ascore",
        scores=scores,
        details=details,
    )


if __name__ == "__main__":
    asyncio.run(run())
