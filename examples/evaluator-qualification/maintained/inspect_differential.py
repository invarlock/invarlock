"""Observe pinned or candidate Inspect semantics without granting qualification."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from maintained.inspect_semantics import (  # noqa: E402
    SCORER_CONFIGURATION,
    project_result,
    validate_cases,
)


async def audit(corpus: dict[str, Any], score: Any, *, version: str) -> dict[str, Any]:
    if corpus["scorer_configuration"] != SCORER_CONFIGURATION:
        raise ValueError("Inspect differential scorer configuration changed")
    observations = []
    for item in corpus["cases"]:
        case = {
            "record_id": item["id"],
            "output": item["output"],
            "reference": item["reference"],
        }
        result = await score(case)
        problems = []
        if result.value != ("C" if item["native_correct"] else "I"):
            problems.append("native score changed")
        try:
            validate_cases([case])
        except ValueError:
            supported = False
        else:
            supported = True
        if supported != item["supported"]:
            problems.append("supported pair domain changed")
        if supported:
            try:
                project_result(case, result)
            except ValueError as exc:
                problems.append(str(exc))
        observations.append(
            {
                "case_id": item["id"],
                "native_value": result.value,
                "native_answer": result.answer,
                "supported": supported,
                "problems": problems,
            }
        )
    return {
        "format": "invarlock/inspect-literal-differential-observation-v1",
        "authority": "none",
        "observed_package_version": version,
        "reference_package_version": corpus["upstream_version"],
        "candidate_dependency": version != corpus["upstream_version"],
        "scorer_configuration": SCORER_CONFIGURATION,
        "semantic_drift": any(item["problems"] for item in observations),
        "cases": observations,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from inspect_ai.model import ChatMessageUser, ModelOutput
    from inspect_ai.scorer import Target, match
    from inspect_ai.solver import TaskState

    scorer = match(**SCORER_CONFIGURATION)

    async def score(case: dict[str, Any]) -> Any:
        state = TaskState(
            model="offline",
            sample_id=case["record_id"],
            epoch=1,
            input="differential boundary case",
            messages=[ChatMessageUser(content="differential boundary case")],
            output=ModelOutput.from_content(model="offline", content=case["output"]),
        )
        return await scorer(state, Target(case["reference"]))

    corpus = json.loads((ROOT / "inspect-boundaries.json").read_bytes())
    result = asyncio.run(
        audit(corpus, score, version=importlib.metadata.version("inspect-ai"))
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(
            result,
            stream,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        stream.write("\n")
    print(
        json.dumps(
            {
                "authority": "none",
                "semantic_drift": result["semantic_drift"],
                "case_count": len(result["cases"]),
            }
        )
    )
    return 2 if result["semantic_drift"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
