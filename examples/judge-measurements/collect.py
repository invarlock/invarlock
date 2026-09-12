"""Run the bounded Inspect judge collector for this directory's frozen inputs."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any


def _input(path: Path, *, maximum: int) -> dict[str, Any]:
    from invarlock.evidence_pack_json import parse_json_bytes, read_regular_file_bytes

    value = parse_json_bytes(
        read_regular_file_bytes(path, label=str(path), max_bytes=maximum),
        label=str(path),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


async def _collect(args: argparse.Namespace) -> dict[str, Any]:
    from inspect_ai.model import get_model
    from invarlock_addins.inspect_judge import (
        CollectionOptions,
        RunnerOptions,
        collect,
    )

    root = args.root.absolute()
    plan = _input(root / args.plan, maximum=16 * 1024 * 1024)
    options = CollectionOptions.from_mapping(
        _input(root / args.collection, maximum=1024 * 1024)
    )
    baseline = _input(root / args.baseline_run, maximum=128 * 1024 * 1024)
    subject = _input(root / args.subject_run, maximum=128 * 1024 * 1024)
    runner = RunnerOptions(
        checkpoint_directory=root / args.checkpoint,
        scorer_id=args.scorer_id,
        invocation_timeout_seconds=args.invocation_timeout_seconds,
    )
    model = get_model(
        options.grader,
        api_key=os.environ["OPENAI_API_KEY"],
        responses_api=False,
        max_retries=0,
        memoize=False,
    )
    client = getattr(getattr(model, "api", None), "client", None)
    try:
        return await collect(
            plan=plan,
            options=options,
            runner=runner,
            model=model,
            baseline_run=baseline,
            subject_run=subject,
        )
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            await close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute-collection",
        action="store_true",
        help="acknowledge that the command may make and bill provider calls",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--plan", default="plan.json")
    parser.add_argument("--collection", default="collection.json")
    parser.add_argument("--baseline-run", default="baseline_run.json")
    parser.add_argument("--subject-run", default="subject_run.json")
    parser.add_argument("--checkpoint", default="judge-checkpoint")
    parser.add_argument("--output", default="measurements-collected.json")
    parser.add_argument("--scorer-id", default="bounded-judge")
    parser.add_argument("--invocation-timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    if not args.execute_collection:
        parser.error(
            "collection is disabled until --execute-collection is supplied after reviewing the plan and resource caps"
        )
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY must be set in the collector environment")
    if os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE"):
        parser.error("custom OpenAI provider URLs are outside this qualified example")
    try:
        from invarlock.filesystem.atomic_file import write_file_no_replace
        from invarlock.judge_measurements.contracts import canonical_payload

        output = (args.root.absolute() / args.output).absolute()
        measurements = asyncio.run(_collect(args))
        write_file_no_replace(output, canonical_payload(measurements))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Judge collection failed: {exc}\n")
    completed = measurements["completeness"]["completed_trials"]
    expected = measurements["completeness"]["expected_trials"]
    print(f"Retained {completed}/{expected} completed trials in {output}")
    if completed != expected:
        print(
            "Collection is incomplete; rerun with the same checkpoint to inspect or resume it."
        )


if __name__ == "__main__":
    main()
