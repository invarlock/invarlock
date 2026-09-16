"""Observe native batch semantics; candidate observations carry no authority."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from maintained.batch_native import execute  # noqa: E402
from maintained.batch_semantics import PROVIDERS, project, validate_domain  # noqa: E402


def audit(
    provider: str,
    corpus: dict[str, Any],
    *,
    version: str,
    run: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> dict[str, Any]:
    definition = next(
        item
        for item in json.loads((ROOT / "batch-profiles.json").read_bytes())["profiles"]
        if item["historical_profile"] == provider
    )
    selected = []
    observations = []
    problems = []
    for case in corpus["cases"]:
        try:
            validate_domain(provider, [case])
            supported = True
        except ValueError:
            supported = False
        if supported != (provider in case["supported_for"]):
            problems.append(f"supported domain changed for {case['record_id']}")
        observations.append(
            {
                "record_id": case["record_id"],
                "supported": supported,
                "literal_score": float(case["output"] == case["reference"])
                if supported
                else None,
            }
        )
        if supported:
            selected.append(case)
    native = run(selected)
    try:
        scores, details = project(provider, selected, native)
    except ValueError as exc:
        problems.append(str(exc))
        scores, details = [], []
    return {
        "format": "invarlock/batch-semantic-observation-v1",
        "provider": provider,
        "upstream_version": version,
        "candidate_dependency": version != definition["upstream"]["version"],
        "authority": "none",
        "semantic_drift": bool(problems),
        "problems": problems,
        "scorer_configuration": definition["scorer_configuration"],
        "cases": observations,
        "native_scores": scores,
        "native_rows": [detail["native_row"] for detail in details],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=PROVIDERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=ROOT / "batch-boundaries.json")
    args = parser.parse_args(argv)
    definition = next(
        item
        for item in json.loads((ROOT / "batch-profiles.json").read_bytes())["profiles"]
        if item["historical_profile"] == args.provider
    )
    version = (
        definition["upstream"]["version"]
        if args.provider == "promptfoo"
        else importlib.metadata.version(definition["upstream"]["name"])
    )
    result = audit(
        args.provider,
        json.loads(args.corpus.read_bytes()),
        version=version,
        run=lambda cases: execute(
            args.provider,
            cases,
            version=version,
            dependency_lock=ROOT.parent / definition["lock"],
        )[0],
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return 2 if result["semantic_drift"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
