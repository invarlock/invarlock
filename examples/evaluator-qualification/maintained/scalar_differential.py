"""Compare current and historical native scalar semantics without granting authority."""

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

from maintained.scalar_native import build_scorer  # noqa: E402
from maintained.scalar_semantics import (  # noqa: E402
    CONFIGURATIONS,
    validate_native,
    validate_pair,
    validate_result,
)


def _observe(
    score: Callable, case: dict[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return score(case), None
    except Exception as exc:
        # Exception classes describe native unsupported boundaries without retaining local paths.
        return None, type(exc).__name__


def audit(
    *,
    definition: dict[str, Any],
    corpus: dict[str, Any],
    current: Callable,
    historical: Callable,
    version: str,
    sources: dict[str, Any],
) -> dict[str, Any]:
    provider = definition["historical_profile"]
    if definition["scorer_configuration"] != CONFIGURATIONS[provider]:
        raise ValueError("current scalar scorer configuration changed")
    problems = []
    if sources != definition["source_bindings"]:
        problems.append("native module or source revision changed")
    records = []
    seen = set()
    for case in corpus["cases"]:
        record_id = case["record_id"]
        if not isinstance(record_id, str) or not record_id or record_id in seen:
            raise ValueError("boundary record IDs must be unique nonempty strings")
        seen.add(record_id)
        try:
            validate_pair(provider, case)
            supported = True
        except ValueError:
            supported = False
        if supported != (provider in case["supported_for"]):
            problems.append(f"supported domain changed for {record_id}")
        observation = {
            "record_id": record_id,
            "supported": supported,
            "current": None,
            "historical": None,
        }
        if all(isinstance(case[key], str) for key in ("input", "output", "reference")):
            expected = case["historical_overrides"].get(
                provider, {"score": float(case["output"] == case["reference"])}
            )
            native, error = _observe(historical, case)
            if error:
                observed = {"error_type": error}
            else:
                try:
                    value = validate_native(
                        provider, native, bool(expected.get("score"))
                    )
                    observed = {"score": value}
                except ValueError:
                    observed = {"error_type": "native_score_or_detail_drift"}
            if observed != expected:
                problems.append(f"historical native semantics changed for {record_id}")
            observation["historical"] = observed
        if supported:
            native, error = _observe(current, case)
            if error:
                observation["current"] = {"error_type": error}
                problems.append(f"current native execution failed for {record_id}")
            else:
                try:
                    observation["current"] = {
                        "score": validate_result(provider, case, native),
                        "native": native,
                    }
                except ValueError:
                    observation["current"] = {
                        "error_type": "native_score_or_detail_drift"
                    }
                    problems.append(f"current native semantics changed for {record_id}")
        records.append(observation)
    return {
        "format": "invarlock/scalar-semantic-observation-v1",
        "provider": provider,
        "authority": "none",
        "upstream_version": version,
        "candidate_dependency": version != definition["upstream"]["version"],
        "scorer_configuration": definition["scorer_configuration"],
        "source_bindings": sources,
        "semantic_drift": bool(problems),
        "problems": problems,
        "cases": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=CONFIGURATIONS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    definition = next(
        item
        for item in json.loads((ROOT / "scalar-profiles.json").read_bytes())["profiles"]
        if item["historical_profile"] == args.provider
    )
    current, sources = build_scorer(args.provider)
    historical, _ = build_scorer(args.provider, historical=True)
    result = audit(
        definition=definition,
        corpus=json.loads((ROOT / "scalar-boundaries.json").read_bytes()),
        current=current,
        historical=historical,
        version=importlib.metadata.version(definition["upstream"]["name"]),
        sources=sources,
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return 2 if result["semantic_drift"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
