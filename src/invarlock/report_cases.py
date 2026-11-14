"""Case explanations from retained facts only; rendering never invokes scoring."""

from __future__ import annotations

from typing import Any


def captured_case_readings(
    inputs: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
) -> tuple[tuple[str, str], ...]:
    baseline = {r["id"]: r for r in inputs["baseline"]["records"]}
    candidate = {r["id"]: r for r in inputs["subject"]["records"]}
    rows = [
        (
            "Case selection",
            "For each overall metric, show the first record in stable ID order in each available category. Improved/regressed categories require retained per-case scores; otherwise only output changed, unchanged and missing are described, without rescoring. Counts cover the complete paired run, not overlapping slices. Raw inputs and outputs remain in companion evidence.",
        )
    ]
    for metric in inputs["policy"]["metrics"]:
        recorded = metric["kind"] == "recorded"
        missing_ids = set(
            next(
                m["missing_ids"]
                for m in comparison["metrics"]
                if m["name"] == metric["name"] and m["slice"] == "overall"
            )
        )
        categories = (
            ("improved", "regressed", "unchanged", "missing")
            if recorded
            else ("output changed", "unchanged", "missing")
        )
        counts = dict.fromkeys(categories, 0)
        examples: dict[str, str] = {}
        for key in sorted(baseline):
            left, right = baseline[key], candidate[key]
            score_key = metric.get("score_key")
            missing = key in missing_ids
            if missing:
                category = "missing"
                detail = "a retained result is missing or errored"
            elif recorded:
                old, new = left["scores"][score_key], right["scores"][score_key]
                category = (
                    "unchanged"
                    if old == new
                    else "improved"
                    if (new > old) == (metric["direction"] == "higher")
                    else "regressed"
                )
                detail = f"retained baseline score {old}; retained candidate score {new} {metric['unit']}"
            else:
                category = (
                    "unchanged"
                    if left["output"] == right["output"]
                    else "output changed"
                )
                detail = "retained output comparison only; task-score direction is not inferred"
            counts[category] += 1
            examples.setdefault(category, f"Record ID {key}: {detail}.")
        rows.append(
            (
                f"{metric['name']} case population",
                f"{len(baseline)} paired records; "
                + ", ".join(f"{n} {k}" for k, n in counts.items())
                + ". Attempted and excluded totals outside these retained runs are unknown.",
            )
        )
        rows.extend(
            (
                f"{metric['name']}: {category}",
                examples.get(
                    category, "No cases in this category in the retained population."
                ),
            )
            for category in categories
        )
    return tuple(rows)
