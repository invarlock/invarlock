from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _filter_rule_id_items(
    items: list[Any],
    *,
    id_key: str,
    excluded_rules: set[str],
) -> list[Any]:
    filtered: list[Any] = []
    for item in items:
        if not isinstance(item, dict):
            filtered.append(item)
            continue
        if str(item.get(id_key, "")) in excluded_rules:
            continue
        filtered.append(item)
    return filtered


def _filter_run(run: dict[str, Any], excluded_rules: set[str]) -> dict[str, Any]:
    results = run.get("results")
    if isinstance(results, list):
        run["results"] = _filter_rule_id_items(
            results,
            id_key="ruleId",
            excluded_rules=excluded_rules,
        )

    driver = run.get("tool", {}).get("driver")
    if isinstance(driver, dict):
        rules = driver.get("rules")
        if isinstance(rules, list):
            driver["rules"] = _filter_rule_id_items(
                rules,
                id_key="id",
                excluded_rules=excluded_rules,
            )

    extensions = run.get("tool", {}).get("extensions")
    if isinstance(extensions, list):
        for extension in extensions:
            if not isinstance(extension, dict):
                continue
            rules = extension.get("rules")
            if isinstance(rules, list):
                extension["rules"] = _filter_rule_id_items(
                    rules,
                    id_key="id",
                    excluded_rules=excluded_rules,
                )

    return run


def filter_sarif(payload: dict[str, Any], excluded_rules: set[str]) -> dict[str, Any]:
    runs = payload.get("runs")
    if not isinstance(runs, list):
        return payload
    payload["runs"] = [
        _filter_run(run, excluded_rules) if isinstance(run, dict) else run
        for run in runs
    ]
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter selected Scorecard SARIF rules before upload."
    )
    parser.add_argument("--input", required=True, dest="input_path")
    parser.add_argument("--output", required=True, dest="output_path")
    parser.add_argument(
        "--exclude-rule",
        action="append",
        default=[],
        dest="excluded_rules",
        help="Rule ID to remove from SARIF results.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    filtered = filter_sarif(payload, set(args.excluded_rules))
    output_path.write_text(
        json.dumps(filtered, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
