#!/usr/bin/env python3
"""Validate the neutral example-scenario catalog and its runbooks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jsonschema
import yaml

FORMAT_VERSION = "invarlock/example-scenario-v1"
REQUIRED_RUNBOOK_SECTIONS = (
    "## When to use this example",
    "## Inputs you bring",
    "## InvarLock transaction",
    "## What the result establishes",
    "## Interpretation boundary",
    "## Run it",
)
ALLOWED_SCENARIO_FILES = frozenset({"README.md", "scenario.yaml"})
ALLOWED_CATALOG_FILES = frozenset({"README.md", "scenario.schema.json"})
SCENARIO_CATEGORIES = frozenset({"changes", "imports", "journeys"})


def _is_ignored_metadata(path: Path) -> bool:
    return path.name == ".DS_Store" or path.name.startswith("._")


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"{path}: could not read YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one YAML object")
    return value


def _read_schema(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: could not read JSON schema: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: schema must be one JSON object")
    jsonschema.Draft202012Validator.check_schema(value)
    return value


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _validate_runbook(
    errors: list[str], *, manifest: Path, scenario: dict[str, Any], root: Path
) -> None:
    runbook = manifest.parent / "README.md"
    label = _relative(runbook, root)
    try:
        text = runbook.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        errors.append(f"{label}: could not read runbook: {exc}")
        return
    expected_title = f"# {scenario.get('title', '')}\n"
    if not text.startswith(expected_title):
        errors.append(f"{label}: runbook title must match scenario title")
    for section in REQUIRED_RUNBOOK_SECTIONS:
        if section not in text:
            errors.append(f"{label}: missing required section {section!r}")


def _validate_related_paths(
    errors: list[str], *, scenario: dict[str, Any], root: Path, manifest: Path
) -> None:
    workflow = scenario.get("workflow")
    if not isinstance(workflow, dict):
        return
    for value in workflow.get("related_paths", []):
        if not isinstance(value, str):
            continue
        candidate = root / value
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            errors.append(
                f"{_relative(manifest, root)}: related path is unavailable or "
                f"outside the repository: {value}"
            )
            continue
        if candidate.is_symlink() or not resolved.is_file():
            errors.append(
                f"{_relative(manifest, root)}: related path must be a regular "
                f"repository file: {value}"
            )


def _validate_scenario_files(errors: list[str], *, manifest: Path, root: Path) -> None:
    for candidate in sorted(manifest.parent.iterdir()):
        if _is_ignored_metadata(candidate):
            continue
        if (
            candidate.name not in ALLOWED_SCENARIO_FILES
            or candidate.is_symlink()
            or not candidate.is_file()
        ):
            errors.append(
                f"{_relative(candidate, root)}: unexpected scenario file; "
                "change creation and runtime machinery stay outside scenario recipes"
            )


def _validate_catalog_layout(
    errors: list[str], *, scenario_root: Path, root: Path
) -> None:
    for candidate in sorted(scenario_root.iterdir()):
        if _is_ignored_metadata(candidate):
            continue
        if candidate.name in ALLOWED_CATALOG_FILES:
            if candidate.is_symlink() or not candidate.is_file():
                errors.append(
                    f"{_relative(candidate, root)}: catalog document must be a "
                    "regular repository file"
                )
            continue
        if candidate.name not in SCENARIO_CATEGORIES:
            errors.append(
                f"{_relative(candidate, root)}: unexpected scenario catalog path; "
                "change creation and runtime machinery stay outside scenario recipes"
            )
            continue
        if candidate.is_symlink() or not candidate.is_dir():
            errors.append(
                f"{_relative(candidate, root)}: scenario category must be a regular "
                "repository directory"
            )

    for category in sorted(SCENARIO_CATEGORIES):
        category_path = scenario_root / category
        if category_path.is_symlink() or not category_path.is_dir():
            errors.append(
                f"{_relative(category_path, root)}: required scenario category is "
                "missing or invalid"
            )
            continue
        for candidate in sorted(category_path.iterdir()):
            if _is_ignored_metadata(candidate):
                continue
            if candidate.is_symlink() or not candidate.is_dir():
                errors.append(
                    f"{_relative(candidate, root)}: unexpected scenario category "
                    "entry; each entry must be a scenario directory"
                )
                continue
            if not (candidate / "scenario.yaml").is_file():
                errors.append(
                    f"{_relative(candidate, root)}: unregistered scenario directory; "
                    "scenario.yaml is required"
                )


def validate_repository(root: Path) -> tuple[list[str], list[dict[str, str]]]:
    root = root.resolve()
    scenario_root = root / "examples" / "scenarios"
    errors: list[str] = []
    summaries: list[dict[str, str]] = []
    try:
        schema = _read_schema(scenario_root / "scenario.schema.json")
    except (ValueError, jsonschema.SchemaError) as exc:
        return [str(exc)], summaries
    _validate_catalog_layout(errors, scenario_root=scenario_root, root=root)
    validator = jsonschema.Draft202012Validator(schema)
    identifiers: dict[str, Path] = {}
    manifests = sorted(scenario_root.glob("*/*/scenario.yaml"))
    if not manifests:
        return ["examples/scenarios: no scenario manifests found"], summaries

    for manifest in manifests:
        label = _relative(manifest, root)
        try:
            scenario = _read_object(manifest)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        schema_errors = sorted(
            validator.iter_errors(scenario), key=lambda item: list(item.path)
        )
        for error in schema_errors:
            location = ".".join(str(item) for item in error.path) or "<root>"
            errors.append(f"{label}: {location}: {error.message}")
        scenario_id = scenario.get("scenario_id")
        if isinstance(scenario_id, str):
            previous = identifiers.get(scenario_id)
            if previous is not None:
                errors.append(
                    f"{label}: duplicate scenario_id also used by "
                    f"{_relative(previous, root)}"
                )
            else:
                identifiers[scenario_id] = manifest
            if manifest.parent.name != scenario_id:
                errors.append(f"{label}: directory name must equal scenario_id")
        _validate_scenario_files(errors, manifest=manifest, root=root)
        _validate_runbook(errors, manifest=manifest, scenario=scenario, root=root)
        _validate_related_paths(errors, scenario=scenario, root=root, manifest=manifest)
        if isinstance(scenario_id, str):
            summaries.append(
                {
                    "scenario_id": scenario_id,
                    "title": str(scenario.get("title", "")),
                    "audience": str(scenario.get("audience", "")),
                    "availability": str(scenario.get("availability", "")),
                    "path": manifest.parent.relative_to(root).as_posix(),
                }
            )

    catalog = scenario_root / "README.md"
    try:
        catalog_text = catalog.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        errors.append(f"examples/scenarios/README.md: could not read catalog: {exc}")
    else:
        for scenario_id in sorted(identifiers):
            if f"`{scenario_id}`" not in catalog_text:
                errors.append(
                    f"examples/scenarios/README.md: missing catalog entry for "
                    f"{scenario_id}"
                )

    return errors, sorted(summaries, key=lambda item: item["scenario_id"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate example scenarios without executing external changes."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root containing examples/scenarios.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = parser.parse_args(argv)
    errors, scenarios = validate_repository(args.root)
    if args.json:
        print(
            json.dumps(
                {
                    "format_version": FORMAT_VERSION,
                    "ok": not errors,
                    "scenario_count": len(scenarios),
                    "scenarios": scenarios,
                    "errors": errors,
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
    else:
        print(f"Example scenario catalog OK ({len(scenarios)} scenarios)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
