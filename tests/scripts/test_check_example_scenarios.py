from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.checks.check_example_scenarios import main, validate_repository

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_SCENARIOS = {
    "evidence-handoff",
    "external-harness",
    "fine-tuned-checkpoint",
    "gguf-conversion",
    "hf-quantized-checkpoint",
    "model-upgrade",
    "multimodal-upgrade",
    "pruned-checkpoint",
    "serving-endpoint",
    "tensorrt-deployment",
}

RUNBOOK = """# {title}

## When to use this example

Use it for a bounded test.

## Inputs you bring

Bring immutable inputs.

## InvarLock transaction

Run the public transaction.

## What the result establishes

It establishes the bounded result.

## Interpretation boundary

Interpret only the selected records.

## Run it

Run the public commands.
"""


def _scenario_root(root: Path) -> Path:
    scenario_root = root / "examples" / "scenarios"
    scenario_root.mkdir(parents=True)
    for category in ("changes", "imports", "journeys"):
        (scenario_root / category).mkdir()
    (scenario_root / "scenario.schema.json").write_text(
        (REPO_ROOT / "examples" / "scenarios" / "scenario.schema.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    return scenario_root


def _scenario_value(
    scenario_id: str = "unexpected", *, related_paths: list[str] | None = None
) -> dict[str, object]:
    workflow: dict[str, object] = {
        "commands": ["evaluate", "verify", "report"],
        "runbook": "README.md",
    }
    if related_paths is not None:
        workflow["related_paths"] = related_paths
    return {
        "format_version": "invarlock/example-scenario-v1",
        "scenario_id": scenario_id,
        "title": "Unexpected",
        "audience": "Test operators",
        "availability": "operator_recipe",
        "input_boundary": {
            "kind": "model_upgrade",
            "supplied_by": "external_tool_or_team",
            "candidate_input": "Candidate checkpoint",
        },
        "transaction": {
            "task": "text_causal",
            "metric": "exact_match",
            "execution_mode": "run",
            "providers": ["hf_transformers"],
            "acceptance_question": "Is the candidate acceptable?",
        },
        "workflow": workflow,
        "interpretation": {
            "establishes": ["A bounded result."],
            "boundary": ["Only the selected schedule."],
        },
    }


def _write_scenario(
    root: Path,
    *,
    directory: str = "unexpected",
    scenario_id: str = "unexpected",
    related_paths: list[str] | None = None,
    runbook: str | None = None,
) -> Path:
    example = root / "examples" / "scenarios" / "changes" / directory
    example.mkdir(parents=True)
    (example / "scenario.yaml").write_text(
        yaml.safe_dump(
            _scenario_value(scenario_id, related_paths=related_paths),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (example / "README.md").write_text(
        runbook if runbook is not None else RUNBOOK.format(title="Unexpected"),
        encoding="utf-8",
    )
    return example


def _write_catalog(root: Path, *scenario_ids: str) -> None:
    catalog = root / "examples" / "scenarios" / "README.md"
    catalog.write_text(
        "# Catalog\n\n" + "\n".join(f"- `{value}`" for value in scenario_ids),
        encoding="utf-8",
    )


def test_repository_scenario_catalog_is_closed_and_complete(
    capsys: pytest.CaptureFixture[str],
) -> None:
    errors, scenarios = validate_repository(REPO_ROOT)

    assert errors == []
    assert len(scenarios) == len(EXPECTED_SCENARIOS)
    assert {item["scenario_id"] for item in scenarios} == EXPECTED_SCENARIOS
    assert {item["availability"] for item in scenarios} == {
        "operator_recipe",
        "runnable",
    }
    assert main(["--root", str(REPO_ROOT), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["scenario_count"] == len(EXPECTED_SCENARIOS)


def test_every_scenario_declares_the_external_input_boundary() -> None:
    manifests = sorted((REPO_ROOT / "examples" / "scenarios").glob("**/scenario.yaml"))

    assert len(manifests) == len(EXPECTED_SCENARIOS)
    for manifest in manifests:
        value = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        assert value["input_boundary"]["supplied_by"] == "external_tool_or_team"
        assert value["input_boundary"]["candidate_input"]
        assert value["transaction"]["acceptance_question"].endswith("?")
        assert value["interpretation"]["establishes"]
        assert value["interpretation"]["boundary"]


def test_scenario_checker_rejects_unexpected_files(tmp_path: Path) -> None:
    _scenario_root(tmp_path)
    example = _write_scenario(tmp_path)
    _write_catalog(tmp_path, "unexpected")
    (example / "build-model.py").write_text("print('not allowed')\n", encoding="utf-8")

    errors, _scenarios = validate_repository(tmp_path)

    assert any("unexpected scenario file" in error for error in errors)


def test_scenario_checker_closes_catalog_and_category_roots(tmp_path: Path) -> None:
    scenario_root = _scenario_root(tmp_path)
    _write_scenario(tmp_path)
    _write_catalog(tmp_path, "unexpected")
    (scenario_root / "build-model.py").write_text(
        "print('not allowed')\n", encoding="utf-8"
    )
    (scenario_root / "changes" / "convert.py").write_text(
        "print('not allowed')\n", encoding="utf-8"
    )
    (scenario_root / "changes" / "unregistered").mkdir()

    errors, _scenarios = validate_repository(tmp_path)

    combined = "\n".join(errors)
    assert "unexpected scenario catalog path" in combined
    assert "unexpected scenario category entry" in combined
    assert "unregistered scenario directory" in combined


def test_scenario_checker_reports_contract_and_runbook_drift(tmp_path: Path) -> None:
    _scenario_root(tmp_path)
    missing = _write_scenario(
        tmp_path,
        directory="wrong-directory",
        scenario_id="duplicate",
        related_paths=["missing.txt"],
        runbook="# Wrong title\n",
    )
    duplicate = _write_scenario(
        tmp_path,
        directory="duplicate",
        scenario_id="duplicate",
    )
    value = yaml.safe_load((duplicate / "scenario.yaml").read_text(encoding="utf-8"))
    value["unknown"] = True
    (duplicate / "scenario.yaml").write_text(
        yaml.safe_dump(value, sort_keys=False), encoding="utf-8"
    )
    _write_catalog(tmp_path)

    errors, scenarios = validate_repository(tmp_path)

    assert len(scenarios) == 2
    combined = "\n".join(errors)
    assert "duplicate scenario_id" in combined
    assert "directory name must equal scenario_id" in combined
    assert "Additional properties are not allowed" in combined
    assert "runbook title must match scenario title" in combined
    assert "missing required section" in combined
    assert "related path is unavailable" in combined
    assert "missing catalog entry" in combined
    assert missing.is_dir()


def test_scenario_checker_rejects_invalid_schema_and_yaml(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scenario_root = tmp_path / "examples" / "scenarios"
    scenario_root.mkdir(parents=True)
    (scenario_root / "scenario.schema.json").write_text("{", encoding="utf-8")

    errors, scenarios = validate_repository(tmp_path)

    assert scenarios == []
    assert "could not read JSON schema" in errors[0]
    assert main(["--root", str(tmp_path)]) == 1
    assert "ERROR:" in capsys.readouterr().err

    (scenario_root / "scenario.schema.json").write_text(
        (REPO_ROOT / "examples/scenarios/scenario.schema.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    manifest = scenario_root / "changes" / "broken" / "scenario.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("- not-an-object\n", encoding="utf-8")
    _write_catalog(tmp_path)

    errors, scenarios = validate_repository(tmp_path)

    assert scenarios == []
    assert any("expected one YAML object" in error for error in errors)
