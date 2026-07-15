from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / "contracts/architecture_policy.toml"
CHECKER_PATH = REPO_ROOT / "scripts/checks/check_architecture_fragmentation.py"


def _policy() -> dict[str, object]:
    with POLICY_PATH.open("rb") as handle:
        return tomllib.load(handle)


def test_architecture_limits_are_category_owned() -> None:
    policy = _policy()
    assert policy["schema"] == "invarlock/architecture-policy/v1"
    categories = policy["categories"]
    assert isinstance(categories, list)
    assert categories
    assert {category["name"] for category in categories} == {
        "operational_python",
        "operational_shell",
        "python_tests",
        "shell_tests",
        "source_python",
    }
    for category in categories:
        assert "path" not in category
        assert "paths" not in category
        assert category["soft_lines"] <= category["hard_lines"]
        assert category["soft_direct_files"] <= category["hard_direct_files"]


def test_architecture_policy_expresses_semantic_dependency_direction() -> None:
    policy = _policy()
    rules = policy["dependency_rules"]
    assert isinstance(rules, list)
    library_rule = next(
        rule for rule in rules if rule["name"] == "library_layers_do_not_depend_on_cli"
    )
    assert library_rule["forbid_import_prefixes"] == ["invarlock.cli"]
    assert library_rule["include"] == ["src/invarlock/**/*.py"]
    assert "src/invarlock/cli/**/*.py" in library_rule["exclude"]
    reporting_rule = next(
        rule for rule in rules if rule["name"] == "reporting_has_no_cli_output"
    )
    assert set(reporting_rule["forbid_call_prefixes"]) == {
        "rich.console.Console.print",
        "typer.echo",
    }
    rule_names = {rule["name"] for rule in rules}
    assert "cli_evaluation_phases_do_not_construct_reports" in rule_names
    assert "orchestrator_environment_does_not_execute" in rule_names
    assert "orchestrator_helpers_are_leaf_support" in rule_names


def test_architecture_checker_replaces_historical_topology_caps() -> None:
    result = subprocess.run(
        [sys.executable, str(CHECKER_PATH), "--json"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["format_version"] == "architecture-fragmentation-v1"
    assert payload["uncategorized_code"] == []
    assert payload["category_file_counts"]["source_python"] > 0
    assert payload["category_file_counts"]["operational_python"] > 0
    assert payload["category_file_counts"]["python_tests"] > 0
    assert result.returncode == (0 if payload["release_ready"] else 1)
