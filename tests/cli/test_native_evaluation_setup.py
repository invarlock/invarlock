"""The native judge starter is available through installed package resources."""

import json
from importlib import resources
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.public_contracts import load_evaluation_request_schema

RUNNER = CliRunner()
FILES = {"request.yaml", "judge-policy.json", "cases.jsonl", "README.md"}


def test_native_judge_setup_uses_complete_packaged_starter(tmp_path):
    destination = tmp_path / "native"
    result = RUNNER.invoke(
        app,
        ["evaluate", "--init", str(destination), "--example", "native-judge", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] and payload["details"]["example"] == "native-judge"
    assert {path.name for path in destination.iterdir()} == FILES
    packaged = resources.files("invarlock").joinpath(
        "_data", "examples", "native-judge"
    )
    for name in FILES:
        assert (destination / name).read_bytes() == packaged.joinpath(name).read_bytes()
    request = yaml.safe_load((destination / "request.yaml").read_text())
    Draft202012Validator(load_evaluation_request_schema()).validate(request)
    assert request["execution"]["mode"] == "run"
    assert request["comparison"]["metric"] == "judge"
    assert request["comparison"]["judge"]["workspace"]
    policy = json.loads((destination / "judge-policy.json").read_text())
    assert policy["format"] == "invarlock/native-judge-policy-v1"
    assert "baseline_run_sha256" not in policy["plan"]
    assert "plan_sha256" not in policy["analysis"]
    assert "expected_trials" not in policy["plan"]["schedule"]
    assert all(
        json.loads(line)
        for line in (destination / "cases.jsonl").read_text().splitlines()
    )
    assert not (destination / "artifacts").exists()


def test_native_judge_setup_never_overwrites_user_files(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "request.yaml"
    marker.write_text("user material")
    result = RUNNER.invoke(
        app,
        ["evaluate", "--init", str(destination), "--example", "native-judge", "--json"],
    )
    assert result.exit_code == 2
    assert marker.read_text() == "user material"
    assert {path.name for path in destination.iterdir()} == {"request.yaml"}


def test_packaged_native_starter_matches_maintained_example():
    packaged = resources.files("invarlock").joinpath(
        "_data", "examples", "native-judge"
    )
    maintained = Path(__file__).parents[2] / "examples" / "native-judge"
    for name in FILES:
        assert packaged.joinpath(name).read_bytes() == (maintained / name).read_bytes()


def test_recorded_judge_setup_stays_explicitly_captured(tmp_path):
    destination = tmp_path / "recorded"
    result = RUNNER.invoke(
        app, ["evaluate", "--init", str(destination), "--example", "judge", "--json"]
    )
    assert result.exit_code == 0, result.output
    request = yaml.safe_load((destination / "request.yaml").read_text())
    assert request["execution"]["mode"] == "captured"
    assert "recorded ratings" in (destination / "README.txt").read_text()


def test_missing_packaged_starter_fails_without_a_partial_directory(
    tmp_path, monkeypatch
):
    from invarlock.cli import evaluation_setup

    missing_package = tmp_path / "missing-package"
    missing_package.mkdir()
    monkeypatch.setattr(
        evaluation_setup.resources, "files", lambda _package: missing_package
    )
    destination = tmp_path / "native"
    result = RUNNER.invoke(
        app,
        ["evaluate", "--init", str(destination), "--example", "native-judge", "--json"],
    )
    assert result.exit_code == 2
    assert json.loads(result.stdout)["ok"] is False
    assert not destination.exists()


@pytest.mark.parametrize("example", ["classification", "extraction"])
def test_captured_starters_remain_available(tmp_path, example):
    destination = tmp_path / example
    result = RUNNER.invoke(
        app, ["evaluate", "--init", str(destination), "--example", example, "--json"]
    )
    assert result.exit_code == 0, result.output
    assert (
        yaml.safe_load((destination / "request.yaml").read_text())["execution"]["mode"]
        == "captured"
    )
    assert "recorded ratings" not in (destination / "README.txt").read_text()


def test_unknown_starter_names_do_not_publish(tmp_path):
    destination = tmp_path / "unsupported"
    result = RUNNER.invoke(
        app,
        ["evaluate", "--init", str(destination), "--example", "unsupported", "--json"],
    )
    assert result.exit_code == 2
    assert "native-judge" in json.loads(result.stdout)["errors"][0]
    assert not destination.exists()
