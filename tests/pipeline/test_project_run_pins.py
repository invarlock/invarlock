"""Reviewed project inputs cannot silently drift between comparisons."""

import json

import pytest
from typer.testing import CliRunner

from invarlock.pipeline.adapters import load_run
from invarlock.pipeline.cli import app
from invarlock.pipeline.contracts import digest


def _project(tmp_path):
    runner = CliRunner()
    project = tmp_path / "project"
    assert runner.invoke(app, ["init", str(project)]).exit_code == 0
    config = json.loads((project / "pipeline.json").read_text())
    for side in ("baseline", "candidate"):
        config[side]["expected_run_digest"] = digest(
            load_run(project / f"{side}.json", adapter="invarlock")
        )
    return runner, project, config


def _compare(runner, project, config, *options):
    (project / "pipeline.json").write_text(json.dumps(config))
    return runner.invoke(
        app,
        [
            "compare",
            str(project / "pipeline.json"),
            "--output",
            str(project / "result"),
            *options,
        ],
    )


def test_matching_pins_accept_json_formatting_changes(tmp_path):
    runner, project, config = _project(tmp_path)
    path = project / "baseline.json"
    path.write_text(json.dumps(json.loads(path.read_text()), indent=4, sort_keys=True))
    result = _compare(runner, project, config)
    assert result.exit_code == 0, result.stdout
    evidence = json.loads((project / "result/evidence.json").read_text())
    for side in ("baseline", "candidate"):
        assert (
            evidence["comparison"]["bindings"][side]
            == config[side]["expected_run_digest"]
        )


@pytest.mark.parametrize("side", ["baseline", "candidate"])
@pytest.mark.parametrize("override", [False, True])
def test_pin_rejects_changed_records_before_signing_or_publication(
    tmp_path, side, override
):
    runner, project, config = _project(tmp_path)
    original = project / f"{side}.json"
    run = json.loads(original.read_text())
    run["records"][0]["output"] = "changed after review"
    changed = tmp_path / "replacement.json" if override else original
    changed.write_text(json.dumps(run))
    options = [f"--{side}", str(changed)] if override else []
    # A pin mismatch must be detected before even opening the private key.
    result = _compare(
        runner, project, config, *options, "--signing-key", str(tmp_path / "absent.pem")
    )
    assert result.exit_code == 2
    assert "run digest does not match expected_run_digest" in result.stdout
    assert not (project / "result").exists()


def test_pin_checks_normalized_native_export_including_source_identity(tmp_path):
    runner, project, config = _project(tmp_path)
    normalized = json.loads((project / "baseline.json").read_text())
    raw = project / "baseline.jsonl"
    raw.write_text("\n".join(json.dumps(row) for row in normalized["records"]) + "\n")
    options = {
        "adapter": "jsonl",
        "source": {"name": "jsonl", "version": "1"},
        "run_id": normalized["run_id"],
        "artifact_digest": normalized["artifact_digest"],
        "score_provenance": normalized["score_provenance"],
    }
    config["baseline"] = {
        "path": raw.name,
        **options,
        "expected_run_digest": digest(load_run(raw, **options)),
    }
    result = _compare(runner, project, config)
    assert result.exit_code == 0, result.stdout
    config["baseline"]["source"]["version"] = "2"
    result = _compare(runner, project, config)
    assert result.exit_code == 2
    assert "run digest does not match expected_run_digest" in result.stdout


@pytest.mark.parametrize("invalid", [None, "abc", "sha256:" + "F" * 64])
def test_malformed_pin_is_integration_error(tmp_path, invalid):
    runner, project, config = _project(tmp_path)
    config["baseline"]["expected_run_digest"] = invalid
    result = _compare(runner, project, config)
    assert result.exit_code == 2
    assert not (project / "result").exists()


@pytest.mark.parametrize("side", ["baseline", "candidate"])
def test_matching_override_retains_the_pin_and_succeeds(tmp_path, side):
    runner, project, config = _project(tmp_path)
    replacement = tmp_path / "reviewed-copy.json"
    replacement.write_bytes((project / f"{side}.json").read_bytes())
    result = _compare(runner, project, config, f"--{side}", str(replacement))
    assert result.exit_code == 0, result.stdout


@pytest.mark.parametrize("field", ["artifact_digest", "context"])
def test_same_outputs_with_changed_execution_identity_reject_old_pin(tmp_path, field):
    runner, project, config = _project(tmp_path)
    path = project / "candidate.json"
    run = json.loads(path.read_text())
    if field == "artifact_digest":
        run[field] = "sha256:" + "f" * 64
    else:
        run["records"][0]["context"] = {"runtime": "changed"}
    path.write_text(json.dumps(run))
    result = _compare(runner, project, config)
    assert result.exit_code == 2
    assert "run digest does not match expected_run_digest" in result.stdout
    assert not (project / "result").exists()
