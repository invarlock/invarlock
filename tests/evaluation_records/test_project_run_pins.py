"""Reviewed project inputs cannot silently drift between comparisons."""

import json

import pytest
from typer.testing import CliRunner

from invarlock import captured_evidence_publication
from invarlock.cli import app
from invarlock.evaluation_comparison import comparison
from tests._evaluation_support import digest, load_run


def _project(tmp_path):
    runner = CliRunner()
    project = tmp_path / "project"
    assert runner.invoke(app, ["evaluate", "--init", str(project)]).exit_code == 0
    config = json.loads((project / "request.yaml").read_text())
    for side in ("baseline", "subject"):
        config["comparison"][side]["expected_run_digest"] = digest(
            load_run(project / f"inputs/{side}.json", adapter="invarlock")
        )
    return runner, project, config


def _compare(runner, project, config, *, output="result", signing_key=None):
    config["output"]["evidence"] = output
    request_path = project / "request.yaml"
    request_path.chmod(0o644)
    request_path.write_text(json.dumps(config))
    return runner.invoke(
        app,
        [
            "evaluate",
            str(project / "request.yaml"),
            *(
                ["--unsigned"]
                if signing_key is None
                else ["--signing-key", str(signing_key)]
            ),
            "--json",
        ],
    )


def test_matching_pins_accept_json_formatting_changes(tmp_path):
    runner, project, config = _project(tmp_path)
    path = project / "inputs/baseline.json"
    path.chmod(0o644)
    path.write_text(json.dumps(json.loads(path.read_text()), indent=4, sort_keys=True))
    result = _compare(runner, project, config)
    assert result.exit_code == 0, result.stdout
    evidence = json.loads((project / "result/manifest.json").read_text())
    for side in ("baseline", "subject"):
        assert (
            evidence["files"][side]["digest"]
            == config["comparison"][side]["expected_run_digest"]
        )


@pytest.mark.parametrize("side", ["baseline", "subject"])
@pytest.mark.parametrize("override", [False, True])
def test_pin_rejects_changed_records_before_signing_or_publication(
    tmp_path, monkeypatch, side, override
):
    runner, project, config = _project(tmp_path)
    original = project / f"inputs/{side}.json"
    run = json.loads(original.read_text())
    run["records"][0]["output"] = "changed after review"
    changed = project / "replacement.json" if override else original
    changed.parent.mkdir(parents=True, exist_ok=True)
    if changed == original:
        changed.chmod(0o644)
    changed.write_text(json.dumps(run))
    if override:
        config["comparison"][side]["path"] = "replacement.json"

    def forbidden_private_key(*args, **kwargs):
        pytest.fail("a rejected run pin must not load the private key")

    def forbidden_score(*args, **kwargs):
        pytest.fail("a rejected run pin must not score records")

    monkeypatch.setattr(
        captured_evidence_publication, "_private_key", forbidden_private_key
    )
    monkeypatch.setattr(comparison, "score", forbidden_score)
    result = _compare(
        runner, project, config, output="result-2", signing_key=tmp_path / "absent.pem"
    )
    assert result.exit_code == 2, result.stdout
    assert "run digest differs from expected pin" in result.stdout
    assert not (project / "result-2").exists()
    assert not (project / "result").exists()


def test_pin_checks_normalized_native_export_including_source_identity(tmp_path):
    runner, project, config = _project(tmp_path)
    normalized = json.loads((project / "inputs/baseline.json").read_text())
    raw = project / "inputs/baseline.jsonl"
    raw.write_text("\n".join(json.dumps(row) for row in normalized["records"]) + "\n")
    options = {
        "adapter": "jsonl",
        "source": {"name": "jsonl", "version": "1"},
        "run_id": normalized["run_id"],
        "artifact_digest": normalized["artifact_digest"],
        "score_provenance": normalized["score_provenance"],
    }
    config["comparison"]["baseline"] = {
        "path": "inputs/baseline.jsonl",
        **options,
        "expected_run_digest": digest(load_run(raw, **options)),
    }
    result = _compare(runner, project, config, output="result-2")
    assert result.exit_code == 0, result.stdout
    config["comparison"]["baseline"]["source"]["version"] = "2"
    result = _compare(runner, project, config)
    assert result.exit_code == 2
    assert "run digest differs from expected pin" in result.stdout
    assert not (project / "result").exists()


@pytest.mark.parametrize("invalid", [None, "abc", "sha256:" + "F" * 64])
def test_malformed_pin_is_integration_error(tmp_path, invalid):
    runner, project, config = _project(tmp_path)
    config["comparison"]["baseline"]["expected_run_digest"] = invalid
    result = _compare(runner, project, config)
    assert result.exit_code == 2
    assert not (project / "result").exists()


@pytest.mark.parametrize("side", ["baseline", "subject"])
def test_matching_override_retains_the_pin_and_succeeds(tmp_path, side):
    runner, project, config = _project(tmp_path)
    replacement = project / "reviewed-copy.json"
    replacement.write_bytes((project / f"inputs/{side}.json").read_bytes())
    config["comparison"][side]["path"] = replacement.name
    result = _compare(runner, project, config)
    assert result.exit_code == 0, result.stdout


@pytest.mark.parametrize("field", ["artifact_digest", "context"])
def test_same_outputs_with_changed_execution_identity_reject_old_pin(tmp_path, field):
    runner, project, config = _project(tmp_path)
    path = project / "inputs/subject.json"
    run = json.loads(path.read_text())
    if field == "artifact_digest":
        run[field] = "sha256:" + "f" * 64
    else:
        run["records"][0]["context"] = {"runtime": "changed"}
    path.chmod(0o644)
    path.write_text(json.dumps(run))
    result = _compare(runner, project, config)
    assert result.exit_code == 2
    assert "run digest differs from expected pin" in result.stdout
    assert not (project / "result").exists()
