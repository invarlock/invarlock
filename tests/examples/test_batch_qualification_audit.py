from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/evaluator-qualification"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fixture(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    return load(
        "native_projection_fixtures",
        ROOT / "tests/examples/test_batch_native_projection.py",
    )


@pytest.mark.parametrize(
    "provider",
    ["promptfoo", "evidently", "langfuse", "azure-ai-evaluation", "pydantic-evals"],
)
def test_audit_exercises_boundaries_and_reports_candidate_without_authority(
    fixture, provider
):
    module = load(
        "batch_differential_test", EXAMPLE / "maintained/batch_differential.py"
    )
    corpus = json.loads((EXAMPLE / "maintained/batch-boundaries.json").read_bytes())

    def run(cases):
        return fixture.native(provider, cases)

    result = module.audit(provider, corpus, version="99.0.0", run=run)
    assert result["authority"] == "none" and result["candidate_dependency"]
    assert not result["semantic_drift"] and len(result["cases"]) == 34
    assert sum(case["supported"] for case in result["cases"]) == (
        23 if provider == "promptfoo" else 30
    )
    corpus["cases"][0]["supported_for"] = []
    assert module.audit(provider, corpus, version="99.0.0", run=run)["semantic_drift"]

    def malformed(cases):
        return {"metrics": {"exact_match": 1.0}}

    result = module.audit(provider, corpus, version="99.0.0", run=malformed)
    assert result["semantic_drift"] and result["native_rows"] == []


def test_audit_cli_executes_actual_adapter_and_refuses_overwrite(
    fixture, monkeypatch, tmp_path
):
    module = load(
        "batch_differential_cli_test", EXAMPLE / "maintained/batch_differential.py"
    )
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "0.7.21")

    def execute(provider, cases, **kwargs):
        assert provider == "evidently" and kwargs["version"] == "0.7.21"
        return fixture.native(provider, cases), None

    monkeypatch.setattr(module, "execute", execute)
    output = tmp_path / "audit.json"
    assert module.main(["--provider", "evidently", "--output", str(output)]) == 0
    assert not json.loads(output.read_bytes())["candidate_dependency"]
    with pytest.raises(FileExistsError):
        module.main(["--provider", "evidently", "--output", str(output)])
    monkeypatch.setattr(module, "execute", lambda *_, **__: ({}, None))
    assert (
        module.main(
            ["--provider", "promptfoo", "--output", str(tmp_path / "drift.json")]
        )
        == 2
    )


@pytest.mark.parametrize("failure", [None, "subprocess", "score"])
def test_qualification_publishes_only_complete_independently_qualified_outputs(
    fixture, monkeypatch, tmp_path, failure
):
    module = load("qualify_batch_test", EXAMPLE / "maintained/qualify_batch.py")
    import runner_support

    monkeypatch.setattr(
        runner_support.importlib.metadata, "version", lambda _: "0.7.21"
    )

    def run(command, **kwargs):
        assert kwargs["check"] and kwargs["env"]["PYTHONPATH"] == str(EXAMPLE)
        if failure == "subprocess":
            raise subprocess.CalledProcessError(1, command)
        args = argparse.Namespace(
            **{
                name.replace("-", "_"): Path(command[command.index("--" + name) + 1])
                for name in (
                    "profile",
                    "schedule",
                    "cases",
                    "dependency-lock",
                    "raw-output",
                    "export",
                )
            }
        )
        runner_support.finish_deterministic(
            args=args,
            entrypoint="test native orchestration",
            scores=[0.0 if failure == "score" else 1.0, 0.0],
            details=[{}, {}],
            environment=[],
        )

    monkeypatch.setattr(module.subprocess, "run", run)
    output = tmp_path / "qualification"
    if failure:
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            module.execute(
                provider="evidently",
                cases=EXAMPLE / "cases.json",
                schedule=EXAMPLE / "schedule.json",
                output=output,
            )
        assert not output.exists() and not list(tmp_path.iterdir())
    else:
        result = module.execute(
            provider="evidently",
            cases=EXAMPLE / "cases.json",
            schedule=EXAMPLE / "schedule.json",
            output=output,
        )
        assert result["profile_id"] == "evidently-strict-batch-v1"
        assert (output / "cases.json").read_bytes() == (
            EXAMPLE / "cases.json"
        ).read_bytes()
        with pytest.raises(ValueError, match="already exists"):
            module.execute(
                provider="evidently",
                cases=EXAMPLE / "cases.json",
                schedule=EXAMPLE / "schedule.json",
                output=output,
            )


def test_qualification_cli_reports_failure_and_success(
    fixture, monkeypatch, tmp_path, capsys
):
    module = load("qualify_batch_cli_test", EXAMPLE / "maintained/qualify_batch.py")

    def failure(**kwargs):
        raise ValueError("source mismatch")

    monkeypatch.setattr(module, "execute", failure)
    assert module.main(["--provider", "evidently", "--output", str(tmp_path)]) == 2
    assert "source mismatch" in capsys.readouterr().err
    monkeypatch.setattr(module, "execute", lambda **_: {"ok": True})
    assert module.main(["--provider", "evidently", "--output", str(tmp_path)]) == 0


@pytest.mark.parametrize(
    "provider",
    ["promptfoo", "evidently", "langfuse", "azure-ai-evaluation", "pydantic-evals"],
)
def test_retained_current_native_proofs_replay_all_historical_scores(fixture, provider):
    import matrix
    from maintained.batch_semantics import project
    from maintained.profile_binding import require_current_profile

    maintained = EXAMPLE / "maintained"
    definition = next(
        item
        for item in matrix.load(maintained / "batch-profiles.json")["profiles"]
        if item["historical_profile"] == provider
    )
    artifact = maintained / "artifacts" / definition["profile_id"]
    require_current_profile(matrix.load(artifact / "profile.json"), definition, EXAMPLE)
    replayed = matrix.qualify(
        definition, artifacts=artifact.parent, schedule=artifact / "schedule.json"
    ).as_dict()
    assert (
        matrix.canonical_json_bytes(replayed)
        == (artifact / "qualification-result.json").read_bytes()
    )
    historical = matrix.load(
        EXAMPLE / "authoritative/artifacts" / provider / "qualification-result.json"
    )
    assert replayed["record_count"] == 102
    assert replayed["scores"] == historical["scores"]
    assert replayed["records_sha256"] == historical["records_sha256"]
    cases = matrix.load(artifact / "cases.json")["records"]
    native = fixture.native(provider, cases)
    rows = fixture.rows_for(provider, native)
    rows[:] = [
        row["detail"]["native_row"]
        for row in matrix.load(artifact / "upstream-output.json")["records"]
    ]
    assert project(provider, cases, native)[0] == replayed["scores"]
    manifest = matrix.load(artifact / "execution-manifest.json")
    for name, digest in manifest["sources_sha256"].items():
        assert (
            "sha256:" + hashlib.sha256((maintained / name).read_bytes()).hexdigest()
            == digest
        )
    for name, digest in manifest["artifacts_sha256"].items():
        assert (
            "sha256:" + hashlib.sha256((artifact / name).read_bytes()).hexdigest()
            == digest
        )
    observation = matrix.load(artifact / "boundary-observation.json")
    assert observation["authority"] == "none" and not observation["semantic_drift"]
    assert len(observation["cases"]) == 34
    for path in artifact.iterdir():
        data = path.read_bytes().lower()
        assert b"/users/" not in data and b"/private/tmp/" not in data
