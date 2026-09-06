from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/evaluator-qualification"
MAINTAINED = EXAMPLE / "maintained"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, MAINTAINED / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _upstream(monkeypatch, score):
    calls = []

    def match(**configuration):
        calls.append(configuration)
        return score

    modules = {
        name: ModuleType(name)
        for name in (
            "inspect_ai",
            "inspect_ai.model",
            "inspect_ai.scorer",
            "inspect_ai.solver",
        )
    }
    modules["inspect_ai.model"].ChatMessageUser = lambda **kwargs: SimpleNamespace(
        **kwargs
    )
    modules["inspect_ai.model"].ModelOutput = SimpleNamespace(
        from_content=lambda **kwargs: SimpleNamespace(completion=kwargs["content"])
    )
    modules["inspect_ai.scorer"].Target = lambda value: value
    modules["inspect_ai.scorer"].match = match
    modules["inspect_ai.solver"].TaskState = lambda **kwargs: SimpleNamespace(**kwargs)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.syspath_prepend(str(EXAMPLE))
    return calls


def test_versioned_runner_uses_native_results_and_rejects_other_profiles(monkeypatch):
    observed = []

    async def score(state, target):
        observed.append(state.sample_id)
        output = state.output.completion
        return SimpleNamespace(
            value="C" if output == target else "I",
            answer=output.strip(),
            explanation=output,
        )

    calls = _upstream(monkeypatch, score)
    runner = _load("inspect_literal_runner_test", "inspect_literal_runner.py")
    cases = [
        {
            "record_id": "one",
            "input": "question",
            "output": " answer",
            "reference": " answer",
        },
        {"record_id": "two", "input": "question", "output": "x", "reference": "y"},
    ]
    profile = {"profile_id": runner.PROFILE_ID}
    monkeypatch.setattr(runner, "arguments", lambda: "args")
    monkeypatch.setattr(runner, "load_inputs", lambda _: (profile, {}, cases))
    checked = []
    monkeypatch.setattr(runner, "require_profile_package", lambda p: checked.append(p))
    finished = []
    monkeypatch.setattr(
        runner, "finish_deterministic", lambda **kwargs: finished.append(kwargs)
    )
    asyncio.run(runner.run())
    assert calls == [{"location": "exact", "ignore_case": False, "numeric": False}]
    assert observed == ["one", "two"] and checked == [profile]
    assert finished[0]["scores"] == [1.0, 0.0]
    assert finished[0]["details"][0] == {"answer": "answer", "score_value": "C"}
    profile["profile_id"] = "inspect-ai"
    with pytest.raises(ValueError, match="separate versioned profile"):
        asyncio.run(runner.run())
    profile["profile_id"] = runner.PROFILE_ID
    cases[0]["reference"] = "answer"
    with pytest.raises(ValueError, match="normalization collision"):
        asyncio.run(runner.run())
    assert observed == ["one", "two"]


def test_differential_reports_domain_and_native_drift_without_authority():
    module = _load("inspect_differential_test", "inspect_differential.py")
    corpus = json.loads((MAINTAINED / "inspect-boundaries.json").read_bytes())
    by_id = {case["id"]: case for case in corpus["cases"]}

    async def score(case):
        native = by_id[case["record_id"]]
        return SimpleNamespace(
            value="C" if native["native_correct"] else "I",
            answer=case["output"].strip(),
            explanation=case["output"],
        )

    result = asyncio.run(module.audit(corpus, score, version="0.3.254"))
    assert not result["semantic_drift"]
    assert len(result["cases"]) == 28 and result["authority"] == "none"
    candidate = asyncio.run(module.audit(corpus, score, version="99.0.0"))
    assert candidate["candidate_dependency"] and candidate["authority"] == "none"
    corpus["cases"][0]["supported"] = False

    async def changed(case):
        return SimpleNamespace(value="I", answer="changed", explanation=case["output"])

    result = asyncio.run(module.audit(corpus, changed, version="99.0.0"))
    assert result["semantic_drift"]
    assert "supported pair domain changed" in result["cases"][0]["problems"]
    corpus["scorer_configuration"]["ignore_case"] = True
    with pytest.raises(ValueError, match="scorer configuration changed"):
        asyncio.run(module.audit(corpus, score, version="99.0.0"))


def test_differential_cli_runs_boundary_corpus_and_refuses_overwrite(
    tmp_path, monkeypatch
):
    corpus = json.loads((MAINTAINED / "inspect-boundaries.json").read_bytes())
    by_id = {case["id"]: case for case in corpus["cases"]}

    async def score(state, target):
        case = by_id[state.sample_id]
        assert target == case["reference"]
        return SimpleNamespace(
            value="C" if case["native_correct"] else "I",
            answer=state.output.completion.strip(),
            explanation=state.output.completion,
        )

    _upstream(monkeypatch, score)
    module = _load("inspect_differential_cli_test", "inspect_differential.py")
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "0.3.254")
    output = tmp_path / "observation.json"
    assert module.main(["--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["authority"] == "none"
    with pytest.raises(FileExistsError):
        module.main(["--output", str(output)])
    by_id["plain_equal"]["native_correct"] = False
    assert module.main(["--output", str(tmp_path / "drift.json")]) == 2


def test_fresh_qualification_replays_projection_and_cleans_failed_output(
    tmp_path, monkeypatch
):
    module = _load("qualify_inspect_test", "qualify_inspect.py")
    import runner_support

    monkeypatch.setattr(
        runner_support.importlib.metadata, "version", lambda _: "0.3.254"
    )
    bad_score = False

    def run(command, **kwargs):
        assert kwargs["check"] is True
        assert kwargs["env"]["PYTHONPATH"] == str(EXAMPLE)
        args = argparse.Namespace(
            **{
                name.replace("-", "_"): Path(command[command.index("--" + name) + 1])
                for name in (
                    "cases",
                    "dependency-lock",
                    "export",
                    "profile",
                    "raw-output",
                    "schedule",
                )
            }
        )
        runner_support.finish_deterministic(
            args=args,
            entrypoint="inspect_ai.scorer.match",
            scores=[0.0 if bad_score else 1.0, 0.0],
            details=[{"score_value": "C"}, {"score_value": "I"}],
            environment=[],
        )

    monkeypatch.setattr(module.subprocess, "run", run)
    output = tmp_path / "qualified"
    result = module.execute(
        cases=EXAMPLE / "cases.json", schedule=EXAMPLE / "schedule.json", output=output
    )
    assert result["profile_id"] == "inspect-ai-literal-pairs-v1"
    assert result["scores"] == [1.0, 0.0]
    assert (output / "schedule.json").read_bytes() == (
        EXAMPLE / "schedule.json"
    ).read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        module.execute(
            cases=EXAMPLE / "cases.json",
            schedule=EXAMPLE / "schedule.json",
            output=output,
        )
    bad_score = True
    with pytest.raises(ValueError, match="score"):
        module.execute(
            cases=EXAMPLE / "cases.json",
            schedule=EXAMPLE / "schedule.json",
            output=tmp_path / "invalid",
        )
    assert not (tmp_path / "invalid").exists()
    assert not list(tmp_path.glob(".inspect-qualification-*"))


def test_fresh_qualification_cli_success_and_upstream_failure(tmp_path, monkeypatch):
    module = _load("qualify_inspect_cli_test", "qualify_inspect.py")
    monkeypatch.setattr(
        module, "execute", lambda **kwargs: {"outcome": "qualified_for_import"}
    )
    assert module.main(["--output", str(tmp_path / "new")]) == 0

    def failed(**kwargs):
        raise subprocess.CalledProcessError(2, ["upstream"])

    monkeypatch.setattr(module, "execute", failed)
    assert module.main(["--output", str(tmp_path / "new")]) == 2


@pytest.mark.parametrize("record_id", ["", None, []])
def test_invalid_ids_are_explicitly_rejected(record_id):
    module = _load("inspect_invalid_ids_test", "inspect_semantics.py")
    with pytest.raises(ValueError, match="unique nonempty record IDs"):
        module.validate_cases(
            [{"record_id": record_id, "output": "a", "reference": "a"}]
        )


def test_retained_literal_profile_replays_without_changing_historical_outputs():
    module = _load("retained_inspect_literal_test", "qualify_inspect.py")
    artifacts = MAINTAINED / "artifacts"
    artifact = artifacts / "inspect-ai-literal-pairs-v1"
    definition = module.matrix.load(MAINTAINED / "inspect-profile.json")
    expected_profile = module.matrix.canonical_json_bytes(
        module.matrix.qualification_profile(definition)
    )
    assert (artifact / "profile.json").read_bytes() == expected_profile
    replayed = module.matrix.qualify(
        definition, artifacts=artifacts, schedule=artifact / "schedule.json"
    ).as_dict()
    assert (
        artifact / "qualification-result.json"
    ).read_bytes() == module.matrix.canonical_json_bytes(replayed)
    historical = module.matrix.load(
        EXAMPLE / "authoritative/artifacts/inspect-ai/qualification-result.json"
    )
    assert replayed["record_count"] == 102
    assert replayed["scores"] == historical["scores"]
    assert replayed["records_sha256"] == historical["records_sha256"]
    manifest = module.matrix.load(artifact / "execution-manifest.json")
    for name, digest in manifest["sources_sha256"].items():
        assert (
            "sha256:" + hashlib.sha256((MAINTAINED / name).read_bytes()).hexdigest()
            == digest
        )
    for name, digest in manifest["artifacts_sha256"].items():
        assert (
            "sha256:" + hashlib.sha256((artifact / name).read_bytes()).hexdigest()
            == digest
        )
    observation = module.matrix.load(artifact / "boundary-observation.json")
    assert observation["authority"] == "none" and not observation["semantic_drift"]
    assert len(observation["cases"]) == 28
    for path in artifact.iterdir():
        data = path.read_bytes().lower()
        assert b"/users/" not in data and b"/private/tmp/" not in data
