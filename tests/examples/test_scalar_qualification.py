from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/evaluator-qualification"
MAINTAINED = EXAMPLE / "maintained"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def definitions(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    return json.loads((MAINTAINED / "scalar-profiles.json").read_bytes())["profiles"]


def test_scalar_runner_binds_sources_and_freezes_each_input(
    definitions, monkeypatch, tmp_path
):
    import matrix
    import runner_support

    module = load("scalar_runner_test", MAINTAINED / "scalar_runner.py")
    definition = definitions[0]
    profile = matrix.write_profile(definition, artifacts=tmp_path / "artifacts")
    cases = tmp_path / "cases.json"
    cases.write_bytes((EXAMPLE / "cases.json").read_bytes())
    args = argparse.Namespace(
        profile=profile,
        cases=cases,
        schedule=EXAMPLE / "schedule.json",
        dependency_lock=EXAMPLE / definition["lock"],
        export=profile.parent / "export.json",
        raw_output=profile.parent / "upstream-output.json",
    )
    monkeypatch.setattr(module, "arguments", lambda: args)
    monkeypatch.setattr(
        runner_support.importlib.metadata,
        "version",
        lambda _: definition["upstream"]["version"],
    )
    calls = []

    def score(case):
        calls.append(case["record_id"])
        cases.write_text('{"records":[]}')
        return {"score": float(case["output"] == case["reference"])}

    monkeypatch.setattr(
        module, "build_scorer", lambda _: (score, definition["source_bindings"])
    )
    module.main()
    assert calls == ["record-1", "record-2"]
    assert matrix.qualify(
        definition, artifacts=profile.parent.parent, schedule=args.schedule
    ).scores == (1.0, 0.0)
    cases.write_bytes((EXAMPLE / "cases.json").read_bytes())
    monkeypatch.setattr(
        module, "build_scorer", lambda _: (score, {"module_sha256": "changed"})
    )
    with pytest.raises(ValueError, match="native source"):
        module.main()
    monkeypatch.setattr(
        module, "CONFIGURATIONS", {**module.CONFIGURATIONS, "lm-evaluation-harness": {}}
    )
    with pytest.raises(ValueError, match="configuration"):
        module.main()
    changed = matrix.load(profile)
    changed["profile_id"] = "lm-evaluation-harness"
    profile.write_bytes(matrix.canonical_json_bytes(changed))
    with pytest.raises(ValueError, match="separate current"):
        module.main()


@pytest.mark.parametrize(
    "provider",
    [
        "lm-evaluation-harness",
        "deepeval",
        "ragas",
        "lighteval",
        "hugging-face-evaluate",
        "autoevals",
        "openevals",
        "openai-evals",
        "arize-phoenix-evals",
        "opik",
        "trulens",
    ],
)
def test_scalar_differential_checks_current_and_historical_boundaries(
    definitions, provider
):
    module = load("scalar_differential_test", MAINTAINED / "scalar_differential.py")
    fixtures = load(
        "scalar_result_fixtures",
        ROOT / "tests/examples/test_scalar_evaluator_semantics.py",
    )
    definition = next(d for d in definitions if d["historical_profile"] == provider)
    corpus = json.loads((MAINTAINED / "scalar-boundaries.json").read_bytes())

    def current(case):
        return fixtures.result(provider, case["output"] == case["reference"])

    def historical(case):
        override = case["historical_overrides"].get(
            provider, {"score": float(case["output"] == case["reference"])}
        )
        if "error_type" in override:
            raise type(override["error_type"], (Exception,), {})(
                "expected native boundary"
            )
        return fixtures.result(provider, bool(override["score"]))

    args = {
        "definition": definition,
        "corpus": corpus,
        "current": current,
        "historical": historical,
        "version": definition["upstream"]["version"],
        "sources": definition["source_bindings"],
    }
    observation = module.audit(**args)
    assert observation["authority"] == "none" and not observation["semantic_drift"]
    assert len(observation["cases"]) == 39
    assert module.audit(**(args | {"version": "99.0.0"}))["candidate_dependency"]
    assert module.audit(**(args | {"sources": {}}))["semantic_drift"]
    corpus["cases"][0]["supported_for"] = []
    assert module.audit(**args)["semantic_drift"]

    def wrong(case):
        return {"score": "1"}

    assert module.audit(**(args | {"current": wrong, "historical": wrong}))[
        "semantic_drift"
    ]

    def failure(case):
        raise RuntimeError("native failure")

    assert module.audit(**(args | {"current": failure}))["semantic_drift"]
    changed = copy.deepcopy(definition)
    changed["scorer_configuration"] = {}
    with pytest.raises(ValueError, match="configuration"):
        module.audit(**(args | {"definition": changed}))
    corpus["cases"][1]["record_id"] = corpus["cases"][0]["record_id"]
    with pytest.raises(ValueError, match="unique nonempty"):
        module.audit(**args)


def test_scalar_audit_cli_reports_drift_and_refuses_overwrite(
    definitions, monkeypatch, tmp_path
):
    module = load("scalar_audit_cli_test", MAINTAINED / "scalar_differential.py")
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "0.4.12")
    monkeypatch.setattr(module, "build_scorer", lambda *_, **__: (lambda _: {}, {}))
    output = tmp_path / "observation.json"
    assert (
        module.main(["--provider", "lm-evaluation-harness", "--output", str(output)])
        == 2
    )
    assert json.loads(output.read_bytes())["authority"] == "none"
    with pytest.raises(FileExistsError):
        module.main(["--provider", "lm-evaluation-harness", "--output", str(output)])


@pytest.mark.parametrize("race", [False, True])
def test_current_scalar_qualification_uses_atomic_no_replace(
    definitions, monkeypatch, tmp_path, race
):
    import runner_support

    module = load("scalar_qualify_test", MAINTAINED / "qualify_scalar.py")
    monkeypatch.setattr(
        runner_support.importlib.metadata, "version", lambda _: "0.4.12"
    )
    output = tmp_path / "qualified"

    def run(command, **kwargs):
        assert kwargs["env"]["UV_PYTHON"] == sys.executable
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
            entrypoint="native scalar test",
            scores=[1.0, 0.0],
            details=[{}, {}],
            environment=[],
        )
        if race:
            output.mkdir()

    monkeypatch.setattr(module.subprocess, "run", run)
    if race:
        with pytest.raises(OSError) as error:
            module.execute(
                provider="lm-evaluation-harness",
                cases=EXAMPLE / "cases.json",
                schedule=EXAMPLE / "schedule.json",
                output=output,
            )
        assert error.value.errno == 17
        assert output.is_dir() and list(output.iterdir()) == []
    else:
        assert module.execute(
            provider="lm-evaluation-harness",
            cases=EXAMPLE / "cases.json",
            schedule=EXAMPLE / "schedule.json",
            output=output,
        )["scores"] == [1.0, 0.0]
    with pytest.raises(ValueError, match="already exists"):
        module.execute(
            provider="lm-evaluation-harness",
            cases=EXAMPLE / "cases.json",
            schedule=EXAMPLE / "schedule.json",
            output=output,
        )
    monkeypatch.setattr(module, "execute", lambda **_: {"ok": True})
    assert (
        module.main(["--provider", "lm-evaluation-harness", "--output", str(output)])
        == 0
    )

    def failure(**_):
        raise ValueError("unsupported source")

    monkeypatch.setattr(module, "execute", failure)
    assert (
        module.main(["--provider", "lm-evaluation-harness", "--output", str(output)])
        == 2
    )


@pytest.mark.parametrize(
    "provider",
    [
        "lm-evaluation-harness",
        "deepeval",
        "ragas",
        "lighteval",
        "hugging-face-evaluate",
        "autoevals",
        "openevals",
        "openai-evals",
        "arize-phoenix-evals",
        "opik",
        "trulens",
    ],
)
def test_retained_current_scalar_proofs_bind_native_calls_and_replay_all_scores(
    definitions, provider
):
    import matrix
    from maintained.profile_binding import require_current_profile
    from maintained.scalar_semantics import validate_result

    definition = next(d for d in definitions if d["historical_profile"] == provider)
    artifact = MAINTAINED / "artifacts" / definition["profile_id"]
    require_current_profile(matrix.load(artifact / "profile.json"), definition, EXAMPLE)
    result = matrix.qualify(
        definition, artifacts=artifact.parent, schedule=artifact / "schedule.json"
    ).as_dict()
    assert (
        matrix.canonical_json_bytes(result)
        == (artifact / "qualification-result.json").read_bytes()
    )
    historical = matrix.load(
        EXAMPLE / "authoritative/artifacts" / provider / "qualification-result.json"
    )
    assert result["record_count"] == 102
    assert result["scores"] == historical["scores"]
    assert result["records_sha256"] == historical["records_sha256"]
    cases = matrix.load(artifact / "cases.json")["records"]
    rows = matrix.load(artifact / "upstream-output.json")["records"]
    scores = []
    for case, row in zip(cases, rows, strict=True):
        assert row["record_id"] == case["record_id"]
        assert row["detail"]["source_bindings"] == definition["source_bindings"]
        scores.append(validate_result(provider, case, row["detail"]["native"]))
    assert scores == result["scores"]
    manifest = matrix.load(artifact / "execution-manifest.json")
    assert len(manifest["sources_sha256"]) == 9
    assert len(manifest["artifacts_sha256"]) == 7
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
    observation = matrix.load(artifact / "boundary-observation.json")
    assert observation["authority"] == "none" and not observation["semantic_drift"]
    assert len(observation["cases"]) == 39
    for path in artifact.iterdir():
        data = path.read_bytes().lower()
        assert b"/users/" not in data and b"/private/tmp/" not in data
