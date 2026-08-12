from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from invarlock.evidence_pack_contract import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "evaluator-qualification"
AUTHORITATIVE = EXAMPLE / "authoritative"


def _module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _runner_args(
    tmp_path: Path,
    *,
    profile_id: str = "lm-evaluation-harness",
    authoritative: bool = False,
) -> argparse.Namespace:
    source = AUTHORITATIVE if authoritative else EXAMPLE
    artifact = source / "artifacts" / profile_id
    return argparse.Namespace(
        cases=source / "cases.json",
        dependency_lock=EXAMPLE / "locks" / f"{profile_id}.txt",
        export=tmp_path / "export.json",
        profile=artifact / "profile.json",
        raw_output=tmp_path / "upstream-output.json",
        schedule=source / "schedule.json",
    )


def test_runner_support_authors_deterministic_and_observation_exports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    support = _module(
        "qualification_runner_support_test",
        EXAMPLE / "runner_support.py",
    )
    args = _runner_args(tmp_path)
    profile, schedule, cases = support.load_inputs(args)
    assert schedule["schedule_id"] == "offline-two-record-qualification"
    assert [case["record_id"] for case in cases] == ["record-1", "record-2"]
    monkeypatch.setattr(
        support.importlib.metadata,
        "version",
        lambda _name: profile["upstream"]["package"]["version"],
    )
    support.finish_deterministic(
        args=args,
        entrypoint="real.upstream.entrypoint",
        scores=[1.0, 0.0],
        details=[{"native": 1}, {"native": 0}],
        environment=[{"name": "lm-eval", "version": "0.4.12"}],
    )
    export = json.loads(args.export.read_bytes())
    raw = json.loads(args.raw_output.read_bytes())
    assert len(export["records"]) == 2
    assert raw["entrypoint"] == "real.upstream.entrypoint"
    assert "source_evaluation" not in raw

    authoritative_args = _runner_args(
        tmp_path / "authoritative",
        authoritative=True,
    )
    support.finish_deterministic(
        args=authoritative_args,
        entrypoint="real.upstream.entrypoint",
        scores=[1.0 if index < 52 else 0.0 for index in range(102)],
        details=[{"native": index} for index in range(102)],
        environment=[],
    )
    authoritative_raw = json.loads(authoritative_args.raw_output.read_bytes())
    assert authoritative_raw["source_evaluation"]["kind"] == "model_execution"

    observation_args = _runner_args(
        tmp_path / "observation",
        profile_id="mlflow",
    )
    monkeypatch.setattr(support.importlib.metadata, "version", lambda _name: "3.14.0")
    support.finish_observation(
        args=observation_args,
        entrypoint="mlflow.models.evaluate",
        summary_kind="aggregate_metrics",
        summary_data={"accuracy": 0.5},
        environment=[],
    )
    observation = json.loads(observation_args.export.read_bytes())
    assert observation["records"] == []
    assert observation["summary"]["kind"] == "aggregate_metrics"


def test_runner_support_rejects_malformed_inputs_and_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    support = _module(
        "qualification_runner_support_failures_test",
        EXAMPLE / "runner_support.py",
    )
    args = _runner_args(tmp_path)
    profile = json.loads(args.profile.read_bytes())
    monkeypatch.setattr(
        support.importlib.metadata,
        "version",
        lambda _name: profile["upstream"]["package"]["version"],
    )
    with pytest.raises(ValueError, match="cover every scheduled record"):
        support.finish_deterministic(
            args=args,
            entrypoint="entrypoint",
            scores=[1.0],
            details=[{}],
        )
    with pytest.raises(ValueError, match="only 0 or 1"):
        support.finish_deterministic(
            args=args,
            entrypoint="entrypoint",
            scores=[0.5, 0.0],
            details=[{}, {}],
        )
    monkeypatch.setattr(support.importlib.metadata, "version", lambda _name: "wrong")
    with pytest.raises(ValueError, match="does not match profile version"):
        support.require_profile_package(profile)

    bad_json = tmp_path / "array.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="one JSON object"):
        support.load_json(bad_json)

    copied_profile = tmp_path / "profile.json"
    copied_profile.write_bytes(args.profile.read_bytes())
    bad_args = argparse.Namespace(**vars(args))
    bad_args.profile = copied_profile
    bad_args.dependency_lock = tmp_path / "lock.txt"
    bad_args.dependency_lock.write_text("wrong", encoding="utf-8")
    with pytest.raises(ValueError, match="dependency declaration"):
        support.load_inputs(bad_args)

    cases = json.loads(args.cases.read_bytes())
    cases["records"][0]["input_sha256"] = 7
    bad_args = argparse.Namespace(**vars(args))
    bad_args.cases = tmp_path / "cases.json"
    bad_args.cases.write_bytes(canonical_json_bytes(cases))
    with pytest.raises(ValueError, match="invalid input_sha256"):
        support.load_inputs(bad_args)

    malformed_cases = (
        ("records", "wrong", "array of objects"),
        ("records", [], "same record count"),
    )
    for field, value, message in malformed_cases:
        document = json.loads(args.cases.read_bytes())
        document[field] = value
        bad_args.cases = tmp_path / f"{message.replace(' ', '-')}.json"
        bad_args.cases.write_bytes(canonical_json_bytes(document))
        with pytest.raises(ValueError, match=message):
            support.load_inputs(bad_args)

    document = json.loads(args.cases.read_bytes())
    document["records"][0]["output"] = None
    bad_args.cases = tmp_path / "invalid-field.json"
    bad_args.cases.write_bytes(canonical_json_bytes(document))
    with pytest.raises(ValueError, match="invalid field"):
        support.load_inputs(bad_args)

    document = json.loads(args.cases.read_bytes())
    document["records"][0]["record_id"] = "wrong"
    bad_args.cases = tmp_path / "schedule-mismatch.json"
    bad_args.cases.write_bytes(canonical_json_bytes(document))
    with pytest.raises(ValueError, match="independent schedule"):
        support.load_inputs(bad_args)


def test_runner_support_inventory_arguments_and_invalid_source_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    support = _module(
        "qualification_runner_support_misc_test",
        EXAMPLE / "runner_support.py",
    )

    class Distribution:
        metadata = {"Name": "Mixed_Name"}
        version = "1.2.3"

    monkeypatch.setattr(
        support.importlib.metadata,
        "distributions",
        lambda: [Distribution()],
    )
    assert support.installed_inventory() == [{"name": "mixed-name", "version": "1.2.3"}]
    private = {"upstream": {"package": {"ecosystem": "private", "name": "x"}}}
    assert support.require_profile_package(private)["name"] == "x"

    args = _runner_args(tmp_path)
    argv = ["runner"]
    for name in (
        "cases",
        "dependency-lock",
        "export",
        "profile",
        "raw-output",
        "schedule",
    ):
        argv.extend([f"--{name}", str(getattr(args, name.replace("-", "_")))])
    monkeypatch.setattr(sys, "argv", argv)
    parsed = support.arguments()
    assert parsed.profile == args.profile

    document = json.loads(args.cases.read_bytes())
    document["source_evaluation"] = "invalid"
    args.cases = tmp_path / "invalid-source-evaluation.json"
    args.cases.write_bytes(canonical_json_bytes(document))
    profile = json.loads(args.profile.read_bytes())
    monkeypatch.setattr(
        support.importlib.metadata,
        "version",
        lambda _name: profile["upstream"]["package"]["version"],
    )
    with pytest.raises(ValueError, match="source_evaluation must be an object"):
        support.finish_deterministic(
            args=args,
            entrypoint="entrypoint",
            scores=[1.0, 0.0],
            details=[{}, {}],
        )


def test_matrix_execution_orchestration_and_cli_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _module("qualification_matrix_test", EXAMPLE / "matrix.py")
    profiles = matrix.profiles()
    pypi = profiles[0]
    npm = next(profile for profile in profiles if profile["profile_id"] == "promptfoo")
    assert matrix.runner_command(pypi, tmp_path / "pypi/profile.json")[0] == "uv"
    assert (
        matrix.runner_command(npm, tmp_path / "npm/profile.json")[0] == sys.executable
    )

    calls = []
    monkeypatch.setattr(
        matrix.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    class Result:
        def as_dict(self) -> dict[str, object]:
            return {"outcome": "qualified"}

    monkeypatch.setattr(matrix, "qualify", lambda *_args, **_kwargs: Result())
    replayed = []
    monkeypatch.setattr(
        matrix,
        "replay_import",
        lambda profile_id, *, write: replayed.append((profile_id, write)),
    )
    artifacts = tmp_path / "artifacts"
    matrix._execute_profiles(
        [pypi, npm],
        selected={npm["profile_id"]},
        artifacts=artifacts,
        cases=EXAMPLE / "cases.json",
        schedule=EXAMPLE / "schedule.json",
        replayable=True,
    )
    assert len(calls) == 1
    assert replayed == [("promptfoo", True)]
    assert json.loads(
        (artifacts / "promptfoo" / "qualification-result.json").read_bytes()
    ) == {"outcome": "qualified"}

    dispatched = []
    monkeypatch.setattr(matrix, "execute", lambda selected: dispatched.append(selected))
    monkeypatch.setattr(
        matrix,
        "execute_replayable",
        lambda selected: dispatched.append(selected),
    )
    monkeypatch.setattr(
        matrix, "verify_replayable", lambda: dispatched.append("replayable")
    )
    monkeypatch.setattr(matrix, "verify", lambda: dispatched.append("verify"))
    for namespace in (
        argparse.Namespace(command="execute", profiles=["one"]),
        argparse.Namespace(command="execute-replayable", profiles=["two"]),
        argparse.Namespace(command="verify-replayable"),
        argparse.Namespace(command="verify"),
    ):
        monkeypatch.setattr(matrix, "parse_args", lambda value=namespace: value)
        matrix.main()
    assert dispatched == [
        {"one"},
        {"two"},
        "replayable",
        "verify",
    ]


def test_matrix_execution_wrappers_and_primary_validation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _module("qualification_matrix_wrappers_test", EXAMPLE / "matrix.py")
    calls = []
    monkeypatch.setattr(
        matrix,
        "_execute_profiles",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    matrix.execute({"promptfoo"})
    matrix.execute_replayable({"inspect-ai"})
    assert calls[0][1]["replayable"] is False
    assert calls[1][1]["replayable"] is True

    profiles = matrix.profiles()
    observation_profiles = [
        {
            **profile,
            "authority": {
                "metric": None,
                "mode": "observation_only",
                "reason": "aggregate_only",
            },
        }
        for profile in profiles
    ]
    monkeypatch.setattr(matrix, "profiles", lambda: observation_profiles)
    with pytest.raises(ValueError, match="replayable and observation-only"):
        matrix.verify()

    monkeypatch.setattr(matrix, "profiles", lambda: profiles)
    monkeypatch.setattr(matrix, "demonstration_levels", lambda: {})
    with pytest.raises(ValueError, match="cover exactly"):
        matrix.verify()

    monkeypatch.setattr(
        matrix,
        "load",
        lambda _path: {
            "format": "wrong",
            "source_evaluation": {},
            "records": [],
        },
    )
    with pytest.raises(ValueError, match="102-record model execution"):
        matrix.verify_replayable()


def test_matrix_rejects_invalid_control_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _module("qualification_matrix_failures_test", EXAMPLE / "matrix.py")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="contain an object"):
        matrix.load(invalid)

    monkeypatch.setattr(matrix, "load", lambda _path: {"profiles": "wrong"})
    with pytest.raises(ValueError, match="matrix profiles"):
        matrix.profiles()
    with pytest.raises(ValueError, match="demonstration levels"):
        matrix.demonstration_levels()

    monkeypatch.setattr(matrix, "matrix_document", lambda: {"categories": []})
    with pytest.raises(ValueError, match="matrix categories"):
        matrix.categories()
    monkeypatch.setattr(matrix, "matrix_document", lambda: {"selection": []})
    with pytest.raises(ValueError, match="matrix selection"):
        matrix.selection_policy()
    monkeypatch.setattr(matrix, "matrix_document", lambda: {"release_focus": []})
    with pytest.raises(ValueError, match="release focus"):
        matrix.release_focus()

    duplicate = matrix.load = lambda _path: {
        "profiles": [{"profile_id": "same", "authority": {"mode": "observation_only"}}]
        * 12
    }
    assert duplicate is not None
    monkeypatch.setattr(matrix, "profiles", lambda: matrix.load(Path())["profiles"])
    with pytest.raises(ValueError, match="unique profile identifiers"):
        matrix.verify()


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("category", "category is invalid"),
        ("support", "support status is invalid"),
        ("selection", "selection review metadata is invalid"),
        ("level", "demonstration status is invalid"),
        ("focus", "release-focus profile is missing"),
        ("profile", "profile is stale"),
        ("result", "qualification result is stale"),
        ("format", "raw output format is invalid"),
        ("upstream", "upstream identity is invalid"),
    ],
)
def test_matrix_rejects_stale_or_invalid_retained_matrix_state(
    failure: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _module(
        f"qualification_matrix_validation_{failure}_test",
        EXAMPLE / "matrix.py",
    )
    profiles = matrix.profiles()[:12]
    categories = matrix.categories()
    selection = matrix.selection_policy()
    levels = {
        profile["profile_id"]: {
            "retained_signed_transaction": False,
        }
        for profile in profiles
    }

    class Result:
        def __init__(self, profile_id: str) -> None:
            self.profile_id = profile_id
            self.outcome = "qualified_for_import"

        def as_dict(self) -> dict[str, object]:
            return {"profile_id": self.profile_id}

    results = {
        profile["profile_id"]: Result(profile["profile_id"]) for profile in profiles
    }
    artifacts = tmp_path / "artifacts"
    for profile in profiles:
        artifact = artifacts / profile["profile_id"]
        artifact.mkdir(parents=True)
        artifact.joinpath("profile.json").write_bytes(
            canonical_json_bytes(matrix.qualification_profile(profile))
        )
        artifact.joinpath("qualification-result.json").write_bytes(
            canonical_json_bytes(results[profile["profile_id"]].as_dict())
        )
        artifact.joinpath("upstream-output.json").write_bytes(
            canonical_json_bytes(
                {
                    "format": "invarlock/upstream-evaluator-execution-v1",
                    "upstream": profile["upstream"],
                }
            )
        )

    monkeypatch.setattr(matrix, "ARTIFACTS", artifacts)
    monkeypatch.setattr(matrix, "profiles", lambda: profiles)
    monkeypatch.setattr(matrix, "categories", lambda: categories)
    monkeypatch.setattr(matrix, "selection_policy", lambda: selection)
    monkeypatch.setattr(matrix, "demonstration_levels", lambda: levels)
    monkeypatch.setattr(matrix, "release_focus", lambda: [])
    monkeypatch.setattr(
        matrix,
        "qualify",
        lambda profile, **_kwargs: results[profile["profile_id"]],
    )

    first = profiles[0]
    first_artifact = artifacts / first["profile_id"]
    if failure == "category":
        profiles[0] = {**first, "category": "missing"}
    elif failure == "support":
        profiles[0] = {**first, "support_status": "unmaintained"}
    elif failure == "selection":
        monkeypatch.setattr(matrix, "selection_policy", lambda: {})
    elif failure == "level":
        levels[first["profile_id"]] = {"retained_signed_transaction": "yes"}
    elif failure == "focus":
        monkeypatch.setattr(matrix, "release_focus", lambda: ["missing"])
    elif failure == "profile":
        first_artifact.joinpath("profile.json").write_bytes(b"{}\n")
    elif failure == "result":
        first_artifact.joinpath("qualification-result.json").write_bytes(b"{}\n")
    elif failure == "format":
        document = json.loads(
            first_artifact.joinpath("upstream-output.json").read_bytes()
        )
        document["format"] = "wrong"
        first_artifact.joinpath("upstream-output.json").write_bytes(
            canonical_json_bytes(document)
        )
    else:
        document = json.loads(
            first_artifact.joinpath("upstream-output.json").read_bytes()
        )
        document["upstream"] = {}
        first_artifact.joinpath("upstream-output.json").write_bytes(
            canonical_json_bytes(document)
        )

    with pytest.raises(ValueError, match=message):
        matrix.verify()


def test_authoritative_replay_write_staleness_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = _module("qualification_replay_test", AUTHORITATIVE / "replay.py")
    copied = tmp_path / "authoritative"
    shutil.copytree(AUTHORITATIVE, copied)
    shutil.rmtree(copied / "__pycache__", ignore_errors=True)
    replay.ROOT = copied

    document = replay.replay("inspect-ai", write=True)
    assert document["record_count"] == 102
    replay.replay("inspect-ai", write=False)

    artifact = copied / "artifacts" / "inspect-ai"
    records = artifact / "runtime-import-records.jsonl"
    original_records = records.read_bytes()
    records.write_bytes(original_records + b"\n")
    with pytest.raises(ValueError, match="records are stale"):
        replay.replay("inspect-ai", write=False)
    records.write_bytes(original_records)

    retained_replay = artifact / "import-replay.json"
    original_replay = retained_replay.read_bytes()
    retained_replay.write_bytes(original_replay + b"\n")
    with pytest.raises(ValueError, match="import replay is stale"):
        replay.replay("inspect-ai", write=False)
    retained_replay.write_bytes(original_replay)

    monkeypatch.setattr(
        replay,
        "load_external_scoring_records_jsonl",
        lambda *_args, **_kwargs: (),
    )
    with pytest.raises(ValueError, match="changed qualified records"):
        replay.replay("inspect-ai", write=False)

    calls = []
    monkeypatch.setattr(
        replay,
        "replay",
        lambda profile_id, *, write: calls.append((profile_id, write)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["replay.py", "write", "inspect-ai"],
    )
    replay.main()
    assert calls == [("inspect-ai", True)]


def test_authoritative_replay_rejects_authority_and_source_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = _module("qualification_replay_failures_test", AUTHORITATIVE / "replay.py")
    non_object = tmp_path / "array.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="one JSON object"):
        replay._load(non_object)

    class Observation:
        authority = "observation_only"

    monkeypatch.setattr(
        replay,
        "qualify_evaluator_export",
        lambda **_kwargs: Observation(),
    )
    replay.ROOT = tmp_path
    with pytest.raises(ValueError, match="requires verdict authority"):
        replay.replay("profile", write=False)
