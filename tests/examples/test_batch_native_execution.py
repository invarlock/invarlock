from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/evaluator-qualification"


def load(filename):
    spec = importlib.util.spec_from_file_location(
        filename[:-3] + "_test", EXAMPLE / "maintained" / filename
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def modules(monkeypatch, mapping):
    for name, values in mapping.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def cases():
    return [
        {
            "record_id": "one",
            "input": "same prompt",
            "output": "one",
            "reference": "one",
        },
        {
            "record_id": "two",
            "input": "same prompt",
            "output": "two",
            "reference": "one",
        },
    ]


def test_evidently_executes_declared_descriptor_and_preserves_dataframe(
    monkeypatch, cases
):
    module = load("batch_native.py")
    seen = []

    def from_pandas(source, *, data_definition, descriptors):
        assert data_definition == "definition"
        assert descriptors == [
            {"columns": ["output", "reference"], "alias": "exact_match"}
        ]
        seen.extend(source)
        return SimpleNamespace(
            as_dataframe=lambda: SimpleNamespace(
                to_dict=lambda orient: (
                    [
                        {**row, "exact_match": row["output"] == row["reference"]}
                        for row in source
                    ]
                    if orient == "records"
                    else None
                )
            )
        )

    modules(
        monkeypatch,
        {
            "pandas": {"DataFrame": lambda rows: rows},
            "evidently": {
                "DataDefinition": lambda: "definition",
                "Dataset": SimpleNamespace(from_pandas=from_pandas),
            },
            "evidently.descriptors": {"ExactMatch": lambda **kwargs: kwargs},
        },
    )
    native, environment = module.execute(
        "evidently", cases, version="0.7.21", dependency_lock=Path("unused")
    )
    assert environment is None
    assert [row["record_id"] for row in seen] == ["one", "two"]
    assert [row["exact_match"] for row in native["rows"]] == [True, False]


def test_langfuse_retains_task_outputs_and_metric_metadata(monkeypatch, cases):
    module = load("batch_native.py")

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["tracing_enabled"] is False

        def run_experiment(self, **kwargs):
            assert kwargs["max_concurrency"] == 1
            items = []
            for item in kwargs["data"]:
                output = kwargs["task"](item=item)
                evaluation = kwargs["evaluators"][0](
                    output=output, expected_output=item["expected_output"]
                )
                items.append(
                    SimpleNamespace(item=item, output=output, evaluations=[evaluation])
                )
            return SimpleNamespace(
                item_results=items,
                run_evaluations=[
                    SimpleNamespace(name="unexpected", value=0.5, data_type="NUMERIC")
                ],
            )

    modules(
        monkeypatch, {"langfuse": {"Langfuse": Client, "Evaluation": SimpleNamespace}}
    )
    result = module.langfuse(cases)
    assert [row["output"] for row in result["item_results"]] == ["one", "two"]
    assert result["item_results"][0]["evaluations"] == [
        {"name": "exact_match", "value": True, "data_type": "BOOLEAN"}
    ]
    assert result["run_evaluations"][0]["value"] == 0.5


def test_pydantic_duplicate_prompts_keep_distinct_captured_outputs(monkeypatch, cases):
    module = load("batch_native.py")

    class Dataset:
        def __init__(self, **kwargs):
            self.cases = kwargs["cases"]
            assert kwargs["evaluators"] == [{"evaluation_name": "exact_match"}]

        def evaluate_sync(self, task, *, max_concurrency):
            assert max_concurrency == 1
            return SimpleNamespace(
                cases=[
                    SimpleNamespace(
                        name=case.name,
                        inputs=case.inputs,
                        expected_output=case.expected_output,
                        output=task(case.inputs),
                        assertions={
                            "exact_match": SimpleNamespace(
                                value=task(case.inputs) == case.expected_output
                            )
                        },
                        evaluator_failures=["failure"],
                        scores={"unexpected": SimpleNamespace(value=1.0)},
                        labels={"label": SimpleNamespace(value="C")},
                    )
                    for case in self.cases
                ],
                failures=["task failure"],
                report_evaluator_failures=["report failure"],
            )

    modules(
        monkeypatch,
        {
            "pydantic_evals": {"Case": SimpleNamespace, "Dataset": Dataset},
            "pydantic_evals.evaluators": {"EqualsExpected": lambda **kwargs: kwargs},
        },
    )
    result = module.pydantic(cases)
    assert [row["output"] for row in result["cases"]] == ["one", "two"]
    assert [row["assertions"]["exact_match"]["value"] for row in result["cases"]] == [
        True,
        False,
    ]
    assert result["cases"][0]["evaluator_failures"] == ["failure"]
    assert result["failures"] == ["task failure"]
    assert result["report_evaluator_failures"] == ["report failure"]


def test_azure_uses_explicit_column_mapping_and_retains_native_rows(monkeypatch, cases):
    module = load("batch_native.py")

    def evaluate(**kwargs):
        assert kwargs["fail_on_evaluator_errors"] is True
        assert kwargs["evaluator_config"] == {
            "exact_match": {
                "column_mapping": {
                    "ground_truth": "${data.ground_truth}",
                    "response": "${data.response}",
                }
            }
        }
        source = [json.loads(line) for line in kwargs["data"].read_text().splitlines()]
        return {
            "rows": [
                {
                    **{f"inputs.{key}": value for key, value in row.items()},
                    "outputs.exact_match.exact_match": kwargs["evaluators"][
                        "exact_match"
                    ](response=row["response"], ground_truth=row["ground_truth"])[
                        "exact_match"
                    ],
                }
                for row in source
            ]
        }

    modules(monkeypatch, {"azure.ai.evaluation": {"evaluate": evaluate}})
    result = module.azure(cases)
    assert [row["inputs.record_id"] for row in result["rows"]] == ["one", "two"]
    assert [row["outputs.exact_match.exact_match"] for row in result["rows"]] == [
        1.0,
        0.0,
    ]


@pytest.mark.parametrize("failure", [None, "package", "integrity", "shasum", "exit"])
def test_promptfoo_pins_package_and_assertion_execution(
    monkeypatch, tmp_path, cases, failure
):
    module = load("batch_native.py")
    lock = tmp_path / "lock.txt"
    lock.write_text(
        "package=promptfoo@"
        + ("other" if failure == "package" else "0.121.19")
        + "\nintegrity=expected\nshasum=expected\n"
    )
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[0] == "npm":
            return SimpleNamespace(
                stdout=json.dumps(
                    {
                        "dist.integrity": "wrong"
                        if failure == "integrity"
                        else "expected",
                        "dist.shasum": "wrong" if failure == "shasum" else "expected",
                    }
                )
            )
        assert kwargs["env"]["PROMPTFOO_DISABLE_TELEMETRY"] == "1"
        config = json.loads(Path(command[command.index("--config") + 1]).read_bytes())
        assert config["evaluateOptions"]["maxConcurrency"] == 1
        assert config["tests"][0]["assert"] == [{"type": "equals", "value": "one"}]
        Path(command[command.index("--output") + 1]).write_text(
            '{"results":{"results":[]}}'
        )
        return SimpleNamespace(returncode=2 if failure == "exit" else 100)

    monkeypatch.setattr(module.subprocess, "run", run)
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            module.promptfoo(cases, version="0.121.19", dependency_lock=lock)
    else:
        result, inventory = module.execute(
            "promptfoo", cases, version="0.121.19", dependency_lock=lock
        )
        assert result == {"results": {"results": []}}
        assert inventory == [
            {
                "name": "promptfoo",
                "version": "0.121.19",
                "integrity": "expected",
                "shasum": "expected",
            }
        ]
        assert commands[0][2] == commands[1][2] == "promptfoo@0.121.19"


def test_runner_freezes_scored_inputs_and_rejects_historical_profile(
    monkeypatch, tmp_path, cases
):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    module = load("batch_runner.py")
    args = argparse.Namespace(
        **{
            name: tmp_path / f"{name}.json"
            for name in (
                "profile",
                "cases",
                "schedule",
                "dependency_lock",
                "export",
                "raw_output",
            )
        }
    )
    for name in ("profile", "cases", "schedule", "dependency_lock"):
        getattr(args, name).write_text(name)
    profile = {
        "profile_id": "evidently-strict-batch-v1",
        "upstream": {"package": {"version": "0.7.21"}},
    }
    monkeypatch.setattr(module, "arguments", lambda: args)
    monkeypatch.setattr(module, "load_inputs", lambda _: (profile, {}, cases))
    monkeypatch.setattr(module, "require_profile_package", lambda _: None)
    monkeypatch.setattr(module, "require_current_profile", lambda *_: None)
    checked = []

    def execute(provider, frozen_cases, **kwargs):
        assert provider == "evidently" and frozen_cases == cases
        args.cases.write_text("changed while native scoring")
        return {
            "rows": [
                {key: case[key] for key in ("record_id", "output", "reference")}
                | {"exact_match": case["output"] == case["reference"]}
                for case in cases
            ]
        }, None

    def finish(**kwargs):
        checked.append(kwargs)
        assert kwargs["args"].cases.read_text() == "cases"
        assert kwargs["args"].export == args.export
        assert kwargs["args"].raw_output == args.raw_output

    monkeypatch.setattr(module, "execute", execute)
    monkeypatch.setattr(module, "finish_deterministic", finish)
    module.main()
    assert checked[0]["scores"] == [1.0, 0.0]
    assert not checked[0]["args"].cases.exists()
    profile["profile_id"] = "evidently"
    with pytest.raises(ValueError, match="separate versioned"):
        module.main()


def test_complete_current_profile_binding_rejects_candidate_and_stale_sources(
    monkeypatch,
):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    binding = load("profile_binding.py")
    import matrix

    definition = json.loads((EXAMPLE / "maintained/batch-profiles.json").read_bytes())[
        "profiles"
    ][0]
    expected = matrix.qualification_profile(definition)
    assert binding.current_profile(definition, EXAMPLE) == expected
    binding.require_current_profile(expected, definition, EXAMPLE)
    for section, field in [
        ("execution", "dependency_lock_sha256"),
        ("execution", "runner_sha256"),
        ("upstream", "project_url"),
    ]:
        changed = json.loads(json.dumps(expected))
        changed[section][field] = "changed"
        with pytest.raises(ValueError, match="current definition"):
            binding.require_current_profile(changed, definition, EXAMPLE)
    expected = json.loads(json.dumps(expected))
    expected["upstream"]["package"]["version"] = "99.0.0"
    with pytest.raises(ValueError, match="current definition"):
        binding.require_current_profile(expected, definition, EXAMPLE)
