from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "examples" / "judge-measurements" / "collect.py"


def test_collection_example_is_inert_without_explicit_execution(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--execute-collection" in result.stderr
    assert not any(tmp_path.iterdir())


def test_collection_example_help_needs_no_optional_sdk():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--execute-collection" in result.stdout


def test_collection_example_checks_output_before_loading_optional_sdk(tmp_path):
    output = tmp_path / "measurements-collected.json"
    output.write_text("already present")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--execute-collection",
        ],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "OPENAI_API_KEY": "unused-test-key"},
    )
    assert result.returncode == 2
    assert "must be a new file" in result.stderr


@pytest.mark.parametrize("unsafe", ("ancestor_symlink", "parent_traversal"))
def test_collection_example_rejects_unsafe_output_ancestry_before_sdk(tmp_path, unsafe):
    output = "../outside.json"
    if unsafe == "ancestor_symlink":
        real = tmp_path / "real" / "nested"
        real.mkdir(parents=True)
        (tmp_path / "alias").symlink_to(tmp_path / "real", target_is_directory=True)
        output = "alias/nested/result.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--execute-collection",
            "--output",
            output,
        ],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "OPENAI_API_KEY": "unused-test-key"},
    )
    assert result.returncode == 2
    assert (
        "relative path" in result.stderr or "non-symlink directories" in result.stderr
    )
    assert not (tmp_path.parent / "outside.json").exists()


def test_incomplete_collection_prints_a_followable_resume_action(
    tmp_path, monkeypatch, capsys
):
    spec = importlib.util.spec_from_file_location("judge_collect_example", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    async def incomplete(_args):
        return {
            "format": "fixture",
            "completeness": {"completed_trials": 1, "expected_trials": 2},
        }

    monkeypatch.setattr(module, "_collect", incomplete)
    monkeypatch.setenv("OPENAI_API_KEY", "unused-test-key")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect",
            "--root",
            str(tmp_path),
            "--execute-collection",
        ],
    )
    module.main()
    output = capsys.readouterr().out
    assert "same checkpoint and a new output path" in output
    assert "--output measurements-resumed.json" in output
    assert (tmp_path / "measurements-collected.json").exists()


def test_documented_source_install_does_not_name_an_unpublished_distribution():
    root = SCRIPT.parents[2]
    for path in (
        root / "docs/reference/judge-measurements.md",
        root / "examples/judge-measurements/README.md",
        root / "addins/inspect_judge/README.md",
    ):
        text = path.read_text(encoding="utf-8")
        assert "python -m pip install ." in text
        assert "invarlock-inspect-judge[inspect]==0.15.0" not in text


@pytest.fixture
def collector():
    spec = importlib.util.spec_from_file_location("judge_collector_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_input_loader_rejects_nonobject_and_enforces_size(tmp_path, collector):
    path = tmp_path / "input.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        collector._input(path, maximum=10)
    path.write_text('{"value":100}')
    with pytest.raises(ValueError):
        collector._input(path, maximum=2)
    assert collector._input(path, maximum=100) == {"value": 100}


@pytest.mark.parametrize(
    "close_client,fail", [(True, False), (False, False), (True, True)]
)
def test_collector_wires_frozen_inputs_and_closes_client(
    tmp_path, collector, monkeypatch, close_client, fail
):
    import argparse
    import asyncio
    import json
    from types import ModuleType, SimpleNamespace

    for name in ("plan", "collection", "baseline", "subject"):
        (tmp_path / f"{name}.json").write_text(json.dumps({"identity": name}))
    observed = {}

    async def close():
        observed["closed"] = True

    client = SimpleNamespace(close=close) if close_client else object()
    model = SimpleNamespace(api=SimpleNamespace(client=client))

    def get_model(grader, **options):
        observed["model_options"] = (grader, options)
        return model

    model_module = ModuleType("inspect_ai.model")
    model_module.get_model = get_model
    monkeypatch.setitem(sys.modules, "inspect_ai.model", model_module)
    module = ModuleType("invarlock_addins.inspect_judge")
    module.CollectionOptions = SimpleNamespace(
        from_mapping=lambda value: SimpleNamespace(grader="openai/pinned", config=value)
    )
    module.RunnerOptions = lambda **kwargs: kwargs

    async def collect(**kwargs):
        observed["call"] = kwargs
        if fail:
            raise ValueError("retained failure")
        return {"completed": 1}

    module.collect = collect
    monkeypatch.setitem(sys.modules, "invarlock_addins.inspect_judge", module)
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-only")
    args = argparse.Namespace(
        root=tmp_path,
        plan="plan.json",
        collection="collection.json",
        baseline_run="baseline.json",
        subject_run="subject.json",
        checkpoint="checkpoint",
        scorer_id="fixed-scorer",
        invocation_timeout_seconds=7,
    )
    if fail:
        with pytest.raises(ValueError, match="retained failure"):
            asyncio.run(collector._collect(args))
    else:
        assert asyncio.run(collector._collect(args)) == {"completed": 1}
    assert observed["model_options"] == (
        "openai/pinned",
        {
            "api_key": "fixture-only",
            "responses_api": False,
            "max_retries": 0,
            "memoize": False,
        },
    )
    assert observed["call"]["plan"] == {"identity": "plan"}
    assert observed["call"]["baseline_run"] == {"identity": "baseline"}
    assert observed["call"]["subject_run"] == {"identity": "subject"}
    assert observed["call"]["runner"] == {
        "checkpoint_directory": tmp_path / "checkpoint",
        "scorer_id": "fixed-scorer",
        "invocation_timeout_seconds": 7,
    }
    assert observed.get("closed", False) == close_client


@pytest.mark.parametrize(
    "kind,message",
    [
        ("missing_root", "existing real directory"),
        ("file_root", "existing real directory"),
        ("checkpoint_escape", "relative path"),
        ("output_in_checkpoint", "outside the checkpoint"),
        ("missing_key", "OPENAI_API_KEY"),
        ("custom_url", "custom OpenAI"),
        ("collection_error", "Judge collection failed"),
    ],
)
def test_collector_cli_rejects_invalid_environment_and_paths(
    tmp_path, collector, monkeypatch, capsys, kind, message
):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-only")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    root = tmp_path
    extra = []
    if kind == "missing_root":
        root = tmp_path / "absent"
    elif kind == "file_root":
        root = tmp_path / "file"
        root.write_text("fixture")
    elif kind == "checkpoint_escape":
        extra = ["--checkpoint", "../escape"]
    elif kind == "output_in_checkpoint":
        extra = ["--output", "judge-checkpoint/output.json"]
    elif kind == "missing_key":
        monkeypatch.delenv("OPENAI_API_KEY")
    elif kind == "custom_url":
        monkeypatch.setenv("OPENAI_BASE_URL", "https://invalid.example")

    async def forbidden(_args):
        if kind == "collection_error":
            raise ValueError("fixture collection failure")
        pytest.fail("invalid invocation reached collector")

    monkeypatch.setattr(collector, "_collect", forbidden)
    monkeypatch.setattr(
        sys, "argv", ["collect", "--root", str(root), "--execute-collection", *extra]
    )
    with pytest.raises(SystemExit) as exc:
        collector.main()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "measurements-collected.json").exists()


def test_complete_collection_publishes_without_resume_message(
    tmp_path, collector, monkeypatch, capsys
):
    async def complete(_args):
        return {"completeness": {"completed_trials": 2, "expected_trials": 2}}

    monkeypatch.setattr(collector, "_collect", complete)
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-only")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setattr(
        sys, "argv", ["collect", "--root", str(tmp_path), "--execute-collection"]
    )
    collector.main()
    output = capsys.readouterr().out
    assert "Retained 2/2" in output
    assert "incomplete" not in output
