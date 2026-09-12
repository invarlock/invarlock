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
