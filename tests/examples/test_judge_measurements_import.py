from __future__ import annotations

import builtins
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
EXAMPLE = ROOT / "examples/judge-measurements"
SCRIPT = EXAMPLE / "import_inspect.py"


@pytest.fixture
def example(tmp_path, monkeypatch):
    for name in (
        "plan.json",
        "collection-inspect.json",
        "baseline_run.json",
        "subject_run.json",
        "inspect-export.json",
        "request-inspect.yaml",
        "analysis_policy.json",
    ):
        shutil.copyfile(EXAMPLE / name, tmp_path / name)
    monkeypatch.syspath_prepend(str(ROOT / "addins/inspect_judge/src"))
    spec = importlib.util.spec_from_file_location("judge_import_example", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, tmp_path


def invoke(example, monkeypatch, *extra):
    module, root = example
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--root",
            str(root),
            "--export",
            "inspect-export.json",
            "--collection",
            "collection-inspect.json",
            "--output",
            "measurements-inspect.json",
            *extra,
        ],
    )
    module.main()


def test_offline_export_import_and_publication(example, monkeypatch, capsys):
    original = builtins.__import__

    def offline(name, *args, **kwargs):
        if name.split(".")[0] in {"inspect_ai", "openai", "anthropic", "httpx"}:
            pytest.fail(f"offline example imported provider module {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", offline)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    invoke(example, monkeypatch)
    root = example[1]
    imported = json.loads((root / "measurements-inspect.json").read_text())
    exported = json.loads((root / "inspect-export.json").read_text())
    assert imported["completeness"]["completed_trials"] == 2
    retained = json.loads(imported["sources"][0]["content"])
    for actual, expected in zip(retained["records"], exported["samples"], strict=True):
        assert actual["events"] == expected["events"]
    assert "Imported 2/2 completed trials" in capsys.readouterr().out

    from typer.testing import CliRunner

    from invarlock.cli.app import app

    result = CliRunner().invoke(
        app, ["evaluate", str(root / "request-inspect.yaml"), "--unsigned", "--json"]
    )
    assert result.exit_code == 0, result.output
    report = CliRunner().invoke(
        app,
        [
            "report",
            str(root / "evidence-inspect"),
            "--html",
            str(root / "report.html"),
            "--json",
        ],
    )
    assert report.exit_code == 0, report.output
    assert (root / "report.html").is_file()


@pytest.mark.parametrize(
    "kind",
    [
        "existing",
        "leaf_symlink",
        "parent_symlink",
        "missing_parent",
        "absolute",
        "traversal",
        "root_file",
        "root_missing",
    ],
)
def test_unsafe_output_never_overwrites_or_creates(example, monkeypatch, capsys, kind):
    root = example[1]
    extra = []
    if kind == "existing":
        (root / "measurements-inspect.json").write_text("keep")
    elif kind == "leaf_symlink":
        (root / "measurements-inspect.json").symlink_to(root / "missing")
    elif kind == "parent_symlink":
        (root / "real").mkdir()
        (root / "alias").symlink_to(root / "real", target_is_directory=True)
        extra = ["--output", "alias/result.json"]
    elif kind == "missing_parent":
        extra = ["--output", "missing/result.json"]
    elif kind == "absolute":
        extra = ["--output", str(root / "outside.json")]
    elif kind == "traversal":
        extra = ["--output", "../outside.json"]
    elif kind == "root_file":
        extra = ["--root", str(root / "plan.json")]
    elif kind == "root_missing":
        extra = ["--root", str(root / "missing")]
    with pytest.raises(SystemExit) as error:
        invoke(example, monkeypatch, *extra)
    assert error.value.code == 2
    assert "Judge import failed:" in capsys.readouterr().err
    if kind == "existing":
        assert (root / "measurements-inspect.json").read_text() == "keep"
    assert not (root / "outside.json").exists()
    assert not (root / "missing").exists()


@pytest.mark.parametrize(
    "kind",
    [
        "altered_answer",
        "mismatched_collection",
        "not_object",
        "duplicate_json",
        "symlink_input",
        "oversized_export",
        "malformed_export",
        "arbitrary_archive",
    ],
)
def test_invalid_inputs_fail_without_publication(example, monkeypatch, capsys, kind):
    root = example[1]
    target = root / "baseline_run.json"
    if kind == "altered_answer":
        run = json.loads(target.read_text())
        run["records"][0]["output"] = "changed"
        target.write_text(json.dumps(run))
    elif kind == "mismatched_collection":
        config = json.loads((root / "collection-inspect.json").read_text())
        config["requests_per_minute"] = 1
        (root / "collection-inspect.json").write_text(json.dumps(config))
    elif kind == "not_object":
        target.write_text("[]")
    elif kind == "duplicate_json":
        target.write_text('{"records":[],"records":[]}')
    elif kind == "symlink_input":
        target.unlink()
        target.symlink_to(root / "subject_run.json")
    elif kind == "oversized_export":
        from invarlock.judge_measurements.contracts import MEASUREMENTS_MAX_BYTES

        with (root / "inspect-export.json").open("wb") as handle:
            handle.truncate(MEASUREMENTS_MAX_BYTES + 1)
    elif kind == "malformed_export":
        (root / "inspect-export.json").write_text("{")
    elif kind == "arbitrary_archive":
        (root / "inspect-export.json").write_text('{"samples": []}')
    with pytest.raises(SystemExit) as error:
        invoke(example, monkeypatch)
    assert error.value.code == 2
    assert "Judge import failed:" in capsys.readouterr().err
    assert not (root / "measurements-inspect.json").exists()


def test_incomplete_export_preserves_missing_trials(example, monkeypatch, capsys):
    root = example[1]
    path = root / "inspect-export.json"
    exported = json.loads(path.read_text())
    exported["samples"][0]["events"] = []
    path.write_text(json.dumps(exported))
    invoke(example, monkeypatch)
    measurements = json.loads((root / "measurements-inspect.json").read_text())
    assert measurements["completeness"]["completed_trials"] == 1
    assert measurements["completeness"]["expected_trials"] == 2
    assert "incomplete" in capsys.readouterr().out


def test_missing_addin_has_actionable_error(example, monkeypatch, capsys):
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "invarlock_addins.inspect_judge":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(SystemExit) as error:
        invoke(example, monkeypatch)
    assert error.value.code == 2
    assert (
        "requires matching core and inspect_judge packages" in capsys.readouterr().err
    )


def test_explicit_export_and_output_are_required(example, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    with pytest.raises(SystemExit) as error:
        example[0].main()
    assert error.value.code == 2
    assert "--export" in capsys.readouterr().err
