"""Retained judge workflows use the native public command with no interpreter."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from invarlock.cli.app import app
from tests.cli.test_import_journey import _key
from tests.judge_measurements.test_workflow import staged  # noqa: F401

BIN = Path(os.environ["CARGO_TARGET_DIR"]) / "debug/invarlock-native-verify"


@pytest.mark.parametrize(
    "mode",
    [
        "preflight",
        "signed",
        "unsigned",
        "missing",
        "bad_binding",
        "escape",
        "native_answers",
        "overlap",
        "fail_policy",
    ],
)
def test_judge_import_parity(staged, mode):  # noqa: F811
    path, value = staged
    root = path.parent.resolve()
    key, _ = _key(root / "signer.pem")
    if mode == "missing":
        (root / "measurements.json").unlink()
    if mode == "bad_binding":
        plan = json.loads((root / "plan.json").read_text())
        plan["answer_bindings"][0]["baseline_answer_sha256"] = "0" * 64
        (root / "plan.json").write_text(json.dumps(plan))
    if mode == "escape":
        value["comparison"]["plan"] = "../plan.json"
    if mode == "native_answers":
        run = json.loads((root / "baseline_run.json").read_text())
        run["source"]["name"] = "invarlock-native-judge"
        (root / "baseline_run.json").write_text(json.dumps(run))
    if mode == "overlap":
        value["comparison"]["plan"] = "python-out/plan.json"
    preflight = mode in {"preflight", "missing"}
    extra = (
        ["--preflight"]
        if preflight
        else ["--unsigned"]
        if mode == "unsigned"
        else ["--signing-key", str(key)]
    )
    if mode == "fail_policy":
        extra += ["--fail-on-policy"]
    value["output"]["evidence"] = "python-out"
    path.write_text(json.dumps(value))
    reference = CliRunner().invoke(app, ["evaluate", str(path), "--json", *extra])
    value["output"]["evidence"] = "rust-out"
    if mode == "overlap":
        value["comparison"]["plan"] = "rust-out/plan.json"
    path.write_text(json.dumps(value))
    actual = subprocess.run(
        [str(BIN), "evaluate", str(path), "--json", *extra],
        capture_output=True,
        text=True,
        env={"PATH": "/nonexistent"},
    )
    assert actual.returncode == reference.exit_code, (
        reference.output,
        actual.stdout,
        actual.stderr,
    )
    if mode in {"bad_binding", "escape", "native_answers", "overlap"}:
        assert actual.returncode != 0
        assert not (root / "rust-out").exists()
        return
    expected = json.loads(reference.stdout)
    expected["evidence"] = str(root / "rust-out")
    assert json.loads(actual.stdout) == expected
    if not preflight:

        def files(p):
            return {
                x.relative_to(p).as_posix(): x.read_bytes()
                for x in p.rglob("*")
                if x.is_file()
            }

        assert files(root / "python-out") == files(root / "rust-out")
