from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_script_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "extract_and_run_invarlock_cmds.py"
    spec = importlib.util.spec_from_file_location(
        "tests_extract_and_run_invarlock_cmds", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_split_env_and_argv_parses_inline_assignments() -> None:
    module = _load_script_module()
    inline_env, argv = module._split_env_and_argv(
        'FOO=1 BAR="x y" python -m invarlock --help'
    )
    assert inline_env == {"FOO": "1", "BAR": "x y"}
    assert argv == ["python", "-m", "invarlock", "--help"]


def test_run_commands_uses_argv_and_merges_inline_env(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module()

    called: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        called["cmd"] = cmd
        called["kwargs"] = kwargs

        class Proc:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return Proc()

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    results_path = tmp_path / "results.jsonl"
    commands = [
        module.Command(id=1, file="README.md", line=1, cmd="FOO=bar invarlock --help")
    ]
    module.run_commands(commands, results_path)

    assert isinstance(called["cmd"], list)
    assert called["cmd"][:3] == [sys.executable, "-m", "invarlock"]
    kwargs = called["kwargs"]
    assert "shell" not in kwargs
    assert kwargs["env"]["FOO"] == "bar"

    records = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["exit_code"] == 0


def test_run_commands_records_parse_errors(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()

    def _never_called(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("subprocess.run should not be called for parse errors")

    monkeypatch.setattr(module.subprocess, "run", _never_called)

    results_path = tmp_path / "results.jsonl"
    commands = [
        module.Command(
            id=1,
            file="README.md",
            line=1,
            cmd='FOO="unterminated invarlock --help',
        )
    ]
    module.run_commands(commands, results_path)

    records = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["exit_code"] is None
    assert "invalid command syntax" in records[0]["error"]
