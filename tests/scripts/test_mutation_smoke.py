from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType


def _load_mutation_smoke(repo_root: Path) -> ModuleType:
    module_path = repo_root / "scripts" / "coverage" / "mutation_smoke.py"
    spec = importlib.util.spec_from_file_location(
        "mutation_smoke_under_test", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _result(returncode: int, output: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["pytest"], returncode=returncode, stdout=output, stderr=None
    )


def test_configured_mutation_oracle_nodes_resolve() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    module = _load_mutation_smoke(repo_root)
    missing: list[str] = []

    for mutant in module.MUTANTS:
        for selector in mutant.killed_by:
            relative_path, separator, node_name = selector.partition("::")
            path = repo_root / relative_path
            if not separator or not node_name or not path.is_file():
                missing.append(selector)
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            function_names = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            }
            if node_name not in function_names:
                missing.append(selector)

    assert missing == []


def test_mutation_smoke_refuses_to_count_preexisting_oracle_failure(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = _load_mutation_smoke(Path(__file__).resolve().parents[2])
    mutant = module.Mutant(
        name="must-not-run",
        path="src/example.py",
        original="before",
        mutated="after",
        killed_by=("tests/test_example.py::test_behavior",),
    )
    monkeypatch.setattr(module, "MUTANTS", (mutant,))
    monkeypatch.setattr(
        module,
        "_run_pytest",
        lambda *_args, **_kwargs: _result(1, "baseline assertion failed"),
    )

    assert module.run_mutation_smoke(tmp_path) == 1
    captured = capsys.readouterr()
    assert "baseline oracle tests failed" in captured.err
    assert "baseline assertion failed" in captured.out
    assert not (tmp_path / "src").exists()


def test_mutation_smoke_runs_mutant_after_passing_baseline(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = _load_mutation_smoke(Path(__file__).resolve().parents[2])
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "example.py").write_text("value = 1\n", encoding="utf-8")
    mutant = module.Mutant(
        name="behavior-change",
        path="src/example.py",
        original="value = 1",
        mutated="value = 2",
        killed_by=("tests/test_example.py::test_behavior",),
    )
    monkeypatch.setattr(module, "MUTANTS", (mutant,))
    calls: list[tuple[Path, tuple[str, ...]]] = []

    def fake_run_pytest(
        _repo: Path, worktree: Path, tests: tuple[str, ...]
    ) -> subprocess.CompletedProcess[str]:
        calls.append((worktree, tests))
        if worktree == tmp_path:
            assert (worktree / "src" / "example.py").read_text(
                encoding="utf-8"
            ) == "value = 1\n"
            return _result(0, "1 passed")
        assert (worktree / "src" / "example.py").read_text(
            encoding="utf-8"
        ) == "value = 2\n"
        return _result(1, "assertion failed")

    monkeypatch.setattr(module, "_run_pytest", fake_run_pytest)

    assert module.run_mutation_smoke(tmp_path) == 0
    assert len(calls) == 2
    assert all(tests == mutant.killed_by for _worktree, tests in calls)
    captured = capsys.readouterr()
    assert "baseline: 1 oracle tests passed" in captured.out
    assert "behavior-change: killed" in captured.out
