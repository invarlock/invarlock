from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from coverage import CoverageData

from scripts.ci import coverage_runner as runner
from tests._support_repository_contracts import MakefileContract


def test_all_live_example_helpers_belong_only_to_example_partition():
    paths = {
        path.relative_to(runner.ROOT).as_posix()
        for path in (runner.ROOT / "tests/evaluation_records").glob("test_live_*.py")
    }
    examples = runner.selection("examples")
    core = runner.selection("core")
    assert paths
    assert paths <= set(examples)
    assert all(
        examples.count(path) == 1 and f"--ignore={path}" in core for path in paths
    )
    result = subprocess.run(
        ["make", "--dry-run", "examples-check", "PYTHON=true", "PYTEST_WORKERS=0"],
        cwd=runner.ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert all(result.stdout.count(path) == 1 for path in paths)


def write_data(path: Path, name: str = "src/invarlock/probe.py") -> None:
    data = CoverageData(basename=str(path))
    data.add_arcs({name: [(-1, 1), (1, 2), (2, -1)]})
    data.write()


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "source_identity", lambda: "source")
    root = tmp_path / "artifacts"
    for shard in runner.SHARDS:
        folder = root / shard
        folder.mkdir(parents=True)
        inventory = [f"tests/{shard}/test_probe.py::test_probe"]
        runner._write_json(folder / "inventory.json", inventory)
        write_data(folder / ".coverage", f"src/invarlock/{shard}.py")
        runner._write_json(
            folder / "manifest.json",
            {
                "version": 1,
                "shard": shard,
                "status": "passed",
                "source_identity": "source",
                "exit_code": 0,
                "inventory": inventory,
                "data_sha256": runner._data_digest(folder / ".coverage"),
            },
        )
    return root


def test_combine_preserves_each_shards_branches(artifacts, tmp_path):
    output = tmp_path / "combined"
    write_data(output, "stale.py")
    runner.combine(artifacts, output)
    data = CoverageData(basename=str(output))
    data.read()
    assert data.measured_files() == {
        str(runner.ROOT / f"src/invarlock/{shard}.py") for shard in runner.SHARDS
    }
    assert data.arcs(str(runner.ROOT / "src/invarlock/core.py")) == [
        (-1, 1),
        (1, 2),
        (2, -1),
    ]
    assert all((artifacts / shard / ".coverage").is_file() for shard in runner.SHARDS)


@pytest.mark.parametrize("extra", [False, True])
def test_combine_requires_exactly_four_shards(artifacts, tmp_path, extra):
    if extra:
        folder = artifacts / "unexpected"
        folder.mkdir()
        runner._write_json(folder / "manifest.json", {})
    else:
        (artifacts / "support/manifest.json").unlink()
    with pytest.raises(ValueError, match="exactly"):
        runner.combine(artifacts, tmp_path / "out")


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("shard", "examples"),
        ("status", "running"),
        ("status", "failed"),
        ("exit_code", 1),
        ("source_identity", "other"),
    ],
)
def test_combine_rejects_failed_or_mismatched_run(artifacts, tmp_path, field, value):
    path = artifacts / "core/manifest.json"
    record = json.loads(path.read_text())
    record[field] = value
    runner._write_json(path, record)
    with pytest.raises(ValueError, match="failed or mismatched"):
        runner.combine(artifacts, tmp_path / "out")


@pytest.mark.parametrize("value", [[], ["plain"], [None], ["a::b", "a::b"], {}])
def test_combine_rejects_invalid_inventory(artifacts, tmp_path, value):
    runner._write_json(artifacts / "core/inventory.json", value)
    with pytest.raises(ValueError, match="invalid or empty"):
        runner.combine(artifacts, tmp_path / "out")


@pytest.mark.parametrize("overlap", [False, True])
def test_combine_rejects_changed_or_overlapping_inventory(artifacts, tmp_path, overlap):
    inventory = ["tests/runtime/test_probe.py::test_probe"]
    runner._write_json(artifacts / "core/inventory.json", inventory)
    if overlap:
        path = artifacts / "core/manifest.json"
        record = json.loads(path.read_text())
        record["inventory"] = inventory
        runner._write_json(path, record)
    with pytest.raises(ValueError, match="changed or overlapping"):
        runner.combine(artifacts, tmp_path / "out")


@pytest.mark.parametrize("mode", ["changed", "missing", "lines"])
def test_combine_rejects_invalid_coverage(artifacts, tmp_path, mode):
    path = artifacts / "core/.coverage"
    path.unlink()
    if mode == "changed":
        write_data(path, "changed.py")
    elif mode == "lines":
        data = CoverageData(basename=str(path))
        data.add_lines({"file.py": [1]})
        data.write()
    with pytest.raises(ValueError, match="coverage data"):
        runner.combine(artifacts, tmp_path / "out")


def test_combine_rejects_nonobject_manifest(artifacts, tmp_path):
    runner._write_json(artifacts / "core/manifest.json", [])
    with pytest.raises(ValueError, match="failed or mismatched"):
        runner.combine(artifacts, tmp_path / "out")


@pytest.mark.parametrize(
    "source",
    ["../outside.py", "/outside.py", "src/../../outside.py", "src\\outside.py"],
)
def test_combine_rejects_source_paths_outside_checkout(artifacts, tmp_path, source):
    path = artifacts / "core/.coverage"
    path.unlink()
    write_data(path, source)
    manifest = artifacts / "core/manifest.json"
    record = json.loads(manifest.read_text())
    record["data_sha256"] = runner._data_digest(path)
    runner._write_json(manifest, record)
    with pytest.raises(ValueError, match="invalid relative"):
        runner.combine(artifacts, tmp_path / "out")


def test_source_path_rejects_empty_and_symlink_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    (tmp_path / "linked.py").symlink_to(tmp_path.parent / "outside.py")
    with pytest.raises(ValueError, match="invalid relative"):
        runner._source_path("")
    with pytest.raises(ValueError, match="escapes"):
        runner._source_path("linked.py")


def test_source_identity_includes_content_deletions_links_and_commit(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    source = tmp_path / "file.py"
    source.write_text("before")
    link = tmp_path / "link.py"
    link.symlink_to("file.py")
    commit = [b"commit-one"]

    def git(command, **kwargs):
        assert kwargs["cwd"] == tmp_path
        return commit[0] if command[1] == "rev-parse" else b"file.py\0link.py\0"

    monkeypatch.setattr(runner.subprocess, "check_output", git)
    identities = [runner.source_identity()]
    source.write_text("after")
    identities.append(runner.source_identity())
    source.unlink()
    identities.append(runner.source_identity())
    link.unlink()
    link.symlink_to("elsewhere.py")
    identities.append(runner.source_identity())
    commit[0] = b"commit-two"
    identities.append(runner.source_identity())
    assert len(set(identities)) == len(identities)


def test_inventory_plugin_handles_empty_master_and_atomic_worker_collection(
    tmp_path, monkeypatch
):
    path = tmp_path / "inventory.json"
    monkeypatch.delenv("INVARLOCK_COVERAGE_INVENTORY", raising=False)
    runner.pytest_collection_finish(SimpleNamespace(items=[]))
    monkeypatch.setenv("INVARLOCK_COVERAGE_INVENTORY", str(path))
    runner.pytest_collection_finish(SimpleNamespace(items=[]))
    assert not path.exists()
    runner.pytest_collection_finish(
        SimpleNamespace(
            items=[SimpleNamespace(nodeid="b::test"), SimpleNamespace(nodeid="a::test")]
        )
    )
    assert json.loads(path.read_text()) == ["a::test", "b::test"]
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "shard,workers,exit_code,changed",
    [
        ("core", 0, 0, False),
        ("core", 2, 0, False),
        ("core", 2, 1, False),
        ("core", 0, 0, True),
        ("support", 0, 0, False),
        ("runtime", 0, 0, False),
    ],
)
def test_run_records_success_only_after_validating_outputs(
    tmp_path, monkeypatch, shard, workers, exit_code, changed
):
    identities = iter(["before", "after" if changed else "before"])
    monkeypatch.setattr(runner, "source_identity", lambda: next(identities))

    def execute(command, *, cwd, env, check):
        assert cwd == runner.ROOT and check is False
        assert ("-n" in command) is bool(workers)
        assert "--cov-report=" in command and "--durations=20" in command
        assert "--cov-branch" in command
        assert "-m" in command
        assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        assert "scripts/ci" in env["PYTHONPATH"]
        assert "addins" not in env["PYTHONPATH"]
        path = Path(env["COVERAGE_FILE"])
        manifest = json.loads((path.parent / "manifest.json").read_text())
        assert manifest["status"] == "running"
        write_data(path)
        runner._write_json(
            Path(env["INVARLOCK_COVERAGE_INVENTORY"]), ["tests/test_a.py::test_a"]
        )
        return SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr(runner.subprocess, "run", execute)
    if changed:
        with pytest.raises(ValueError, match="source files changed"):
            runner.run(shard, tmp_path, workers)
    else:
        assert runner.run(shard, tmp_path, workers) == exit_code
    record = json.loads((tmp_path / shard / "manifest.json").read_text())
    assert record["status"] == ("passed" if not exit_code and not changed else "failed")
    assert record["duration_seconds"] >= 0


def test_invalid_shard_and_worker_count_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        runner.selection("unknown")
    with pytest.raises(ValueError, match="nonnegative"):
        runner.run("core", tmp_path, -1)


def test_cli_reports_errors_and_dispatches(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "run", lambda *args: 7)
    assert runner.main(["run", "core"]) == 7
    monkeypatch.setattr(runner, "combine", lambda *args: None)
    assert runner.main(["combine", str(tmp_path)]) == 0

    def failure(*args):
        raise ValueError("missing shard")

    monkeypatch.setattr(runner, "combine", failure)
    assert runner.main(["combine", str(tmp_path)]) == 2
    assert "missing shard" in capsys.readouterr().err


def test_support_inventory_matches_existing_specialized_targets():
    make = MakefileContract.read(runner.ROOT / "Makefile")
    selected = set()
    for target in (
        "coverage-qualification",
        "coverage-release",
        "coverage-maintenance",
    ):
        selected.update(
            re.findall(r"(?<![\w/])tests/[\w/]+\.py", make.target(target).text)
        )
    assert set(runner.SUPPORT_TESTS) == selected
    assert len(runner.SUPPORT_TESTS) == len(selected)
    canary = "tests/runtime/test_tensorrt_llm_canary_preflight.py"
    assert canary in make.target("coverage-qualification").text
    assert any(canary.startswith(path + "/") for path in runner.selection("runtime"))
    assert f"--ignore={canary}" in runner.selection("runtime")


def isolated_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("COVERAGE_", "COV_CORE_", "INVARLOCK_COVERAGE_"))
        and key != "PYTEST_DISABLE_PLUGIN_AUTOLOAD"
    }


def test_real_collection_is_disjoint_and_preserves_marker_exceptions(
    tmp_path, monkeypatch
):
    support = "tests/support/test_support.py"
    monkeypatch.setattr(runner, "SUPPORT_TESTS", (support,))
    monkeypatch.setattr(runner, "RUNTIME_TESTS", ("tests/runtime",))
    files = {
        "tests/test_core.py": "def test_core(): pass\n",
        "tests/test_slow.py": "import pytest\n@pytest.mark.slow\ndef test_slow(): pass\n",
        "tests/compatibility/test_retained.py": "def test_retained(): pass\n",
        "tests/judge_measurements/test_collector.py": "def test_collector(): pass\n",
        "tests/examples/test_duplicate.py": "def test_example(): pass\n",
        "tests/integration/test_evaluator_parity.py": "def test_parity(): pass\n",
        "tests/evaluation_records/test_sdk_capture.py": "def test_sdk(): pass\n",
        support: "import pytest\n@pytest.mark.integration\ndef test_container(): pass\n",
        "tests/runtime/test_duplicate.py": "def test_runtime(): pass\n",
    }
    for name in runner.EXAMPLE_TESTS:
        if name.endswith(".py"):
            files.setdefault(name, "def test_example_helper(): pass\n")
    for name, contents in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers =\n slow\n integration\n")
    inventories = []
    for shard in runner.SHARDS:
        path = tmp_path / f"{shard}.json"
        env = isolated_env()
        env["PYTHONPATH"] = str(runner.ROOT / "scripts/ci")
        env["INVARLOCK_COVERAGE_INVENTORY"] = str(path)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-p",
                "coverage_runner",
                *runner.selection(shard),
            ],
            cwd=tmp_path,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        inventories.append(set(json.loads(path.read_text())))
    union = set().union(*inventories)
    assert sum(map(len, inventories)) == len(union) == len(files) - 1
    examples = inventories[runner.SHARDS.index("examples")]
    assert "tests/integration/test_evaluator_parity.py::test_parity" in examples
    assert "tests/evaluation_records/test_sdk_capture.py::test_sdk" in examples
    assert all(
        any(node.startswith(name + "::") for node in examples)
        for name in runner.EXAMPLE_TESTS
        if name.endswith(".py")
    )
    assert (
        "tests/judge_measurements/test_collector.py::test_collector"
        in inventories[runner.SHARDS.index("core")]
    )
    assert f"{support}::test_container" in union
    assert "tests/compatibility/test_retained.py::test_retained" in union
    assert not any("test_slow" in node for node in union)


def test_shared_config_traces_child_processes_with_relative_paths(tmp_path):
    source = tmp_path / "src/invarlock"
    source.mkdir(parents=True)
    child = source / "child.py"
    child.write_text("value = 1\nif value:\n    print('child')\n")
    parent = source / "parent.py"
    parent.write_text(
        "import subprocess, sys\nsubprocess.run([sys.executable, 'src/invarlock/child.py'], check=True)\n"
    )
    for directory in ("scripts", "examples", "src"):
        (tmp_path / directory).mkdir(exist_ok=True)
    env = isolated_env()
    data_path = tmp_path / "data"
    env["COVERAGE_FILE"] = str(data_path)
    for args in (["run", str(parent)], ["combine"]):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                args[0],
                f"--rcfile={runner.CONFIG}",
                *args[1:],
            ],
            cwd=tmp_path,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    data = CoverageData(basename=str(data_path))
    data.read()
    assert "src/invarlock/child.py" in data.measured_files()
    assert (2, 3) in data.arcs("src/invarlock/child.py")


def test_run_collects_real_xdist_inventory_and_coverage(tmp_path, monkeypatch):
    pytest.importorskip("pytest_cov")
    pytest.importorskip("xdist")
    monkeypatch.setattr(runner, "EXAMPLE_TESTS", ("tests/examples",))
    project = tmp_path / "project"
    tests = project / "tests/examples"
    tests.mkdir(parents=True)
    source = project / "src/invarlock"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("")
    (source / "probe.py").write_text(
        "def choose(value):\n    if value:\n        return 1\n    return 0\n"
    )
    (tests / "test_probe.py").write_text(
        "import pytest\nfrom invarlock.probe import choose\n"
        "@pytest.mark.parametrize('value', [False, True])\n"
        "def test_choose(value):\n    assert choose(value) == int(value)\n"
    )
    plugin = project / "scripts/ci/coverage_runner.py"
    plugin.parent.mkdir(parents=True)
    plugin.symlink_to(Path(runner.__file__))
    for directory in ("examples", "src"):
        (project / directory).mkdir(exist_ok=True)
    monkeypatch.setattr(runner, "ROOT", project)
    monkeypatch.setattr(runner, "source_identity", lambda: "source")
    artifacts = tmp_path / "artifacts"
    assert runner.run("examples", artifacts, 2) == 0
    record = json.loads((artifacts / "examples/manifest.json").read_text())
    assert record["status"] == "passed"
    assert len(record["inventory"]) == 2
    assert record["data_sha256"] == runner._data_digest(
        artifacts / "examples/.coverage"
    )
    assert (artifacts / "examples/junit.xml").is_file()
