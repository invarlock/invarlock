from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import coverage
import pytest

from scripts.ci import coverage_runner
from tests._support_repository_contracts import MakefileContract

ROOT = Path(__file__).resolve().parents[2]
MAKE = MakefileContract.read(ROOT / "Makefile")
SOURCES = {
    "core": "src/invarlock/engine.py",
    "addins": "addins/diagnostics/src/invarlock_addins/diagnostics/observations.py",
    "qualification": "scripts/qualification_source.py",
    "release": "scripts/release/release_preflight.py",
    "examples": "examples/quickstart/run.py",
    "maintenance": "scripts/checks/check_repo_cruft.py",
}


@pytest.fixture(params=["absolute", "relative"])
def corpus(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    _copy_configs(tmp_path)
    (tmp_path / "coverage-path-mode").write_text(request.param, encoding="utf-8")
    return tmp_path


def _copy_configs(tmp_path: Path) -> None:
    for config in ["pyproject.toml", *ROOT.glob("scripts/*.coveragerc")]:
        relative = Path(config) if isinstance(config, str) else config.relative_to(ROOT)
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    destination = tmp_path / "scripts/ci/coverage.coveragerc"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "scripts/ci/coverage.coveragerc", destination)


def _data(root: Path, files: dict[str, tuple[str, list[tuple[int, int]]]]) -> None:
    data = coverage.CoverageData(basename=str(root / ".coverage"))
    relative_files = (root / "coverage-path-mode").read_text(
        encoding="utf-8"
    ) == "relative"
    for relative, (source, arcs) in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        filename = relative if relative_files else str(path)
        data.add_arcs({filename: arcs})
        data.touch_file(filename)
    data.write()
    if relative_files:
        _combine_at(root, root / ".coverage")


def _combine_at(root: Path, data_file: Path) -> None:
    artifacts = root / "artifacts"
    for shard in coverage_runner.SHARDS:
        directory = artifacts / shard
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(data_file, directory / ".coverage")
        inventory = [f"tests/{shard}/test_sample.py::test_sample"]
        coverage_runner._write_json(directory / "inventory.json", inventory)
        coverage_runner._write_json(
            directory / "manifest.json",
            {
                "version": 1,
                "shard": shard,
                "status": "passed",
                "exit_code": 0,
                "source_identity": "same-source",
                "inventory": inventory,
                "data_sha256": coverage_runner._data_digest(directory / ".coverage"),
            },
        )
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        patch.setattr(coverage_runner, "ROOT", root)
        patch.setattr(
            coverage_runner, "CONFIG", root / "scripts/ci/coverage.coveragerc"
        )
        patch.setattr(coverage_runner, "source_identity", lambda: "same-source")
        coverage_runner.combine(artifacts, root / ".coverage")


def _arguments(
    domain: str, kind: str = "report", *, per_file: bool = False
) -> list[str]:
    target = (
        "coverage-check-files"
        if domain == "core" and per_file
        else f"coverage-{domain}-report"
    )
    block = MAKE.target(target).text.replace("\\\n", " ")
    commands = re.findall(rf"-m coverage {kind} ([^\n]+)", block)
    if per_file:
        # Qualification and release enumerate files; the other domains use loops.
        matches = [command for command in commands if "$$source" in command]
        if not matches:
            matches = [
                command
                for command in commands
                if f"--include='{SOURCES[domain]}'" in command
            ]
        command = matches[0].replace("$$source", SOURCES[domain])
    else:
        command = commands[0]
    command = command.split(" || exit", 1)[0].rstrip("; ")
    return [sys.executable, "-m", "coverage", kind, *shlex.split(command)]


def _run(root: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["COVERAGE_FILE"] = str(root / ".coverage")
    env.pop("COVERAGE_PROCESS_START", None)
    return subprocess.run(
        arguments, cwd=root, env=env, text=True, capture_output=True, check=False
    )


@pytest.mark.parametrize("domain", SOURCES)
def test_domain_xml_excludes_other_combined_sources(corpus: Path, domain: str) -> None:
    _data(
        corpus,
        {
            source: ("value = 1\n", [(-1, 1), (1, -1)] if name == domain else [])
            for name, source in SOURCES.items()
        },
    )
    arguments = _arguments(domain, "xml")
    result = _run(corpus, arguments)
    assert result.returncode == 0, result.stdout + result.stderr
    report = ET.parse(corpus / arguments[arguments.index("-o") + 1]).getroot()
    assert report.attrib["lines-valid"] == "1"
    assert report.attrib["lines-covered"] == "1"
    assert len(report.findall(".//class")) == 1


@pytest.mark.parametrize("domain", SOURCES)
@pytest.mark.parametrize("measured", [False, True], ids=["missing-data", "uncovered"])
def test_file_gate_rejects_missing_or_uncovered_file(
    corpus: Path, domain: str, measured: bool
) -> None:
    files = {"other.py": ("value = 1\n", [(-1, 1), (1, -1)])}
    if measured:
        files[SOURCES[domain]] = ("value = 1\n", [])
    else:
        source = corpus / SOURCES[domain]
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("value = 1\n", encoding="utf-8")
    _data(corpus, files)
    result = _run(corpus, _arguments(domain, per_file=True))
    assert result.returncode != 0, result.stdout + result.stderr
    assert (
        "Coverage failure" if measured else "No data to report"
    ) in result.stdout + result.stderr


def test_core_report_reapplies_collection_omissions(corpus: Path) -> None:
    _data(
        corpus,
        {
            SOURCES["core"]: ("value = 1\n", [(-1, 1), (1, -1)]),
            "src/invarlock/__init__.py": ("value = 1\n", []),
            "src/invarlock/test_helper.py": ("value = 1\n", []),
            "src/invarlock/tests/helper.py": ("value = 1\n", []),
        },
    )
    result = _run(corpus, _arguments("core"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "__init__.py" not in result.stdout
    assert "test_helper.py" not in result.stdout
    assert "tests/helper.py" not in result.stdout


@pytest.mark.parametrize("domain", SOURCES)
def test_domain_reports_preserve_their_own_line_exclusions(
    corpus: Path, domain: str
) -> None:
    _data(
        corpus,
        {
            SOURCES[domain]: (
                "def unsupported():\n    raise NotImplementedError\n\nvalue = 1\n",
                [(-1, 1), (1, 4), (4, -1)],
            )
        },
    )
    arguments = _arguments(domain)
    # Examples' XML is informational; its separate file ratchet still uses 95.
    result = _run(corpus, arguments)
    if domain in {"core", "examples"}:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0, result.stdout + result.stderr
    if domain == "examples":
        assert "66.67%" in result.stdout
    # The original addin/example file ratchets read pyproject.toml, whose
    # NotImplementedError exclusion differs from their aggregate configs.
    file_result = _run(corpus, _arguments(domain, per_file=True))
    assert (file_result.returncode == 0) == (domain in {"core", "addins", "examples"})


def test_example_threshold_cannot_be_diluted_by_other_domains(corpus: Path) -> None:
    _data(
        corpus,
        {
            SOURCES["examples"]: ("value = 1\n", []),
            SOURCES["core"]: (
                "\n".join(f"value_{i} = 1" for i in range(100)) + "\n",
                [(-1, 1), *((i, i + 1) for i in range(1, 100)), (100, -1)],
            ),
        },
    )
    block = MAKE.target("coverage-examples-report").text.replace("\\\n", " ")
    command = next(
        command
        for command in re.findall(r"-m coverage report ([^\n]+)", block)
        if "$$exemptions" in command
    )
    arguments = [
        sys.executable,
        "-m",
        "coverage",
        "report",
        *shlex.split(command.replace("$$exemptions", "examples/exempt.py")),
    ]
    result = _run(corpus, arguments)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "0.00%" in result.stdout
    assert SOURCES["core"] not in result.stdout


def test_combined_xml_retains_uncovered_branches(corpus: Path) -> None:
    source = "def choose(flag):\n    if flag:\n        return 1\n    return 2\n"
    _data(
        corpus,
        {SOURCES["core"]: (source, [(-1, 1), (1, -1), (-1, 2), (2, 3), (3, -1)])},
    )
    arguments = _arguments("core", "xml")
    result = _run(corpus, arguments)
    assert result.returncode != 0
    report_path = corpus / arguments[arguments.index("-o") + 1]
    report = ET.parse(report_path).getroot()
    assert report.attrib["branches-valid"] == "2"
    assert report.attrib["branches-covered"] == "1"
    result = _run(
        corpus,
        [
            sys.executable,
            str(ROOT / "scripts/checks/check_coverage_branch_rate.py"),
            str(report_path),
            "--minimum",
            "95",
        ],
    )
    assert result.returncode != 0
    assert "50.00% branch coverage" in result.stdout + result.stderr


@pytest.mark.parametrize("domain", ["core", "addins", "examples"])
def test_collection_discovers_unexecuted_package_files_after_checkout_move(
    tmp_path: Path, domain: str
) -> None:
    collected = tmp_path / "collected"
    collected.mkdir()
    _copy_configs(collected)
    for directory in ("src/invarlock", "addins", "scripts", "examples"):
        (collected / directory).mkdir(parents=True, exist_ok=True)
    unexecuted = collected / SOURCES[domain]
    unexecuted.parent.mkdir(parents=True, exist_ok=True)
    # Coverage's package discovery must find a file never imported or executed.
    for directory in unexecuted.parents:
        if directory == collected:
            break
        (directory / "__init__.py").touch()
    unexecuted.write_text("value = 1\n", encoding="utf-8")
    executed = unexecuted.with_name("executed.py")
    executed.write_text("value = 1\n", encoding="utf-8")
    config = "scripts/ci/coverage.coveragerc"
    result = _run(
        collected,
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            f"--rcfile={config}",
            str(executed.relative_to(collected)),
        ],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = _run(
        collected, [sys.executable, "-m", "coverage", "combine", f"--rcfile={config}"]
    )
    assert result.returncode == 0, result.stdout + result.stderr
    relocated = tmp_path / "relocated"
    collected.rename(relocated)
    data = coverage.CoverageData(basename=str(relocated / ".coverage"))
    data.read()
    assert SOURCES[domain] in data.measured_files()
    assert data.lines(SOURCES[domain]) == []
    _combine_at(relocated, relocated / ".coverage")
    data = coverage.CoverageData(basename=str(relocated / ".coverage"))
    data.read()
    assert str(relocated / SOURCES[domain]) in data.measured_files()
    assert data.lines(str(relocated / SOURCES[domain])) == []
    result = _run(relocated, _arguments(domain, per_file=True))
    assert result.returncode != 0
    assert "Coverage failure" in result.stdout
    arguments = _arguments(domain, "xml")
    result = _run(relocated, arguments)
    assert result.returncode == (0 if domain == "examples" else 2), (
        result.stdout + result.stderr
    )
    report = ET.parse(relocated / arguments[arguments.index("-o") + 1]).getroot()
    assert report.attrib["lines-valid"] == "2"
    assert report.attrib["lines-covered"] == "1"
