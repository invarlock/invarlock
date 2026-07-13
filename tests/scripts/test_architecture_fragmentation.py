from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/checks/check_architecture_fragmentation.py"


def _policy(*, production_hard: int = 10, test_hard: int = 20) -> str:
    return f"""schema = "invarlock/architecture-policy/v1"
governed_roots = ["src/invarlock", "scripts", "tests"]

[[categories]]
name = "python_tests"
language = "python"
role = "test"
include = ["tests/**/*.py"]
soft_lines = 10
hard_lines = {test_hard}
soft_function_lines = 5
hard_function_lines = 10
soft_complexity = 2
hard_complexity = 4
soft_direct_files = 2
hard_direct_files = 3

[[categories]]
name = "shell_tests"
language = "shell"
role = "test"
include = ["tests/**/*.sh", "scripts/**/tests/**/*.sh", "scripts/**/*_test.sh"]
soft_lines = 10
hard_lines = {test_hard}
soft_direct_files = 2
hard_direct_files = 3

[[categories]]
name = "source_python"
language = "python"
role = "production"
include = ["src/invarlock/**/*.py"]
soft_lines = 5
hard_lines = {production_hard}
soft_function_lines = 5
hard_function_lines = 10
soft_complexity = 2
hard_complexity = 4
soft_direct_files = 2
hard_direct_files = 3

[[categories]]
name = "operational_python"
language = "python"
role = "production"
include = ["scripts/**/*.py"]
exclude = ["scripts/**/tests/**/*.py", "scripts/**/*_test.py"]
soft_lines = 5
hard_lines = {production_hard}
soft_function_lines = 5
hard_function_lines = 10
soft_complexity = 2
hard_complexity = 4
soft_direct_files = 2
hard_direct_files = 3

[[categories]]
name = "operational_shell"
language = "shell"
role = "production"
include = ["scripts/**/*.sh"]
exclude = ["scripts/**/tests/**/*.sh", "scripts/**/*_test.sh"]
soft_lines = 5
hard_lines = {production_hard}
soft_direct_files = 2
hard_direct_files = 3

[facades]
allowed_names = ["__init__.py", "__main__.py"]

[tiny_owners]
allowed_names = ["__init__.py", "__main__.py"]

[contract_owners]
include = ["src/invarlock/**/*_protocol.py", "src/invarlock/**/constants.py"]

[[dependency_rules]]
name = "library_no_cli"
include = ["src/invarlock/core/**/*.py", "src/invarlock/reporting/**/*.py"]
forbid_import_prefixes = ["invarlock.cli"]
forbid_call_prefixes = ["rich.console.Console.print", "typer.echo"]
"""


def _repo(
    tmp_path: Path, *, policy: str | None = None, debt: str | None = None
) -> Path:
    repo = tmp_path / "repo"
    (repo / "contracts").mkdir(parents=True)
    (repo / "contracts/architecture_policy.toml").write_text(
        policy or _policy(), encoding="utf-8"
    )
    (repo / "contracts/architecture_debt.toml").write_text(
        debt or 'schema = "invarlock/architecture-debt/v1"\nentries = []\n',
        encoding="utf-8",
    )
    return repo


def _run(
    repo: Path, *args: str
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json", "--repo-root", str(repo), *args],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    return result, json.loads(result.stdout)


def _write(repo: Path, rel_path: str, text: str) -> None:
    path = repo / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _kinds(payload: dict[str, object]) -> set[str]:
    blockers = payload["release_blockers"]
    assert isinstance(blockers, list)
    return {str(item["kind"]) for item in blockers}


def test_repository_policy_metrics_are_machine_readable() -> None:
    result, payload = _run(REPO_ROOT)
    assert payload["format_version"] == "architecture-fragmentation-v1"
    assert payload["policy_schema"] == "invarlock/architecture-policy/v1"
    assert payload["debt_schema"] == "invarlock/architecture-debt/v1"
    assert result.returncode == (0 if payload["release_ready"] else 1)
    blockers = payload["release_blockers"]
    assert isinstance(blockers, list)
    assert payload["release_blocker_count"] == len(blockers)
    assert all("kind" in item and "key" in item and "path" in item for item in blockers)
    tiny_paths = {item["path"] for item in blockers if item["kind"] == "tiny_owner"}
    assert not tiny_paths & {
        "src/invarlock/cli/run_runtime_retry.py",
        "src/invarlock/cli/run_runtime_snapshot.py",
        "src/invarlock/guards/exact_svd.py",
        "src/invarlock/cli/constants.py",
        "src/invarlock/training_protocol.py",
    }
    facade_paths = {
        item["path"] for item in blockers if item["kind"] == "reexport_facade"
    }
    assert not facade_paths & {
        "src/invarlock/cli/constants.py",
        "src/invarlock/training_protocol.py",
    }
    invalid_contract_paths = {
        item["path"] for item in blockers if item["kind"] == "invalid_contract_owner"
    }
    assert not invalid_contract_paths & {
        "src/invarlock/cli/constants.py",
        "src/invarlock/training_protocol.py",
    }


def test_direct_root_files_match_globstar_categories(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write(repo, "src/invarlock/direct.py", "x = 1\ny = 2\n")
    _write(repo, "scripts/direct.py", "x = 1\ny = 2\n")
    _write(repo, "tests/direct.py", "def test_x():\n    assert True\n")
    _write(repo, "tests/direct.sh", "#!/bin/sh\ntrue\n")
    _, payload = _run(repo)
    assert payload["uncategorized_code"] == []
    assert payload["category_file_counts"] == {
        "operational_python": 1,
        "python_tests": 1,
        "shell_tests": 1,
        "source_python": 1,
    }


def test_category_excludes_classify_script_tests_without_precedence(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path, policy=_policy(production_hard=10, test_hard=12))
    _write(repo, "scripts/tool/tests/check.sh", "true\n" * 11)
    _, payload = _run(repo)
    assert payload["category_file_counts"] == {"shell_tests": 1}
    assert "file_lines" not in _kinds(payload)


@pytest.mark.parametrize(
    "policy",
    (
        _policy().replace(
            'allowed_names = ["__init__.py", "__main__.py"]',
            'allowed_names = "__init__.py"',
            1,
        ),
        _policy().replace(
            'governed_roots = ["src/invarlock", "scripts", "tests"]',
            'governed_roots = ["/src/invarlock", "scripts", "tests"]',
        ),
        _policy().replace(
            'governed_roots = ["src/invarlock", "scripts", "tests"]',
            'governed_roots = ["src/invarlock", "scripts", "tests", "tools"]',
        ),
        _policy().replace(
            'include = ["src/invarlock/**/*.py"]',
            'include = ["src/invarlock/**/*.sh"]',
        ),
        _policy().replace(
            'include = ["src/invarlock/**/*.py"]',
            'include = ["src/invarlock/**/*.py"]\nexclude = ["src/invarlock/**/*.py"]',
        ),
        _policy()
        + """
[[dependency_rules]]
name = "library_no_cli"
include = ["src/invarlock/**/*.py"]
forbid_import_prefixes = ["invarlock.cli"]
""",
    ),
)
def test_malformed_policy_contracts_fail(tmp_path: Path, policy: str) -> None:
    repo = _repo(tmp_path, policy=policy)
    _write(repo, "src/invarlock/owner.py", "def owner():\n    return 1\n")
    _, payload = _run(repo)
    assert _kinds(payload) == {"policy_error"}


def test_overlapping_categories_fail_instead_of_using_order(tmp_path: Path) -> None:
    policy = _policy().replace(
        'exclude = ["scripts/**/tests/**/*.sh", "scripts/**/*_test.sh"]',
        "exclude = []",
    )
    repo = _repo(tmp_path, policy=policy)
    _write(repo, "scripts/tool/tests/check.sh", "true\n")
    _, payload = _run(repo)
    assert _kinds(payload) == {"policy_error"}


@pytest.mark.parametrize(
    "source",
    (
        "from elsewhere import value\n__all__ = ['value']\n",
        "from elsewhere import *\n",
    ),
)
def test_explicit_and_wildcard_reexport_facades_fail(
    tmp_path: Path, source: str
) -> None:
    repo = _repo(tmp_path)
    _write(repo, "src/invarlock/facade.py", source)
    _, payload = _run(repo)
    assert "reexport_facade" in _kinds(payload)


def test_facade_padding_with_inert_statements_still_fails(tmp_path: Path) -> None:
    repo = _repo(tmp_path, policy=_policy(production_hard=50))
    source = """from typing import TYPE_CHECKING
from elsewhere import value

__version__ = "1"
SENTINEL = object()
pass
if TYPE_CHECKING:
    from elsewhere import OnlyForTyping
__all__ = ["value"]
"""
    _write(repo, "src/invarlock/facade.py", source)
    _, payload = _run(repo)
    assert "reexport_facade" in _kinds(payload)
    assert "tiny_owner" not in _kinds(payload)


def test_multi_constant_contract_is_a_real_owner(tmp_path: Path) -> None:
    repo = _repo(tmp_path, policy=_policy(production_hard=50))
    source = """from contracts import BASE_SCHEMA
FIRST_SCHEMA = BASE_SCHEMA
SECOND_SCHEMA = "second-v1"
THIRD_FORMAT: str = "third-v1"
__all__ = ["FIRST_SCHEMA", "SECOND_SCHEMA", "THIRD_FORMAT"]
"""
    _write(repo, "src/invarlock/constants.py", source)
    _, payload = _run(repo)
    assert not {"reexport_facade", "tiny_owner"} & _kinds(payload)


def test_uppercase_padding_does_not_make_a_general_module_an_owner(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _write(repo, "src/invarlock/padding.py", "ALPHA = 1\nBETA = 2\nGAMMA = 3\n")
    _, payload = _run(repo)
    assert "tiny_owner" in _kinds(payload)


def test_contract_owner_with_runtime_logic_fails_strict_validation(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path, policy=_policy(production_hard=50))
    source = "SCHEMA = 'x-v1'\ndef execute():\n    return 1\n"
    _write(repo, "src/invarlock/runtime_protocol.py", source)
    _, payload = _run(repo)
    assert "invalid_contract_owner" in _kinds(payload)


def test_contract_owner_rejects_executable_assignment_rhs(tmp_path: Path) -> None:
    repo = _repo(tmp_path, policy=_policy(production_hard=50))
    source = "import os\nSCHEMA = os.system('unexpected')\n"
    _write(repo, "src/invarlock/runtime_protocol.py", source)
    _, payload = _run(repo)
    assert "invalid_contract_owner" in _kinds(payload)


@pytest.mark.parametrize(
    "rhs",
    ("EVIL.payload", "EVIL + EVIL", "-EVIL"),
)
def test_contract_owner_rejects_descriptor_and_operator_rhs(
    tmp_path: Path, rhs: str
) -> None:
    repo = _repo(tmp_path, policy=_policy(production_hard=50))
    source = f"from hostile import EVIL\nSCHEMA = {rhs}\n"
    _write(repo, "src/invarlock/runtime_protocol.py", source)
    _, payload = _run(repo)
    assert "invalid_contract_owner" in _kinds(payload)


def test_contract_owner_allows_signed_numeric_literals(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    source = "LOWER_BOUND = -1\nUPPER_BOUND = +1\nSCHEMA = 'bounds-v1'\n"
    _write(repo, "src/invarlock/bounds_protocol.py", source)
    _, payload = _run(repo)
    assert "invalid_contract_owner" not in _kinds(payload)


def test_dummy_markers_and_version_sentinel_do_not_pad_a_facade(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path, policy=_policy(production_hard=50))
    source = """from contracts import value
PUBLIC_DUMMY = value
_PRIVATE_MARKER = object()
SENTINEL = object()
VERSION = "1"
__version__ = "1"
"""
    _write(repo, "src/invarlock/facade.py", source)
    _, payload = _run(repo)
    facades = [
        item
        for item in payload["release_blockers"]
        if item["path"] == "src/invarlock/facade.py"
        and item["kind"] in {"reexport_facade", "tiny_owner"}
    ]
    assert [item["kind"] for item in facades] == ["reexport_facade"]


def test_entrypoint_wrapper_is_a_real_owner(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    source = "from worker import main\nif __name__ == '__main__':\n    raise SystemExit(main())\n"
    _write(repo, "scripts/entry.py", source)
    _, payload = _run(repo)
    assert not {"reexport_facade", "tiny_owner"} & _kinds(payload)


def test_trivial_delegate_function_does_not_pad_a_facade(tmp_path: Path) -> None:
    repo = _repo(tmp_path, policy=_policy(production_hard=50))
    source = "from worker import execute\ndef run(*args, **kwargs):\n    return execute(*args, **kwargs)\n"
    _write(repo, "scripts/delegate.py", source)
    _, payload = _run(repo)
    assert "reexport_facade" in _kinds(payload)
    assert "tiny_owner" not in _kinds(payload)


@pytest.mark.parametrize(
    "source",
    (
        "from worker import execute\ndef run(*args, **kwargs):\n    marker = 'padding'\n    result = execute(*args, **kwargs)\n    return result\n",
        "from worker import execute\ndef _helper(*args, **kwargs):\n    return execute(*args, **kwargs)\ndef run(*args, **kwargs):\n    return _helper(*args, **kwargs)\n",
        "from worker import execute\nclass Runner:\n    def run(self, *args, **kwargs):\n        result = execute(*args, **kwargs)\n        return result\n",
    ),
)
def test_recursive_delegate_padding_does_not_establish_ownership(
    tmp_path: Path, source: str
) -> None:
    repo = _repo(tmp_path, policy=_policy(production_hard=50))
    _write(repo, "scripts/delegate.py", source)
    _, payload = _run(repo)
    assert "reexport_facade" in _kinds(payload)
    assert "tiny_owner" not in _kinds(payload)


def test_imported_constructor_chain_is_a_delegate(tmp_path: Path) -> None:
    repo = _repo(tmp_path, policy=_policy(production_hard=50))
    source = (
        "from worker import Adapter\ndef run(value):\n    return Adapter().run(value)\n"
    )
    _write(repo, "scripts/delegate.py", source)
    _, payload = _run(repo)
    assert "reexport_facade" in _kinds(payload)


def test_private_noop_padding_does_not_establish_ownership(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write(repo, "src/invarlock/padding.py", "def _padding():\n    return 1\n")
    _, payload = _run(repo)
    assert "tiny_owner" in _kinds(payload)


def test_transform_then_delegate_is_cohesive(tmp_path: Path) -> None:
    repo = _repo(tmp_path, policy=_policy(production_hard=50))
    source = "from worker import execute\ndef run(value):\n    transformed = value + 1\n    return execute(transformed)\n"
    _write(repo, "scripts/transform.py", source)
    _, payload = _run(repo)
    assert not {"reexport_facade", "tiny_owner"} & _kinds(payload)


def test_protocol_declaration_is_not_a_tiny_owner(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write(
        repo,
        "src/invarlock/protocols.py",
        "from typing import Protocol\nclass Reader(Protocol):\n    pass\n",
    )
    _, payload = _run(repo)
    assert "tiny_owner" not in _kinds(payload)
    assert "reexport_facade" not in _kinds(payload)


@pytest.mark.parametrize(
    "source",
    (
        "from typing import TypedDict\nclass Row(TypedDict):\n    value: int\n",
        "from typing import NamedTuple\nclass Row(NamedTuple):\n    value: int\n",
        "from typing import TypeAlias\nRow: TypeAlias = dict[str, int]\n",
        "from dataclasses import dataclass\n@dataclass\nclass Row:\n    value: int\n",
        "from enum import Enum\nclass Value(Enum):\n    ONE = 1\n",
    ),
)
def test_declaration_only_modules_are_real_owners(tmp_path: Path, source: str) -> None:
    repo = _repo(tmp_path)
    _write(repo, "src/invarlock/declarations.py", source)
    _, payload = _run(repo)
    assert "tiny_owner" not in _kinds(payload)


def test_padding_does_not_make_a_tiny_owner_substantive(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write(repo, "src/invarlock/padded.py", "x = 1\n" + "\n# padding\n" * 20)
    _, payload = _run(repo)
    assert "tiny_owner" in _kinds(payload)


def test_package_concentration_counts_only_direct_files(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write(repo, "src/invarlock/pkg/constants.py", "ONE = 1\nTWO = 2\nTHREE = 3\n")
    for index in range(3):
        _write(repo, f"src/invarlock/pkg/m{index}.py", "x = 1\ny = 2\n")
    for index in range(4):
        _write(repo, f"src/invarlock/pkg/nested/m{index}.py", "x = 1\ny = 2\n")
    _, payload = _run(repo)
    blockers = [
        item
        for item in payload["release_blockers"]
        if item["kind"] == "package_concentration"
    ]
    assert {item["path"] for item in blockers} == {
        "src/invarlock/pkg",
        "src/invarlock/pkg/nested",
    }
    assert all(item["actual"] == 4 for item in blockers)


def test_function_span_and_complexity_are_independent_findings(tmp_path: Path) -> None:
    repo = _repo(tmp_path, policy=_policy(production_hard=50))
    source = "def tangled(a, b, c, d):\n    if a:\n        pass\n    if b:\n        pass\n    if c:\n        pass\n    if d:\n        pass\n    value = 1\n    return value\n"
    _write(repo, "src/invarlock/tangled.py", source)
    _, payload = _run(repo)
    assert {"function_lines", "function_complexity"} <= _kinds(payload)


def test_relative_imports_are_resolved_for_dependency_rules(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write(repo, "src/invarlock/core/owner.py", "from ..cli import app\nx = app\n")
    _, payload = _run(repo)
    assert "dependency_direction" in _kinds(payload)


@pytest.mark.parametrize(
    "source",
    (
        "from invarlock import cli\nx = cli\n",
        "from .. import cli\nx = cli\n",
        "from typer import echo as emit\ndef run():\n    emit('x')\n",
        "import typer as command_ui\ndef run():\n    command_ui.echo('x')\n",
        "from rich.console import Console\nconsole = Console()\ndef run():\n    console.print('x')\n",
    ),
)
def test_dependency_aliases_are_resolved(tmp_path: Path, source: str) -> None:
    repo = _repo(tmp_path, policy=_policy(production_hard=50))
    _write(repo, "src/invarlock/core/owner.py", source)
    _, payload = _run(repo)
    assert "dependency_direction" in _kinds(payload)


def test_unrelated_local_console_is_not_cli_output(tmp_path: Path) -> None:
    repo = _repo(tmp_path, policy=_policy(production_hard=50))
    source = "class Console:\n    def print(self, value):\n        return value\nconsole = Console()\ndef run():\n    return console.print('x')\n"
    _write(repo, "src/invarlock/core/owner.py", source)
    _, payload = _run(repo)
    assert "dependency_direction" not in _kinds(payload)


@pytest.mark.parametrize(
    "source",
    (
        "import typer\ndef run(typer):\n    return typer.echo('local')\n",
        "import typer\nclass Local:\n    def echo(self, value):\n        return value\ndef run():\n    typer = Local()\n    return typer.echo('local')\n",
        "import typer\nclass Local:\n    def echo(self, value):\n        return value\ntyper = Local()\ndef run():\n    return typer.echo('local')\n",
        "import typer\ndef run():\n    typer.echo('unbound-before-local')\n    typer = object()\n",
    ),
)
def test_shadowed_import_alias_calls_do_not_create_false_dependency_findings(
    tmp_path: Path, source: str
) -> None:
    repo = _repo(tmp_path, policy=_policy(production_hard=50))
    _write(repo, "src/invarlock/core/owner.py", source)
    _, payload = _run(repo)
    assert "dependency_direction" not in _kinds(payload)


@pytest.mark.parametrize(
    "source",
    (
        "import typer\ndef run(items):\n    return [typer.echo for typer in items]\n",
        "import typer\ndef run():\n    typer.echo('compile-time-local')\n    try:\n        return None\n    except Exception as typer:\n        return typer\n",
        "import typer\ndef run(value):\n    typer.echo('compile-time-local')\n    match value:\n        case {'ui': typer}:\n            return typer\n",
    ),
)
def test_comprehension_exception_and_match_bindings_are_scope_local(
    tmp_path: Path, source: str
) -> None:
    repo = _repo(tmp_path, policy=_policy(production_hard=50))
    _write(repo, "src/invarlock/core/owner.py", source)
    _, payload = _run(repo)
    assert "dependency_direction" not in _kinds(payload)


def test_global_import_alias_call_before_reassignment_remains_a_dependency(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path, policy=_policy(production_hard=50))
    source = "import typer\ndef run():\n    global typer\n    typer.echo('real')\n    typer = object()\n"
    _write(repo, "src/invarlock/core/owner.py", source)
    _, payload = _run(repo)
    assert "dependency_direction" in _kinds(payload)


def test_comprehension_shadow_does_not_erase_prior_imported_call(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path, policy=_policy(production_hard=50))
    source = "import typer\ndef run(items):\n    typer.echo('real')\n    return [typer.echo for typer in items]\n"
    _write(repo, "src/invarlock/core/owner.py", source)
    _, payload = _run(repo)
    assert "dependency_direction" in _kinds(payload)


@pytest.mark.parametrize(
    "source",
    (
        "def run():\n    from rich.console import Console\n    output = Console()\n    output.print('x')\n",
        "from rich.console import Console\ndef run():\n    Console().print('x')\n",
    ),
)
def test_local_and_chained_console_construction_is_resolved(
    tmp_path: Path, source: str
) -> None:
    repo = _repo(tmp_path, policy=_policy(production_hard=50))
    _write(repo, "src/invarlock/core/owner.py", source)
    _, payload = _run(repo)
    assert "dependency_direction" in _kinds(payload)


def test_file_line_hard_limit_is_inclusive(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write(repo, "src/invarlock/at_limit.py", "x = 1\n" * 10)
    _, payload = _run(repo)
    assert "file_lines" not in _kinds(payload)
    _write(repo, "src/invarlock/at_limit.py", "x = 1\n" * 11)
    _, payload = _run(repo)
    assert "file_lines" in _kinds(payload)


def test_nested_lambda_and_comprehension_complexity_is_measured(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path, policy=_policy(production_hard=50))
    source = "def outer():\n    inner = lambda xs: [x for x in xs if x if x > 1] if xs and len(xs) > 1 else []\n    return inner\n"
    _write(repo, "src/invarlock/complexity.py", source)
    _, payload = _run(repo)
    findings = [
        item
        for item in payload["release_blockers"]
        if item["kind"] == "function_complexity"
    ]
    assert any("<lambda>" in str(item["symbol"]) for item in findings)


def test_generated_exclusion_declarations_are_rejected(tmp_path: Path) -> None:
    policy = _policy() + "\n[generated_code]\ndeclarations = []\n"
    repo = _repo(tmp_path, policy=policy)
    _write(repo, "src/invarlock/generated/huge.py", "x = 1\n" * 100)
    _, payload = _run(repo)
    assert _kinds(payload) == {"policy_error"}


def test_malformed_debt_fails_as_policy_error(tmp_path: Path) -> None:
    repo = _repo(
        tmp_path,
        debt='schema = "invarlock/architecture-debt/v1"\n[[entries]]\nkey = "file_lines:src/invarlock/a.py"\n',
    )
    _write(repo, "src/invarlock/a.py", "x = 1\n" * 11)
    result, payload = _run(repo)
    assert result.returncode == 1
    assert _kinds(payload) == {"policy_error"}


@pytest.mark.parametrize(
    ("expires", "key", "expected"),
    (
        ("2025-01-01", "file_lines:src/invarlock/a.py", "expired_debt"),
        ("2026-01-01", "file_lines:src/invarlock/a.py", "expired_debt"),
        ("2027-01-01", "file_lines:src/invarlock/missing.py", "stale_debt"),
    ),
)
def test_expired_and_stale_debt_fail(
    tmp_path: Path, expires: str, key: str, expected: str
) -> None:
    debt = f'''schema = "invarlock/architecture-debt/v1"
[[entries]]
key = "{key}"
ceiling = 20
owner = "maintainers"
reason = "temporary migration"
expires = {expires}
'''
    repo = _repo(tmp_path, debt=debt)
    _write(repo, "src/invarlock/a.py", "x = 1\n" * 11)
    _, payload = _run(repo, "--as-of", "2026-01-01")
    assert expected in _kinds(payload)


def test_nonempty_debt_never_suppresses_release_findings(tmp_path: Path) -> None:
    debt = """schema = "invarlock/architecture-debt/v1"
[[entries]]
key = "file_lines:src/invarlock/a.py"
ceiling = 12
owner = "maintainers"
reason = "temporary migration"
expires = 2027-01-01
"""
    repo = _repo(tmp_path, debt=debt)
    _write(repo, "src/invarlock/a.py", "x = 1\n" * 11)
    _, payload = _run(repo, "--as-of", "2026-01-01")
    assert payload["suppressed_debt_count"] == 0
    assert {"file_lines", "stale_debt_ceiling"} <= _kinds(payload)
    stale = next(
        item
        for item in payload["release_blockers"]
        if item["kind"] == "stale_debt_ceiling"
    )
    assert stale["comparison"] == "<"

    _write(repo, "src/invarlock/a.py", "x = 1\n" * 13)
    _, payload = _run(repo, "--as-of", "2026-01-01")
    assert "debt_regression" in _kinds(payload)


def test_human_output_uses_each_finding_comparator(tmp_path: Path) -> None:
    debt = """schema = "invarlock/architecture-debt/v1"
[[entries]]
key = "file_lines:src/invarlock/a.py"
ceiling = 12
owner = "maintainers"
reason = "temporary migration"
expires = 2027-01-01
"""
    repo = _repo(tmp_path, debt=debt)
    _write(repo, "src/invarlock/a.py", "x = 1\n" * 11)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--as-of",
            "2026-01-01",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert "(11 > 10)" in result.stdout
    assert "(11 < 12)" in result.stdout


def test_debt_at_exact_ceiling_is_still_new_release_blocking_debt(
    tmp_path: Path,
) -> None:
    debt = """schema = "invarlock/architecture-debt/v1"
[[entries]]
key = "file_lines:src/invarlock/a.py"
ceiling = 11
owner = "maintainers"
reason = "temporary migration"
expires = 2027-01-01
"""
    repo = _repo(tmp_path, debt=debt)
    _write(repo, "src/invarlock/a.py", "x = 1\n" * 11)
    _, payload = _run(repo, "--as-of", "2026-01-01")
    assert {"file_lines", "new_debt"} <= _kinds(payload)


def test_duplicate_debt_keys_fail(tmp_path: Path) -> None:
    entry = """
[[entries]]
key = "file_lines:src/invarlock/a.py"
ceiling = 11
owner = "maintainers"
reason = "temporary migration"
expires = 2027-01-01
"""
    repo = _repo(
        tmp_path,
        debt='schema = "invarlock/architecture-debt/v1"\n' + entry + entry,
    )
    _write(repo, "src/invarlock/a.py", "x = 1\n" * 11)
    _, payload = _run(repo, "--as-of", "2026-01-01")
    assert "duplicate_debt" in _kinds(payload)
