from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT_INIT_FILES = (
    REPO_ROOT / "src/invarlock/__init__.py",
    REPO_ROOT / "src/invarlock/core/__init__.py",
    REPO_ROOT / "src/invarlock/adapters/__init__.py",
    REPO_ROOT / "src/invarlock/guards/__init__.py",
)
OWNER_LAYER_ROOTS = (
    REPO_ROOT / "src/invarlock/core",
    REPO_ROOT / "src/invarlock/reporting",
)
REMOVED_REPORTING_MODULES = (
    REPO_ROOT / "src/invarlock/reporting/report_builder.py",
    REPO_ROOT / "src/invarlock/reporting/report_make_support.py",
)
RUN_COMMAND_PATH = REPO_ROOT / "src/invarlock/cli/commands/run.py"
RUN_EXECUTION_PATH = REPO_ROOT / "src/invarlock/cli/run_execution.py"
REPORT_FILES_PATH = REPO_ROOT / "src/invarlock/reporting/report_files.py"
METRICS_PATH = REPO_ROOT / "src/invarlock/eval/metrics.py"
METRICS_LENS_PATH = REPO_ROOT / "src/invarlock/eval/metrics_lens.py"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_package_roots_do_not_reintroduce_lazy_export_mechanisms() -> None:
    banned_snippets = (
        "def __getattr__",
        "__all__.extend(",
        "importlib.import_module(",
        "find_spec(",
    )
    offenders: list[str] = []

    for path in PACKAGE_ROOT_INIT_FILES:
        text = _read_text(path)
        for snippet in banned_snippets:
            if snippet in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} -> {snippet}")

    assert not offenders, "\n".join(offenders)


def test_source_code_does_not_reference_rmt_legacy() -> None:
    offenders: list[str] = []
    for path in (REPO_ROOT / "src").rglob("*.py"):
        text = _read_text(path)
        if "rmt_legacy" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, "\n".join(offenders)


def test_removed_reporting_facades_do_not_reappear() -> None:
    offenders: list[str] = []
    for path in REMOVED_REPORTING_MODULES:
        if path.exists():
            offenders.append(f"unexpected file present: {path.relative_to(REPO_ROOT)}")

    banned_refs = (
        "invarlock.reporting.report_builder",
        "invarlock.reporting.report_make_support",
        "src/invarlock/reporting/report_builder.py",
        "src/invarlock/reporting/report_make_support.py",
    )
    for root in (REPO_ROOT / "src", REPO_ROOT / "docs", REPO_ROOT / "scripts"):
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".md"}:
                continue
            text = _read_text(path)
            for banned in banned_refs:
                if banned in text:
                    offenders.append(f"{path.relative_to(REPO_ROOT)} -> {banned}")

    assert not offenders, "\n".join(sorted(offenders))


def test_run_command_shell_does_not_use_legacy_dependency_map_injection() -> None:
    text = _read_text(RUN_COMMAND_PATH)
    banned_snippets = (
        "_build_run_command_deps",
        "_build_run_execution_deps",
        "_execute_cli_run_request",
        "RunExecutionDeps",
        "deps_builder=",
        "run_impl=_run_command_impl",
        "run_command_impl as _run_command_impl",
    )
    offenders = [snippet for snippet in banned_snippets if snippet in text]
    assert not offenders, "\n".join(offenders)


def test_run_execution_owner_does_not_import_run_command_module() -> None:
    text = _read_text(RUN_EXECUTION_PATH)
    banned_snippets = (
        "from invarlock.cli.commands import run as run_mod",
        "invarlock.cli.commands.run",
        "services=run_mod",
        "def _build_run_execution_services(run_mod",
    )
    offenders = [snippet for snippet in banned_snippets if snippet in text]
    tree = ast.parse(text, filename=str(RUN_EXECUTION_PATH))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "invarlock.cli.run_service"
        ):
            offenders.append("invarlock.cli.run_service")
    assert not offenders, "\n".join(sorted(offenders))


def test_run_service_facade_is_removed() -> None:
    path = REPO_ROOT / "src/invarlock/cli/run_service.py"
    assert not path.exists()


def test_report_files_remains_persistence_only() -> None:
    text = _read_text(REPORT_FILES_PATH)
    banned_snippets = (
        "from .report_make import",
        "make_report(",
        "from .report_schema import validate_report",
        "baseline:",
    )
    offenders = [snippet for snippet in banned_snippets if snippet in text]
    assert not offenders, "\n".join(offenders)


def test_owner_layers_do_not_import_cli_modules() -> None:
    offenders: set[str] = set()
    for root in OWNER_LAYER_ROOTS:
        for path in root.rglob("*.py"):
            text = _read_text(path)
            if "from ..cli" in text or "from .cli" in text:
                offenders.add(f"{path.relative_to(REPO_ROOT)} -> relative cli import")

            tree = ast.parse(text, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "invarlock.cli" or alias.name.startswith(
                            "invarlock.cli."
                        ):
                            offenders.add(
                                f"{path.relative_to(REPO_ROOT)} -> import {alias.name}"
                            )
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module == "invarlock.cli" or module.startswith("invarlock.cli."):
                        offenders.add(
                            f"{path.relative_to(REPO_ROOT)} -> from {module} import ..."
                        )

    assert not offenders, "\n".join(sorted(offenders))


def test_owner_layers_do_not_print_directly() -> None:
    offenders: set[str] = set()
    excluded_files = {
        REPO_ROOT / "src/invarlock/eval/bench.py",
    }
    for root in (
        REPO_ROOT / "src/invarlock/core",
        REPO_ROOT / "src/invarlock/reporting",
        REPO_ROOT / "src/invarlock/eval",
        REPO_ROOT / "src/invarlock/guards",
    ):
        for path in root.rglob("*.py"):
            if path in excluded_files:
                continue
            tree = ast.parse(_read_text(path), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    if isinstance(func, ast.Name) and func.id == "print":
                        offenders.add(f"{path.relative_to(REPO_ROOT)} -> print(")
                    elif (
                        isinstance(func, ast.Attribute)
                        and func.attr == "print"
                        and isinstance(func.value, ast.Name)
                        and func.value.id == "console"
                    ):
                        offenders.add(
                            f"{path.relative_to(REPO_ROOT)} -> console.print("
                        )
                    elif (
                        isinstance(func, ast.Attribute)
                        and func.attr == "echo"
                        and isinstance(func.value, ast.Name)
                        and func.value.id == "typer"
                    ):
                        offenders.add(f"{path.relative_to(REPO_ROOT)} -> typer.echo(")

    assert not offenders, "\n".join(sorted(offenders))


def test_lens_metrics_entrypoint_requires_metrics_config() -> None:
    tree = ast.parse(_read_text(METRICS_LENS_PATH), filename=str(METRICS_LENS_PATH))
    target = None
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "calculate_lens_metrics_for_model"
        ):
            target = node
            break

    assert target is not None, "calculate_lens_metrics_for_model not found"

    positional = [arg.arg for arg in target.args.args]
    kwonly = [arg.arg for arg in target.args.kwonlyargs]
    all_args = positional + kwonly

    assert positional == ["model", "dataloader"]
    assert kwonly == ["config"]
    assert "oracle_windows" not in all_args
    assert "device" not in all_args
