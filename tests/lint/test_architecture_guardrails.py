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
    for root in (
        REPO_ROOT / "src/invarlock/core",
        REPO_ROOT / "src/invarlock/reporting",
        REPO_ROOT / "src/invarlock/eval",
        REPO_ROOT / "src/invarlock/guards",
    ):
        for path in root.rglob("*.py"):
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


def test_eval_bench_is_not_a_shell_entrypoint() -> None:
    path = REPO_ROOT / "src/invarlock/eval/bench.py"
    text = _read_text(path)
    offenders = []
    for snippet in (
        "import argparse",
        "import sys",
        "sys.exit(",
        'if __name__ == "__main__"',
    ):
        if snippet in text:
            offenders.append(snippet)

    assert not offenders, "\n".join(offenders)


def test_dataset_diagnostics_do_not_encode_presentation_metadata() -> None:
    offenders: list[str] = []
    for path in (
        REPO_ROOT / "src/invarlock/eval/data_support.py",
        REPO_ROOT / "src/invarlock/core/run_provider_dataset_plan.py",
        REPO_ROOT / "src/invarlock/eval/window_planning.py",
    ):
        text = _read_text(path)
        for snippet in ('"tag"', '"emoji"', "def tag(", "def emoji("):
            if snippet in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} -> {snippet}")

    assert not offenders, "\n".join(offenders)


def test_retry_core_does_not_expose_legacy_notice_api() -> None:
    offenders: list[str] = []
    for path in (
        REPO_ROOT / "src/invarlock/core/retry.py",
        REPO_ROOT / "src/invarlock/core/run_retry_policy.py",
    ):
        text = _read_text(path)
        if "drain_notices" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, "\n".join(offenders)


def test_hf_adapter_loading_info_stays_structured() -> None:
    path = REPO_ROOT / "src/invarlock/adapters/hf_mixin.py"
    text = _read_text(path)
    offenders = []
    for snippet in (
        "Transformers load info",
        "logging.getLogger(__name__).warning(",
    ):
        if snippet in text:
            offenders.append(snippet)

    assert not offenders, "\n".join(offenders)


def test_quant_rtn_edit_does_not_embed_shell_output_helpers() -> None:
    path = REPO_ROOT / "src/invarlock/edits/quant_rtn.py"
    text = _read_text(path)
    offenders = []
    for snippet in (
        "import logging",
        "[EDIT]",
        "def _emit(",
        "def _configure_runtime(",
    ):
        if snippet in text:
            offenders.append(snippet)

    assert not offenders, "\n".join(offenders)


def test_bench_cli_owns_exit_boundary_without_internal_sys_exit() -> None:
    path = REPO_ROOT / "src/invarlock/cli/bench.py"
    text = _read_text(path)
    assert "sys.exit(" not in text
    assert "raise SystemExit(main())" in text


def test_runtime_verify_is_library_only() -> None:
    path = REPO_ROOT / "src/invarlock/runtime_verify.py"
    text = _read_text(path)
    offenders = []
    for snippet in (
        "import argparse",
        "print(",
        "SystemExit",
        'if __name__ == "__main__"',
    ):
        if snippet in text:
            offenders.append(snippet)

    assert not offenders, "\n".join(offenders)


def test_verify_contract_stays_typed_and_cli_owns_exit_rendering() -> None:
    path = REPO_ROOT / "src/invarlock/reporting/verify_contract.py"
    text = _read_text(path)
    offenders = []
    for snippet in (
        "status_code:",
        "messages: tuple[str",
        "resolve_command_exit_code",
        "configure_runtime_security(",
        "[red]",
        "[green]",
        "[yellow]",
    ):
        if snippet in text:
            offenders.append(snippet)

    assert not offenders, "\n".join(offenders)
    assert "class VerifyOutcome" in text
    assert "class VerifyDiagnostic" in text


def test_run_orchestrator_uses_named_event_types_not_generic_phase_envelope() -> None:
    path = REPO_ROOT / "src/invarlock/core/run_orchestrator.py"
    text = _read_text(path)
    offenders = []
    for snippet in (
        "class RunExecutionEvent",
        "phase: str",
        'RunExecutionEvent(phase=',
    ):
        if snippet in text:
            offenders.append(snippet)

    assert not offenders, "\n".join(offenders)
    for required in (
        "class RunLifecycleEvent",
        "class RunDiagnosticEvent",
        "class RunContextEvent",
        "class RunAggregateEvent",
    ):
        assert required in text


def test_edit_runtime_does_not_expose_shell_emit_controls() -> None:
    path = REPO_ROOT / "src/invarlock/core/api.py"
    text = _read_text(path)
    offenders = []
    for snippet in ("emit: bool", "Imperative shell runtime context"):
        if snippet in text:
            offenders.append(snippet)

    assert not offenders, "\n".join(offenders)


def test_runtime_attestation_does_not_embed_cli_flag_guidance() -> None:
    path = REPO_ROOT / "src/invarlock/runtime_attestation.py"
    text = _read_text(path)
    offenders = []
    for snippet in (
        "--allow-unattested-artifacts",
        "pass --allow-unattested-artifacts",
    ):
        if snippet in text:
            offenders.append(snippet)

    assert not offenders, "\n".join(offenders)


def test_subprocess_verifiers_use_timeouts() -> None:
    offenders: list[str] = []
    expectations = {
        REPO_ROOT / "src/invarlock/runtime_attestation.py": "timeout=",
        REPO_ROOT / "src/invarlock/proof_pack.py": "timeout=",
    }
    for path, required in expectations.items():
        text = _read_text(path)
        if "subprocess.run(" in text and required not in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, "\n".join(offenders)


def test_core_runtime_attestation_is_wrapper_only() -> None:
    path = REPO_ROOT / "src/invarlock/core/runtime_attestation.py"
    tree = ast.parse(_read_text(path), filename=str(path))

    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")

    banned = {
        "json",
        "shutil",
        "subprocess",
        "pathlib",
        "invarlock.runtime_security",
    }
    assert not banned.intersection(imports)

    text = _read_text(path)
    for snippet in (
        "load_runtime_manifest",
        "runtime_verifier_binary",
        "unattested_artifacts_allowed",
    ):
        assert snippet not in text


def test_run_report_contract_is_persistence_only() -> None:
    path = REPO_ROOT / "src/invarlock/reporting/run_report_contract.py"
    tree = ast.parse(_read_text(path), filename=str(path))
    target = None
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "persist_run_report_outputs"
        ):
            target = node
            break

    assert target is not None, "persist_run_report_outputs not found"

    arg_names = [arg.arg for arg in target.args.kwonlyargs]
    assert arg_names == [
        "report",
        "run_dir",
        "run_config",
        "telemetry",
        "save_telemetry_report_fn",
    ]

    text = _read_text(path)
    for snippet in (
        "console",
        "postprocess_and_summarize_fn",
        "subprocess",
        "shutil",
    ):
        assert snippet not in text


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
