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
    REPO_ROOT / "src/invarlock/reporting/verify_checks.py",
)
RUN_COMMAND_PATH = REPO_ROOT / "src/invarlock/cli/commands/run.py"
RUN_EXECUTION_PATH = REPO_ROOT / "src/invarlock/cli/run_execution.py"
REPORT_FILES_PATH = REPO_ROOT / "src/invarlock/reporting/report_files.py"
METRICS_PATH = REPO_ROOT / "src/invarlock/eval/metrics.py"
METRICS_LENS_PATH = REPO_ROOT / "src/invarlock/eval/metrics_lens.py"
CONFIG_RUNTIME_PATH = REPO_ROOT / "src/invarlock/core/config_runtime.py"
CONFIG_LOADER_PATH = REPO_ROOT / "src/invarlock/core/config_loader.py"
RUNTIME_SECURITY_PATH = REPO_ROOT / "src/invarlock/runtime_security.py"
BROAD_EXCEPTION = "except " + "Exception"
BROAD_EXCEPTION_AS_ERROR = BROAD_EXCEPTION + " as error"
BROAD_EXCEPTION_RETURN_ZERO = "except " + "Exception:\\n                return 0.0"


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
        "invarlock.reporting.verify_checks",
        "src/invarlock/reporting/report_builder.py",
        "src/invarlock/reporting/report_make_support.py",
        "src/invarlock/reporting/verify_checks.py",
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


def test_run_execution_services_use_pure_runtime_helpers() -> None:
    text = _read_text(RUN_EXECUTION_PATH)
    required_snippets = (
        "execute_guarded_run=run_runtime_exec_mod.execute_guarded_run",
        "init_retry_controller=run_runtime_exec_mod.init_retry_controller",
        "load_model_with_cfg=run_runtime_exec_mod.load_model_with_cfg",
        "run_bare_control=run_runtime_exec_mod.run_bare_control",
        "typed_failures=True",
    )
    banned_snippets = (
        "execute_guarded_run=_execute_guarded_run_with_runtime_deps",
        "init_retry_controller=_init_retry_controller_with_runtime_deps",
        "load_model_with_cfg=_load_model_with_cfg_with_runtime_deps",
        "run_bare_control=_run_bare_control_with_runtime_deps",
        "validate_and_harvest_baseline_schedule=(\n            run_pairing_mod.validate_and_harvest_baseline_schedule",
    )
    missing = [snippet for snippet in required_snippets if snippet not in text]
    offenders = [snippet for snippet in banned_snippets if snippet in text]
    assert not missing, "\n".join(missing)
    assert not offenders, "\n".join(offenders)


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
            try:
                tree = ast.parse(_read_text(path), filename=str(path))
            except SyntaxError:
                continue
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


def test_config_runtime_is_schema_only_and_loader_lives_elsewhere() -> None:
    runtime_text = _read_text(CONFIG_RUNTIME_PATH)
    loader_text = _read_text(CONFIG_LOADER_PATH)
    legacy_path = REPO_ROOT / "src/invarlock/core/config_dependencies.py"

    banned_runtime_snippets = (
        "def load_config(",
        "def load_tiers(",
        "def apply_profile(",
        "_load_runtime_yaml",
        "_load_raw_config_payload",
        "INVARLOCK_CONFIG_ROOT",
        "yaml.safe_load(",
        "_ires.files(",
    )
    required_loader_snippets = (
        "def load_config(",
        "def load_tiers(",
        "def apply_profile(",
        "def _load_runtime_yaml(",
        "_load_raw_config_payload",
    )

    runtime_offenders = [
        snippet for snippet in banned_runtime_snippets if snippet in runtime_text
    ]
    missing_loader = [
        snippet for snippet in required_loader_snippets if snippet not in loader_text
    ]

    assert not runtime_offenders, "\n".join(runtime_offenders)
    assert not missing_loader, "\n".join(missing_loader)
    assert not legacy_path.exists()


def test_runtime_security_facade_keeps_only_public_surface() -> None:
    text = _read_text(RUNTIME_SECURITY_PATH)
    banned_snippets = (
        "os = _helpers.os",
        "Path = _helpers.Path",
        "shutil = _helpers.shutil",
        "subprocess = _helpers.subprocess",
        "_coerce_bool",
        "_inspect_container_image",
        "_config_digest",
        "_runtime_flag_value",
        "serialize_canonical_json",
    )
    offenders = [snippet for snippet in banned_snippets if snippet in text]
    assert not offenders, "\n".join(offenders)


def test_python_312_floor_removes_typed_dict_compat_shims() -> None:
    pyproject_text = _read_text(REPO_ROOT / "pyproject.toml")
    assert 'requires-python = ">=3.12"' in pyproject_text
    assert "typing_extensions" not in pyproject_text

    for path in (
        REPO_ROOT / "src/invarlock/guards/policies.py",
        REPO_ROOT / "src/invarlock/reporting/report_types.py",
    ):
        text = _read_text(path)
        assert "typing_extensions" not in text
        assert "except ImportError" not in text


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
        for snippet in ("tag: str", "emoji: str", "emit(", "event_fn="):
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


def test_retry_and_evaluate_contracts_use_typed_outcomes() -> None:
    retry_path = REPO_ROOT / "src/invarlock/core/run_retry_policy.py"
    retry_text = _read_text(retry_path)
    for required in (
        "status: str",
        "validation_gates: tuple[str, ...]",
        "diagnostics: tuple[RetryDiagnostic, ...]",
        "error: RetryDiagnostic | None",
    ):
        assert required in retry_text
    for legacy in (
        "action: str",
        "failed_gates: tuple[str, ...]",
        "error_message: str | None",
    ):
        assert legacy not in retry_text

    report_path = REPO_ROOT / "src/invarlock/reporting/run_retry_validation.py"
    report_text = _read_text(report_path)
    for required in (
        "validation_gates: tuple[str, ...]",
        "diagnostic: RetryDiagnostic | None",
    ):
        assert required in report_text
    for legacy in (
        "failed_gates: tuple[str, ...]",
        "error_message: str | None",
    ):
        assert legacy not in report_text

    evaluate_path = REPO_ROOT / "src/invarlock/core/evaluate_contract.py"
    evaluate_text = _read_text(evaluate_path)
    for required in (
        "PrimaryMetricPolicyDiagnostic",
        "diagnostic: PrimaryMetricPolicyDiagnostic | None",
    ):
        assert required in evaluate_text
    for legacy in (
        "warning: str | None",
        '"unknown"',
        "INVARLOCK_STORE_EVAL_WINDOWS",
    ):
        assert legacy not in evaluate_text

    cli_path = REPO_ROOT / "src/invarlock/cli/commands/evaluate.py"
    cli_text = _read_text(cli_path)
    assert "outcome.diagnostic.message" in cli_text
    assert "outcome.warning" not in cli_text

    runner_guards_path = REPO_ROOT / "src/invarlock/core/runner_guards.py"
    runner_guards_text = _read_text(runner_guards_path)
    for legacy in (
        "_decision_from_action",
        'raw.get("warnings")',
        'raw.get("errors")',
        'raw.get("action")',
    ):
        assert legacy not in runner_guards_text

    report_overhead_path = REPO_ROOT / "src/invarlock/reporting/report_overhead.py"
    report_overhead_text = _read_text(report_overhead_path)
    assert (
        'diagnostics = _coerce_diagnostics(payload.get("diagnostics"))'
        in report_overhead_text
    )
    for legacy in (
        '"messages": list(result.messages)',
        '"warnings": list(result.warnings)',
        '"errors": list(result.errors)',
    ):
        assert legacy not in report_overhead_text


def test_guard_overhead_section_is_diagnostic_only() -> None:
    from invarlock.core.runner_guards import _normalize_guard_result
    from invarlock.reporting import report_overhead

    normalized = _normalize_guard_result(
        {
            "passed": False,
            "decision": "block",
            "warnings": ["warn"],
            "errors": ["err"],
        }
    )
    assert normalized["decision"] == "block"
    assert normalized["diagnostics"] == []
    assert "warnings" not in normalized
    assert "errors" not in normalized

    payload, _ = report_overhead.prepare_guard_overhead_section(
        {
            "bare_ppl": 10.0,
            "guarded_ppl": 10.5,
            "messages": ["ok"],
            "warnings": ["warn"],
            "errors": ["err"],
        }
    )
    assert "messages" not in payload
    assert "warnings" not in payload
    assert "errors" not in payload
    assert payload["diagnostics"] == []


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


def test_guard_validation_and_report_contracts_do_not_use_legacy_action_transcripts() -> (
    None
):
    offenders: list[str] = []
    for path in (
        REPO_ROOT / "src/invarlock/guards/invariants.py",
        REPO_ROOT / "src/invarlock/guards/spectral_runtime.py",
        REPO_ROOT / "src/invarlock/guards/variance_runtime.py",
        REPO_ROOT / "src/invarlock/guards/rmt_runtime.py",
        REPO_ROOT / "src/invarlock/core/run_report_payload_policy.py",
        REPO_ROOT / "src/invarlock/eval/bench_runner.py",
        REPO_ROOT / "src/invarlock/reporting/report_types.py",
    ):
        text = _read_text(path)
        for snippet in (
            '"action":',
            '"actions":',
            "def events(",
            "event_records",
        ):
            if snippet in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} -> {snippet}")

    assert not offenders, "\n".join(offenders)


def test_runner_guard_and_report_payload_contracts_use_typed_decisions() -> None:
    runner_guards_text = _read_text(REPO_ROOT / "src/invarlock/core/runner_guards.py")
    for legacy in (
        "_decision_from_action",
        'raw.get("warnings")',
        'raw.get("errors")',
        'raw.get("action")',
    ):
        assert legacy not in runner_guards_text

    report_payload_text = _read_text(
        REPO_ROOT / "src/invarlock/core/run_report_payload_policy.py"
    )
    assert "_decision_from_action" not in report_payload_text
    assert 'guard_result.get("action")' not in report_payload_text

    report_overhead_text = _read_text(
        REPO_ROOT / "src/invarlock/reporting/report_overhead.py"
    )
    assert (
        'diagnostics = _coerce_diagnostics(payload.get("diagnostics"))'
        in report_overhead_text
    )


def test_eval_metrics_contract_does_not_expose_progress_bar_surface() -> None:
    offenders: list[str] = []
    for path in (
        REPO_ROOT / "src/invarlock/eval/metrics_support.py",
        REPO_ROOT / "src/invarlock/eval/metrics_activation.py",
    ):
        text = _read_text(path)
        for snippet in ("progress_bars", "tqdm"):
            if snippet in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} -> {snippet}")

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
    orchestrator_path = REPO_ROOT / "src/invarlock/core/run_orchestrator.py"
    orchestrator_text = _read_text(orchestrator_path)
    offenders = []
    for snippet in (
        'class RunLifecycleEvent:\n    """Lifecycle event emitted by the owner layer."""\n\n    name: str',
        'class RunDiagnosticEvent:\n    """Diagnostic emitted by the owner layer."""\n\n    name: str',
        'class RunContextEvent:\n    """Context emitted by the owner layer."""\n\n    name: str',
        'class RunAggregateEvent:\n    """Aggregate/summary payload emitted by the owner layer."""\n\n    name: str',
        "payload: dict[str, Any] = field(default_factory=dict)",
        "def _emit_status(",
        "def _emit_metadata(",
    ):
        if snippet in orchestrator_text:
            offenders.append(snippet)

    assert not offenders, "\n".join(offenders)
    for required in (
        "class RunExecutionEvent",
        "class RunLifecycleEvent",
        "class RunDiagnosticEvent",
        "class RunContextEvent",
        "class RunAggregateEvent",
        "class RunFailureEvent",
        "class RunPrimaryMetricSummaryEvent",
    ):
        assert required in orchestrator_text

    cli_path = REPO_ROOT / "src/invarlock/cli/run_execution.py"
    cli_text = _read_text(cli_path)
    cli_offenders = []
    for snippet in ("event.name", "if name =="):
        if snippet in cli_text:
            cli_offenders.append(snippet)
    assert not cli_offenders, "\n".join(cli_offenders)


def test_edit_runtime_does_not_expose_shell_emit_controls() -> None:
    path = REPO_ROOT / "src/invarlock/core/api.py"
    text = _read_text(path)
    offenders = []
    for snippet in ("emit: bool", "Imperative shell runtime context"):
        if snippet in text:
            offenders.append(snippet)

    assert not offenders, "\n".join(offenders)


def test_runtime_provenance_does_not_embed_cli_flag_guidance() -> None:
    path = REPO_ROOT / "src/invarlock/runtime_provenance.py"
    text = _read_text(path)
    offenders = []
    for snippet in (
        "--allow-unattested-artifacts",
        "pass --allow-unattested-artifacts",
        "--execution-mode trusted-local",
        "pass --execution-mode trusted-local",
    ):
        if snippet in text:
            offenders.append(snippet)

    assert not offenders, "\n".join(offenders)


def test_runtime_security_policy_surface_is_typed_and_request_scoped() -> None:
    security_path = REPO_ROOT / "src/invarlock/runtime_security.py"
    provenance_path = REPO_ROOT / "src/invarlock/runtime_provenance.py"

    security_text = _read_text(security_path)
    attestation_text = _read_text(provenance_path)

    for snippet in (
        "class RuntimeSecurityPolicy",
        "def build_runtime_security_policy(",
        "policy: RuntimeSecurityPolicy | None = None",
        "ContextVar(",
        "def reset_runtime_allowances(",
    ):
        assert snippet in security_text

    security_offenders = []
    for snippet in ("_RUNTIME_ALLOWANCE_OVERRIDES", "def _set_env_flag("):
        if snippet in security_text:
            security_offenders.append(snippet)
    assert not security_offenders, "\n".join(security_offenders)

    for snippet in (
        "build_runtime_security_policy(",
        "apply_runtime_allowances(policy=policy)",
    ):
        assert snippet in attestation_text


def test_core_run_paths_do_not_read_shell_env_for_execution_policy() -> None:
    offenders: list[str] = []
    for path in (
        REPO_ROOT / "src/invarlock/core/run_orchestrator.py",
        REPO_ROOT / "src/invarlock/core/runner_eval_metrics.py",
    ):
        text = _read_text(path)
        for snippet in (
            'os.environ.get("INVARLOCK_EVAL_DEVICE")',
            'os.environ.get("PACK_DETERMINISM")',
            'os.environ.get("INVARLOCK_DETERMINISM")',
            'os.environ.get("INVARLOCK_DETERMINISM_WARN_ONLY")',
            'os.environ.get("INVARLOCK_TINY_RELAX")',
            'os.environ.get("INVARLOCK_EXPORT_MODEL")',
            'os.environ.get("INVARLOCK_EXPORT_DIR")',
        ):
            if snippet in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} -> {snippet}")

    assert not offenders, "\n".join(offenders)


def test_report_tiny_relax_is_provenance_only() -> None:
    offenders: list[str] = []
    for path in (
        REPO_ROOT / "src/invarlock/runtime_security.py",
        REPO_ROOT / "src/invarlock/reporting/report_make.py",
        REPO_ROOT / "src/invarlock/reporting/report_validation.py",
    ):
        text = _read_text(path)
        for snippet in (
            'os.environ.get("INVARLOCK_TINY_RELAX")',
            "INVARLOCK_TINY_RELAX",
        ):
            if snippet in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} -> {snippet}")

    assert not offenders, "\n".join(offenders)


def test_proof_pack_verify_results_use_typed_outcomes() -> None:
    path = REPO_ROOT / "src/invarlock/proof_pack.py"
    text = _read_text(path)
    offenders = []
    for snippet in (
        "status_code",
        'getattr(result, "outcome"',
        'getattr(result, "status_code"',
    ):
        if snippet in text:
            offenders.append(snippet)

    assert not offenders, "\n".join(offenders)
