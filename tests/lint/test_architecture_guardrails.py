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
    assert 'diagnostics = _coerce_diagnostics(payload.get("diagnostics"))' in report_overhead_text
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
    assert [item["severity"] for item in payload["diagnostics"]] == [
        "info",
        "warning",
        "error",
    ]


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


def test_guard_validation_and_report_contracts_do_not_use_legacy_action_transcripts() -> None:
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
    assert 'diagnostics = _coerce_diagnostics(payload.get("diagnostics"))' in report_overhead_text


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
        "class RunLifecycleEvent:\n    \"\"\"Lifecycle event emitted by the owner layer.\"\"\"\n\n    name: str",
        "class RunDiagnosticEvent:\n    \"\"\"Diagnostic emitted by the owner layer.\"\"\"\n\n    name: str",
        "class RunContextEvent:\n    \"\"\"Context emitted by the owner layer.\"\"\"\n\n    name: str",
        "class RunAggregateEvent:\n    \"\"\"Aggregate/summary payload emitted by the owner layer.\"\"\"\n\n    name: str",
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


def test_runtime_security_policy_surface_is_typed_and_request_scoped() -> None:
    security_path = REPO_ROOT / "src/invarlock/runtime_security.py"
    attestation_path = REPO_ROOT / "src/invarlock/runtime_attestation.py"

    security_text = _read_text(security_path)
    attestation_text = _read_text(attestation_path)

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
    for snippet in ("status_code", 'getattr(result, "outcome"', 'getattr(result, "status_code"'):
        if snippet in text:
            offenders.append(snippet)

    assert not offenders, "\n".join(offenders)


def test_hf_adapter_local_only_retry_uses_cache_miss_detection() -> None:
    path = REPO_ROOT / "src/invarlock/adapters/hf_mixin.py"
    text = _read_text(path)
    tree = ast.parse(text, filename=str(path))

    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_load_pretrained_model":
            target = node
            break

    assert target is not None, "_load_pretrained_model not found"
    source = ast.get_source_segment(text, target) or ""
    assert "_is_local_loader_cache_miss" in source
    assert "prefer_local_files_only" in source


def test_guarded_benchmark_failures_raise_instead_of_continuing() -> None:
    path = REPO_ROOT / "src/invarlock/eval/bench_runner.py"
    text = _read_text(path)
    tree = ast.parse(text, filename=str(path))

    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "execute_single_run":
            target = node
            break

    assert target is not None, "execute_single_run not found"
    source = ast.get_source_segment(text, target) or ""
    assert "Guard construction failed" in source
    assert "RMT detection failed for" in source
    assert "Core report returned invalid edit metadata payload" in source
    assert "Core report returned non-string plan_digest" in source
    assert "Core report returned invalid edit delta payload" in source


def test_execute_scenario_surfaces_benchmark_assembly_failures() -> None:
    path = REPO_ROOT / "src/invarlock/eval/bench_runner.py"
    text = _read_text(path)
    tree = ast.parse(text, filename=str(path))

    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "execute_scenario":
            target = node
            break

    assert target is not None, "execute_scenario not found"
    source = ast.get_source_segment(text, target) or ""
    assert "_assign_dataset_provider(" in source
    assert "_extract_success_report_path(" in source
    assert "Evaluation report generation failed for" in source


def test_spectral_validation_unexpected_failures_raise() -> None:
    path = REPO_ROOT / "src/invarlock/guards/spectral_runtime.py"
    text = _read_text(path)
    tree = ast.parse(text, filename=str(path))

    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "validate_guard":
            target = node
            break

    assert target is not None, "validate_guard not found"
    source = ast.get_source_segment(text, target) or ""
    assert "except Exception as error" not in source


def test_spectral_prepare_and_after_edit_do_not_swallow_runtime_failures() -> None:
    path = REPO_ROOT / "src/invarlock/guards/spectral_runtime.py"
    text = _read_text(path)
    tree = ast.parse(text, filename=str(path))

    targets: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in {
            "prepare_guard",
            "after_edit_guard",
        }:
            targets[node.name] = node

    assert set(targets) == {"prepare_guard", "after_edit_guard"}

    prepare_source = ast.get_source_segment(text, targets["prepare_guard"]) or ""
    assert "except Exception" not in prepare_source
    assert 'raise RuntimeError("Failed to prepare spectral guard.")' in prepare_source

    after_edit_source = ast.get_source_segment(text, targets["after_edit_guard"]) or ""
    assert "except Exception" not in after_edit_source
    assert (
        'raise RuntimeError("Post-edit spectral analysis failed.")'
        in after_edit_source
    )


def test_latency_measurement_failures_raise_runtime_errors() -> None:
    path = REPO_ROOT / "src/invarlock/eval/metrics_runtime.py"
    text = _read_text(path)
    tree = ast.parse(text, filename=str(path))

    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "measure_latency":
            target = node
            break

    assert target is not None, "measure_latency not found"
    source = ast.get_source_segment(text, target) or ""
    assert "_latency_validation_error(" in source
    assert "non-empty evaluation window" in source
    assert "sequence longer than 10 tokens" in source
    assert "attended token" in source
    assert "Latency warmup failed." in source
    assert "Latency measurement failed." in source
    assert "return 0.0" not in source
    assert "except Exception:\n                return 0.0" not in source


def test_metrics_runtime_does_not_hide_device_vocab_or_memory_failures() -> None:
    path = REPO_ROOT / "src/invarlock/eval/metrics_runtime.py"
    text = _read_text(path)
    tree = ast.parse(text, filename=str(path))

    targets: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_resolve_eval_device",
            "_infer_model_vocab_size",
            "measure_memory",
        }:
            targets[node.name] = node

    assert set(targets) == {
        "_resolve_eval_device",
        "_infer_model_vocab_size",
        "measure_memory",
    }

    resolve_source = ast.get_source_segment(text, targets["_resolve_eval_device"]) or ""
    assert "except Exception" not in resolve_source

    vocab_source = ast.get_source_segment(text, targets["_infer_model_vocab_size"]) or ""
    assert "except Exception" not in vocab_source
    assert "return None" in vocab_source

    memory_source = ast.get_source_segment(text, targets["measure_memory"]) or ""
    assert "logger.debug(f\"Memory measurement failed" not in memory_source
    assert 'raise RuntimeError(\n                    f"Memory measurement failed for sample {i}."' in memory_source or 'raise RuntimeError(f"Memory measurement failed for sample {i}.")' in memory_source
    assert "_memory_validation_error(" in memory_source


def test_metrics_validator_model_errors_raise_instead_of_debug_fallback() -> None:
    path = REPO_ROOT / "src/invarlock/eval/metrics_support.py"
    text = _read_text(path)
    tree = ast.parse(text, filename=str(path))

    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "validate_model":
            target = node
            break

    assert target is not None, "validate_model not found"
    source = ast.get_source_segment(text, target) or ""
    assert "Could not count model parameters" not in source
    assert "Model parameter iteration failed" in source
    assert "except Exception as exc" in source


def test_adapter_probe_helpers_do_not_hide_unexpected_failures() -> None:
    auto_path = REPO_ROOT / "src/invarlock/adapters/auto.py"
    auto_text = _read_text(auto_path)
    auto_tree = ast.parse(auto_text, filename=str(auto_path))

    detect_target = None
    for node in ast.walk(auto_tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_detect_quantization_from_path":
            detect_target = node
            break

    assert detect_target is not None, "_detect_quantization_from_path not found"
    detect_source = ast.get_source_segment(auto_text, detect_target) or ""
    assert "except Exception" not in detect_source
    assert "except (OSError, TypeError, ValueError)" in detect_source

    hf_path = REPO_ROOT / "src/invarlock/adapters/hf_causal.py"
    hf_text = _read_text(hf_path)
    hf_tree = ast.parse(hf_text, filename=str(hf_path))

    select_target = None
    can_handle_target = None
    for node in ast.walk(hf_tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_select_spec":
            select_target = node
        if isinstance(node, ast.FunctionDef) and node.name == "can_handle":
            can_handle_target = node

    assert select_target is not None, "_select_spec not found"
    assert can_handle_target is not None, "can_handle not found"
    select_source = ast.get_source_segment(hf_text, select_target) or ""
    can_handle_source = ast.get_source_segment(hf_text, can_handle_target) or ""
    assert "except Exception" not in select_source
    assert "except Exception" not in can_handle_source
    assert "no matching HF causal adapter spec" in select_source


def test_subprocess_verifiers_use_timeouts() -> None:
    offenders: list[str] = []
    expectations = {
        REPO_ROOT / "src/invarlock/runtime_security.py": "timeout=",
        REPO_ROOT / "src/invarlock/runtime_attestation.py": "timeout=",
        REPO_ROOT / "src/invarlock/proof_pack.py": "timeout=",
    }
    for path, required in expectations.items():
        text = _read_text(path)
        if "subprocess.run(" in text and required not in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, "\n".join(offenders)


def test_tokenizer_provenance_helpers_do_not_normalize_unknown_placeholders() -> None:
    expectations = {
        REPO_ROOT / "src/invarlock/model_profile.py": ('return "unknown"',),
        REPO_ROOT / "src/invarlock/cli/run_masking.py": ('"unknown-tokenizer"',),
    }
    offenders: list[str] = []
    for path, snippets in expectations.items():
        text = _read_text(path)
        for snippet in snippets:
            if snippet in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} -> {snippet}")

    assert not offenders, "\n".join(offenders)


def test_dataset_and_report_provenance_paths_preserve_nullable_fields() -> None:
    expectations = {
        REPO_ROOT / "src/invarlock/core/run_provider_dataset_plan.py": (
            'getattr(tokenizer, "name_or_path", "unknown")',
        ),
        REPO_ROOT / "src/invarlock/reporting/report_make.py": (
            'meta_section.get("model_id", "unknown")',
            'meta_section.get("adapter", "unknown")',
            'meta_section.get("device", "unknown")',
            'meta.get("model_id", "unknown")',
            'get("name", "unknown")',
        ),
    }
    offenders: list[str] = []
    for path, snippets in expectations.items():
        text = _read_text(path)
        for snippet in snippets:
            if snippet in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} -> {snippet}")

    assert not offenders, "\n".join(offenders)


def test_reporting_and_container_helpers_do_not_read_ambient_behavior_env() -> None:
    expectations = {
        REPO_ROOT / "src/invarlock/reporting/report_primary_metric_analysis.py": (
            "INVARLOCK_BOOTSTRAP_BCA",
        ),
        REPO_ROOT / "src/invarlock/runtime_security.py": (
            "_BEHAVIOR_ENV_VARS",
            "for name in sorted(_BEHAVIOR_ENV_VARS)",
        ),
        REPO_ROOT / "src/invarlock/reporting/report_make.py": (
            "validation_allowlist_fallback",
        ),
    }
    offenders: list[str] = []
    for path, snippets in expectations.items():
        text = _read_text(path)
        for snippet in snippets:
            if snippet in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} -> {snippet}")

    assert not offenders, "\n".join(offenders)


def test_cli_runtime_helpers_do_not_hide_snapshot_reuse_failures() -> None:
    expectations = {
        REPO_ROOT / "src/invarlock/cli/run_runtime_exec.py": (
            "bare_stub_model",
            "guarded_stub_model",
        ),
        REPO_ROOT / "src/invarlock/cli/run_pairing.py": (
            "except Exception as exc",
        ),
    }
    offenders: list[str] = []
    for path, snippets in expectations.items():
        text = _read_text(path)
        for snippet in snippets:
            if snippet in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} -> {snippet}")

    assert not offenders, "\n".join(offenders)


def test_core_summary_helpers_do_not_embed_display_strings() -> None:
    expectations = {
        REPO_ROOT / "src/invarlock/core/run_guard_overhead_policy.py": (
            'status = "PASS"',
            "threshold_display",
            "overhead_display",
        ),
        REPO_ROOT / "src/invarlock/core/run_timing_policy.py": (
            "Peak Memory",
            "Peak GPU Mem",
            '(\"Load model\", \"load_model\")',
        ),
        REPO_ROOT / "src/invarlock/core/doctor_inventory.py": (
            "pip install",
            "✓ Available",
            "Cache/Net",
        ),
    }
    offenders: list[str] = []
    for path, snippets in expectations.items():
        text = _read_text(path)
        for snippet in snippets:
            if snippet in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} -> {snippet}")
    assert not offenders, "\n".join(offenders)


def test_run_runtime_exec_helpers_do_not_emit_shell_output() -> None:
    path = REPO_ROOT / "src/invarlock/cli/run_runtime_exec.py"
    text = _read_text(path)
    offenders = []
    for snippet in (
        "from invarlock.cli.run_shell_output import _event",
        "_event(",
        "console:",
    ):
        if snippet in text:
            offenders.append(snippet)
    assert not offenders, "\n".join(offenders)


def test_run_execution_consumes_core_timing_summary() -> None:
    path = REPO_ROOT / "src/invarlock/cli/run_execution.py"
    text = _read_text(path)
    for required in (
        "timing_summary = outcome.result.timing_summary",
        "timing_summary.ordered_keys",
        "timing_summary.memory_mb_peak",
        "timing_summary.gpu_memory_mb_peak",
    ):
        assert required in text


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


def test_core_config_helpers_do_not_swallow_unexpected_runtime_errors() -> None:
    expectations = {
        REPO_ROOT / "src/invarlock/core/run_dataset_contract.py": ("except Exception",),
        REPO_ROOT / "src/invarlock/core/run_execution_context_policy.py": (
            "except Exception",
        ),
        REPO_ROOT / "src/invarlock/core/run_provider_dataset_plan.py": (
            "except Exception",
        ),
    }
    offenders: list[str] = []
    for path, snippets in expectations.items():
        text = _read_text(path)
        for snippet in snippets:
            if snippet in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} -> {snippet}")
    assert not offenders, "\n".join(offenders)


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
