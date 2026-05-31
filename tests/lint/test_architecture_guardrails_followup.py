from __future__ import annotations

import ast

from tests.lint.test_architecture_guardrails_import_boundaries import (
    METRICS_PATH,
    REPO_ROOT,
    _read_text,
)

BROAD_EXCEPTION = "except " + "Exception"
BROAD_EXCEPTION_AS_ERROR = BROAD_EXCEPTION + " as error"
BROAD_EXCEPTION_AS_EXC = BROAD_EXCEPTION + " as exc"
TYPE_IGNORE = "type: " + "ignore"
TYPE_IGNORE_IMPORT_UNTYPED = TYPE_IGNORE + "[import-untyped]"


def test_adapter_probe_helpers_do_not_hide_unexpected_failures() -> None:
    auto_path = REPO_ROOT / "src/invarlock/adapters/auto.py"
    auto_text = _read_text(auto_path)
    auto_tree = ast.parse(auto_text, filename=str(auto_path))

    read_config_target = None
    detect_target = None
    for node in ast.walk(auto_tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_read_local_hf_config":
            read_config_target = node
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_detect_quantization_from_path"
        ):
            detect_target = node
            break

    assert read_config_target is not None, "_read_local_hf_config not found"
    assert detect_target is not None, "_detect_quantization_from_path not found"
    read_config_source = ast.get_source_segment(auto_text, read_config_target) or ""
    detect_source = ast.get_source_segment(auto_text, detect_target) or ""
    assert BROAD_EXCEPTION not in read_config_source
    assert BROAD_EXCEPTION not in detect_source
    assert "except (OSError, TypeError, ValueError)" in read_config_source
    assert "_read_local_hf_config(model_id)" in detect_source

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
    assert BROAD_EXCEPTION not in select_source
    assert BROAD_EXCEPTION not in can_handle_source
    assert "no matching HF causal adapter spec" in select_source


def test_subprocess_verifiers_use_timeouts() -> None:
    offenders: list[str] = []
    expectations = {
        REPO_ROOT / "src/invarlock/runtime_security.py": "timeout=",
        REPO_ROOT / "src/invarlock/runtime_provenance.py": "timeout=",
        REPO_ROOT / "src/invarlock/evidence_pack.py": "timeout=",
    }
    for path, required in expectations.items():
        text = _read_text(path)
        if "subprocess.run(" in text and required not in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, "\n".join(offenders)


def test_tokenizer_provenance_helpers_do_not_normalize_unknown_placeholders() -> None:
    expectations = {
        REPO_ROOT / "src/invarlock/model_profile.py": ('return "unknown"',),
        REPO_ROOT / "src/invarlock/cli/run_execution.py": ('"unknown-tokenizer"',),
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
        REPO_ROOT / "src/invarlock/cli/run_pairing.py": (BROAD_EXCEPTION_AS_EXC,),
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
        REPO_ROOT / "src/invarlock/reporting/report_overhead.py": (
            'status = "PASS"',
            "threshold_display",
            "overhead_display",
        ),
        REPO_ROOT / "src/invarlock/core/run_policy.py": (
            "Peak Memory",
            "Peak GPU Mem",
            '("Load model", "load_model")',
        ),
        REPO_ROOT / "src/invarlock/core/doctor_findings.py": (
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


def test_runtime_provenance_has_single_product_owner() -> None:
    assert not (REPO_ROOT / "src/invarlock/core/runtime_provenance.py").exists()

    text = _read_text(REPO_ROOT / "src/invarlock/reporting/verify_contract.py")
    assert "from invarlock.runtime_provenance import (" in text
    assert "from invarlock.core.runtime_provenance" not in text


def test_core_config_helpers_do_not_swallow_unexpected_runtime_errors() -> None:
    expectations = {
        REPO_ROOT / "src/invarlock/core/run_policy.py": (BROAD_EXCEPTION,),
        REPO_ROOT / "src/invarlock/core/run_provider_dataset_plan.py": (
            BROAD_EXCEPTION,
        ),
    }
    offenders: list[str] = []
    for path, snippets in expectations.items():
        text = _read_text(path)
        for snippet in snippets:
            if snippet in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} -> {snippet}")
    assert not offenders, "\n".join(offenders)


def test_hardened_runtime_paths_keep_broad_catch_budgets() -> None:
    budgets = {
        REPO_ROOT / "src/invarlock/core/events.py": 0,
        REPO_ROOT / "src/invarlock/core/plugins_inventory.py": 0,
        REPO_ROOT / "src/invarlock/core/registry.py": 2,
        REPO_ROOT / "src/invarlock/core/runner_eval_metrics.py": 0,
        REPO_ROOT / "src/invarlock/adapters/hf_causal.py": 0,
        REPO_ROOT / "src/invarlock/adapters/hf_mixin.py": 0,
        REPO_ROOT / "src/invarlock/eval/data_tokenization.py": 0,
        REPO_ROOT / "src/invarlock/eval/bench_runner.py": 0,
        REPO_ROOT / "src/invarlock/eval/window_planning.py": 0,
        REPO_ROOT / "src/invarlock/eval/primary_metric.py": 0,
        REPO_ROOT / "src/invarlock/model_profile.py": 0,
        REPO_ROOT / "src/invarlock/observability/alerting.py": 0,
        REPO_ROOT / "src/invarlock/observability/core.py": 0,
        REPO_ROOT / "src/invarlock/observability/exporters.py": 0,
        REPO_ROOT / "src/invarlock/observability/health.py": 0,
        REPO_ROOT / "src/invarlock/adapters/hf_seq2seq.py": 0,
        REPO_ROOT / "src/invarlock/utils/__init__.py": 0,
        REPO_ROOT / "src/invarlock/core/runner_eval_phase.py": 0,
        REPO_ROOT / "src/invarlock/cli/app.py": 0,
        REPO_ROOT / "src/invarlock/cli/commands/report.py": 0,
        REPO_ROOT / "src/invarlock/cli/run_overhead.py": 0,
    }
    offenders: list[str] = []
    for path, budget in budgets.items():
        count = _read_text(path).count(BROAD_EXCEPTION)
        if count > budget:
            offenders.append(f"{path.relative_to(REPO_ROOT)} -> {count} > {budget}")
    assert not offenders, "\n".join(offenders)


def test_source_tree_has_no_import_untyped_suppressions() -> None:
    offenders: list[str] = []
    for path in (REPO_ROOT / "src").rglob("*.py"):
        text = _read_text(path)
        if TYPE_IGNORE_IMPORT_UNTYPED in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, "\n".join(offenders)


def test_hardened_followup_paths_have_no_local_type_ignore_escapes() -> None:
    hardened_paths = (
        REPO_ROOT / "src/invarlock/adapters/hf_causal.py",
        REPO_ROOT / "src/invarlock/adapters/hf_mixin.py",
        REPO_ROOT / "src/invarlock/eval/bench_policy.py",
        REPO_ROOT / "src/invarlock/eval/data.py",
        REPO_ROOT / "src/invarlock/eval/primary_metric.py",
        REPO_ROOT / "src/invarlock/observability/alerting.py",
        REPO_ROOT / "src/invarlock/observability/health.py",
        REPO_ROOT / "src/invarlock/adapters/hf_seq2seq.py",
        REPO_ROOT / "src/invarlock/utils/__init__.py",
    )
    offenders: list[str] = []
    for path in hardened_paths:
        text = _read_text(path)
        if TYPE_IGNORE in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
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
    tree = ast.parse(_read_text(METRICS_PATH), filename=str(METRICS_PATH))
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
