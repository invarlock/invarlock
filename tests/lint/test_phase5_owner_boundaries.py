from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_module(path: Path) -> ast.Module:
    return ast.parse(_read_text(path), filename=str(path))


def _import_from_aliases(path: Path) -> set[tuple[int, str | None, str]]:
    aliases: set[tuple[int, str | None, str]] = set()
    for node in ast.walk(_parse_module(path)):
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            aliases.add((node.level, node.module, alias.name))
    return aliases


def test_phase5_owner_modules_keep_split_imports() -> None:
    expectations = {
        REPO_ROOT / "src/invarlock/reporting/report_make.py": (
            (1, None, "report_make_assembly"),
        ),
        REPO_ROOT / "src/invarlock/reporting/report_builder_support.py": (
            (1, None, "report_builder_telemetry"),
        ),
        REPO_ROOT / "src/invarlock/reporting/render_markdown.py": (
            (1, "render_markdown_tables", "append_accuracy_subgroups"),
            (1, "render_markdown_tables", "append_system_overhead_section"),
        ),
        REPO_ROOT / "src/invarlock/guards/invariants.py": (
            (0, "invarlock.guards", "invariants_standard"),
        ),
        REPO_ROOT / "src/invarlock/eval/data.py": (
            (1, "data_local", "LocalJSONLProvider"),
            (1, "data_local", "LocalJSONLPairsProvider"),
        ),
        REPO_ROOT / "src/invarlock/eval/data_providers.py": (
            (1, "data_hf_providers", "HFSeq2SeqProvider"),
            (1, "data_hf_providers", "HFTextProvider"),
        ),
        REPO_ROOT / "src/invarlock/eval/metrics_runtime.py": (
            (
                0,
                "invarlock.eval.metrics_runtime_resources",
                "latency_validation_error",
            ),
        ),
        REPO_ROOT / "src/invarlock/cli/run_runtime_exec.py": (
            (
                0,
                "invarlock.cli.run_runtime_warnings",
                "suppress_noisy_warnings",
            ),
        ),
        REPO_ROOT / "src/invarlock/cli/commands/plugins.py": (
            (1, "plugins_rendering", "handle_plugins_category"),
        ),
        REPO_ROOT / "src/invarlock/cli/commands/evaluate.py": (
            (2, "evaluate_output", "_evaluation_report_manifest_execution"),
            (2, "evaluate_phases", "BaselineEvaluationRequest"),
            (2, "evaluate_phases", "run_baseline_evaluation_phase"),
        ),
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute.py": (
            (
                0,
                "invarlock.core.run_orchestrator_execute_execution",
                "_load_dataset_state",
            ),
            (
                0,
                "invarlock.core.run_orchestrator_execute_environment",
                "_prepare_run_environment",
            ),
            (
                0,
                "invarlock.core.run_orchestrator_execute_execution",
                "_prepare_execution_state",
            ),
        ),
        REPO_ROOT / "src/invarlock/core/registry.py": (
            (
                1,
                "builtin_plugin_catalog",
                "builtin_plugin_specs",
            ),
        ),
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_attempts.py": (
            (
                0,
                "invarlock.core.run_orchestrator_execute_helpers",
                "RunEventEmitter",
            ),
            (
                0,
                "invarlock.core.run_orchestrator_execute_helpers",
                "_AttemptExecutionState",
            ),
            (
                0,
                "invarlock.core",
                "run_orchestrator_execute_attempt_results",
            ),
        ),
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_execution.py": (
            (
                0,
                "invarlock.core.run_orchestrator_execute_helpers",
                "_RunComponentState",
            ),
            (
                0,
                "invarlock.core.run_orchestrator_execute_helpers",
                "_RunExecutionState",
            ),
        ),
    }

    for path, expected_aliases in expectations.items():
        aliases = _import_from_aliases(path)
        for expected_alias in expected_aliases:
            assert expected_alias in aliases, (
                f"{path.relative_to(REPO_ROOT)} missing import {expected_alias}"
            )


def test_phase5_split_modules_exist() -> None:
    expected_paths = (
        REPO_ROOT / "src/invarlock/reporting/report_make.py",
        REPO_ROOT / "src/invarlock/reporting/report_make_assembly.py",
        REPO_ROOT / "src/invarlock/reporting/report_builder_telemetry.py",
        REPO_ROOT / "src/invarlock/reporting/render_markdown_tables.py",
        REPO_ROOT / "src/invarlock/guards/invariants_standard.py",
        REPO_ROOT / "src/invarlock/eval/data_local.py",
        REPO_ROOT / "src/invarlock/eval/data_hf_providers.py",
        REPO_ROOT / "src/invarlock/eval/metrics_runtime_resources.py",
        REPO_ROOT / "src/invarlock/cli/run_runtime_warnings.py",
        REPO_ROOT / "src/invarlock/reporting/verify_check_helpers_metrics.py",
        REPO_ROOT / "src/invarlock/reporting/verify_check_helpers_consistency.py",
        REPO_ROOT / "src/invarlock/cli/evaluate_output.py",
        REPO_ROOT / "src/invarlock/cli/evaluate_phases.py",
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_environment.py",
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_attempts.py",
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_attempt_results.py",
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_helpers.py",
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_execution.py",
        REPO_ROOT / "src/invarlock/core/builtin_plugin_catalog.py",
        REPO_ROOT / "src/invarlock/evidence_pack_support.py",
        REPO_ROOT / "src/invarlock/cli/commands/plugins_rendering.py",
    )

    missing = [
        str(path.relative_to(REPO_ROOT)) for path in expected_paths if not path.exists()
    ]
    assert not missing, "\n".join(missing)


def test_phase5_legacy_facades_are_removed() -> None:
    assert not (
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_prepare.py"
    ).exists()
    assert not (
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_pipeline.py"
    ).exists()
    assert not (REPO_ROOT / "src/invarlock/core/run_orchestrator_types.py").exists()


def test_phase5_split_modules_stay_in_owner_layers() -> None:
    reporting_paths = (
        REPO_ROOT / "src/invarlock/reporting/report_make.py",
        REPO_ROOT / "src/invarlock/reporting/report_make_assembly.py",
    )
    cli_paths = (
        REPO_ROOT / "src/invarlock/cli/evaluate_output.py",
        REPO_ROOT / "src/invarlock/cli/evaluate_phases.py",
    )

    for path in reporting_paths:
        text = _read_text(path)
        assert "invarlock.cli" not in text
        assert "console.print(" not in text
        assert "typer.echo(" not in text
        assert "_report_make_module(" not in text
        assert "apply_validation_allowlist_schema(" not in text

    for path in cli_paths:
        text = _read_text(path)
        assert "generate_reports(" not in text
        assert "make_report(" not in text


def test_phase5_orchestrator_owner_modules_do_not_cross_regrow() -> None:
    environment_path = (
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_environment.py"
    )
    attempts_path = (
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_attempts.py"
    )
    helpers_path = REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_helpers.py"
    execution_path = (
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_execution.py"
    )

    environment_text = _read_text(environment_path)
    attempts_text = _read_text(attempts_path)
    helpers_text = _read_text(helpers_path)
    execution_text = _read_text(execution_path)

    assert "run_orchestrator_execute_dataset" not in environment_text
    assert "run_orchestrator_execute_execution" not in environment_text

    assert "run_orchestrator_execute_seed" not in attempts_text
    assert "run_orchestrator_execute_environment" not in attempts_text
    assert "run_orchestrator_execute_dataset" not in attempts_text

    assert "run_orchestrator_execute_seed" not in helpers_text
    assert "run_orchestrator_execute_environment" not in helpers_text
    assert "run_orchestrator_execute_dataset" not in helpers_text
    assert "run_orchestrator_execute_execution" not in helpers_text

    assert "run_orchestrator_execute_seed" not in execution_text
    assert "run_orchestrator_execute_environment" not in execution_text
    assert "run_orchestrator_execute_dataset" not in execution_text


def test_phase5_large_modules_do_not_regrow() -> None:
    thresholds = {
        REPO_ROOT / "src/invarlock/model_profile.py": 650,
        REPO_ROOT / "src/invarlock/model_profile_tokenizers.py": 500,
        REPO_ROOT / "src/invarlock/evidence_pack.py": 1000,
        REPO_ROOT / "src/invarlock/evidence_pack_support.py": 650,
        REPO_ROOT / "src/invarlock/runtime_security_helpers.py": 1000,
        REPO_ROOT / "src/invarlock/reporting/verify_check_helpers_consistency.py": 650,
        REPO_ROOT / "src/invarlock/reporting/verify_check_helpers_metrics.py": 650,
        REPO_ROOT / "src/invarlock/reporting/verify_check_helpers_consistency.py": 650,
        REPO_ROOT / "src/invarlock/adapters/hf_mixin.py": 650,
        REPO_ROOT / "src/invarlock/adapters/hf_mixin_snapshot.py": 420,
        REPO_ROOT / "src/invarlock/adapters/hf_mlm.py": 650,
        REPO_ROOT / "src/invarlock/adapters/hf_mlm_structure.py": 420,
        REPO_ROOT / "src/invarlock/cli/commands/plugins.py": 420,
        REPO_ROOT / "src/invarlock/cli/commands/plugins_rendering.py": 760,
        REPO_ROOT / "src/invarlock/guards/policies.py": 800,
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_attempts.py": 480,
        REPO_ROOT
        / "src/invarlock/core/run_orchestrator_execute_attempt_results.py": 540,
        REPO_ROOT / "src/invarlock/reporting/report_builder_support.py": 790,
        REPO_ROOT / "src/invarlock/reporting/report_builder_telemetry.py": 150,
        REPO_ROOT / "src/invarlock/reporting/render_markdown.py": 790,
        REPO_ROOT / "src/invarlock/reporting/render_markdown_tables.py": 140,
        REPO_ROOT / "src/invarlock/guards/invariants.py": 790,
        REPO_ROOT / "src/invarlock/guards/invariants_standard.py": 120,
        REPO_ROOT / "src/invarlock/eval/data.py": 650,
        REPO_ROOT / "src/invarlock/eval/data_local.py": 280,
        REPO_ROOT / "src/invarlock/eval/data_providers.py": 540,
        REPO_ROOT / "src/invarlock/eval/data_hf_providers.py": 380,
        REPO_ROOT / "src/invarlock/eval/metrics_runtime.py": 790,
        REPO_ROOT / "src/invarlock/eval/metrics_runtime_resources.py": 100,
        REPO_ROOT / "src/invarlock/cli/run_runtime_exec.py": 650,
        REPO_ROOT / "src/invarlock/cli/run_runtime_warnings.py": 300,
        REPO_ROOT / "src/invarlock/eval/primary_metric.py": 850,
        REPO_ROOT / "src/invarlock/reporting/report_primary_metric_analysis.py": 840,
        REPO_ROOT / "src/invarlock/cli/commands/doctor.py": 730,
        REPO_ROOT / "src/invarlock/cli/commands/report.py": 800,
        REPO_ROOT / "src/invarlock/cli/commands/calibrate.py": 713,
        REPO_ROOT / "src/invarlock/reporting/report_validation.py": 533,
    }

    for path, threshold in thresholds.items():
        line_count = len(_read_text(path).splitlines())
        assert line_count <= threshold, (
            f"{path.relative_to(REPO_ROOT)} regrew to {line_count} lines "
            f"(threshold {threshold})"
        )
