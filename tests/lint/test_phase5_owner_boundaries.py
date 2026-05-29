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
            (1, None, "report_make_inputs"),
            (1, None, "report_make_assembly"),
            (1, None, "report_make_output"),
        ),
        REPO_ROOT / "src/invarlock/runtime_security_helpers.py": (
            (0, "invarlock", "runtime_security_container"),
            (0, "invarlock", "runtime_security_manifest"),
        ),
        REPO_ROOT / "src/invarlock/cli/commands/evaluate.py": (
            (2, "evaluate_output", "_evaluation_report_manifest_execution"),
            (2, "evaluate_phases", "_run_baseline_evaluation_phase"),
        ),
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute.py": (
            (
                0,
                "invarlock.core.run_orchestrator_execute_pipeline",
                "_execute_run_pipeline_steps",
            ),
        ),
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_pipeline.py": (
            (
                0,
                "invarlock.core.run_orchestrator_execute_dataset",
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
        REPO_ROOT / "src/invarlock/reporting/report_make_inputs.py",
        REPO_ROOT / "src/invarlock/reporting/report_make_assembly.py",
        REPO_ROOT / "src/invarlock/reporting/report_make_output.py",
        REPO_ROOT / "src/invarlock/reporting/verify_check_helpers_metrics.py",
        REPO_ROOT / "src/invarlock/reporting/verify_check_helpers_consistency.py",
        REPO_ROOT / "src/invarlock/runtime_security_container.py",
        REPO_ROOT / "src/invarlock/runtime_security_manifest.py",
        REPO_ROOT / "src/invarlock/cli/evaluate_output.py",
        REPO_ROOT / "src/invarlock/cli/evaluate_phases.py",
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_seed.py",
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_environment.py",
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_dataset.py",
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_attempts.py",
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_helpers.py",
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_execution.py",
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_pipeline.py",
        REPO_ROOT / "src/invarlock/core/builtin_plugin_catalog.py",
        REPO_ROOT / "src/invarlock/guards/policies_presets.py",
        REPO_ROOT / "src/invarlock/guards/policies_resolution.py",
        REPO_ROOT / "src/invarlock/guards/policies_validation.py",
        REPO_ROOT / "src/invarlock/evidence_pack_support.py",
    )

    missing = [
        str(path.relative_to(REPO_ROOT)) for path in expected_paths if not path.exists()
    ]
    assert not missing, "\n".join(missing)


def test_phase5_legacy_facades_are_removed() -> None:
    assert not (
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_prepare.py"
    ).exists()


def test_phase5_split_modules_stay_in_owner_layers() -> None:
    reporting_paths = (
        REPO_ROOT / "src/invarlock/reporting/report_make_inputs.py",
        REPO_ROOT / "src/invarlock/reporting/report_make_assembly.py",
        REPO_ROOT / "src/invarlock/reporting/report_make_output.py",
    )
    runtime_paths = (
        REPO_ROOT / "src/invarlock/runtime_security_container.py",
        REPO_ROOT / "src/invarlock/runtime_security_manifest.py",
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

    for path in runtime_paths:
        text = _read_text(path)
        assert "typer" not in text
        assert "console.print(" not in text

    for path in cli_paths:
        text = _read_text(path)
        assert "generate_reports(" not in text
        assert "make_report(" not in text


def test_phase5_orchestrator_owner_modules_do_not_cross_regrow() -> None:
    seed_path = REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_seed.py"
    environment_path = (
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_environment.py"
    )
    dataset_path = REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_dataset.py"
    attempts_path = (
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_attempts.py"
    )
    helpers_path = REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_helpers.py"
    execution_path = (
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_execution.py"
    )

    seed_text = _read_text(seed_path)
    environment_text = _read_text(environment_path)
    dataset_text = _read_text(dataset_path)
    attempts_text = _read_text(attempts_path)
    helpers_text = _read_text(helpers_path)
    execution_text = _read_text(execution_path)

    assert "run_orchestrator_execute_seed" not in seed_text
    assert "run_orchestrator_execute_environment" not in seed_text
    assert "run_orchestrator_execute_dataset" not in seed_text
    assert "run_orchestrator_execute_execution" not in seed_text

    assert "from .run_orchestrator_execute_seed import (" in environment_text
    assert "run_orchestrator_execute_dataset" not in environment_text
    assert "run_orchestrator_execute_execution" not in environment_text

    assert "run_orchestrator_execute_seed" not in dataset_text
    assert "run_orchestrator_execute_environment" not in dataset_text
    assert "run_orchestrator_execute_execution" not in dataset_text

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
        REPO_ROOT / "src/invarlock/evidence_pack.py": 650,
        REPO_ROOT / "src/invarlock/evidence_pack_support.py": 650,
        REPO_ROOT / "src/invarlock/runtime_security_helpers.py": 650,
        REPO_ROOT / "src/invarlock/reporting/verify_check_helpers.py": 650,
        REPO_ROOT / "src/invarlock/reporting/verify_check_helpers_metrics.py": 650,
        REPO_ROOT / "src/invarlock/reporting/verify_check_helpers_consistency.py": 650,
        REPO_ROOT / "src/invarlock/adapters/hf_mixin.py": 910,
        REPO_ROOT / "src/invarlock/cli/commands/plugins.py": 910,
        REPO_ROOT / "src/invarlock/guards/policies.py": 650,
        REPO_ROOT / "src/invarlock/guards/policies_impl.py": 650,
        REPO_ROOT / "src/invarlock/guards/policies_presets.py": 650,
        REPO_ROOT / "src/invarlock/guards/policies_resolution.py": 650,
        REPO_ROOT / "src/invarlock/guards/policies_validation.py": 650,
        REPO_ROOT / "src/invarlock/core/run_orchestrator_execute_attempts.py": 900,
        REPO_ROOT / "src/invarlock/eval/primary_metric.py": 850,
        REPO_ROOT / "src/invarlock/reporting/report_primary_metric_analysis.py": 840,
        REPO_ROOT / "src/invarlock/cli/commands/doctor.py": 730,
        REPO_ROOT / "src/invarlock/cli/commands/report.py": 613,
        REPO_ROOT / "src/invarlock/cli/commands/calibrate.py": 713,
        REPO_ROOT / "src/invarlock/reporting/report_validation.py": 533,
    }

    for path, threshold in thresholds.items():
        line_count = len(_read_text(path).splitlines())
        assert line_count <= threshold, (
            f"{path.relative_to(REPO_ROOT)} regrew to {line_count} lines "
            f"(threshold {threshold})"
        )
