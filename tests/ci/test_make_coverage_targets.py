from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_coverage_policy():
    policy_path = Path(__file__).resolve().parents[2] / "scripts" / "coverage_policy.py"
    spec = importlib.util.spec_from_file_location("coverage_policy", policy_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _coverage_tests_eval_block() -> str:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")
    marker = "COVERAGE_TESTS_EVAL := \\"
    start = text.index(marker)
    rest = text[start:].splitlines()

    lines: list[str] = []
    for line in rest:
        if line.startswith("COVERAGE_TESTS_CLI_COMMANDS :="):
            break
        lines.append(line)
    return "\n".join(lines)


def test_coverage_target_includes_active_eval_data_and_helper_tests() -> None:
    block = _coverage_tests_eval_block()

    expected_patterns = (
        "tests/eval/test_task_metrics.py",
        "tests/eval/test_eval_bootstrap_wrapper.py",
        "tests/eval/test_metric_gate_summary_inputs.py",
        "tests/eval/test_data*.py",
        "tests/eval/test_hf_text_provider*.py",
        "tests/eval/test_local_jsonl*.py",
        "tests/eval/test_synthetic_provider_paths.py",
        "tests/eval/test_wikitext2_fast_capacity.py",
        "tests/eval/test_provider_deterministic_loader_paths.py",
        "tests/eval/test_difficulty_scorer_modes.py",
        "tests/eval/providers",
    )

    for pattern in expected_patterns:
        assert pattern in block


def test_coverage_target_includes_probe_suite_for_plain_coverage_run() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    assert "COVERAGE_TESTS_EVAL_PROBES :=" in text
    for pattern in (
        "tests/eval/test_fft.py",
        "tests/eval/test_fft_probe_paths.py",
        "tests/eval/test_mi*.py",
        "tests/eval/test_post_attention_probes.py",
        "tests/eval/test_post_attention_probe_paths.py",
    ):
        assert pattern in text
    assert "$(COVERAGE) run --append -m pytest -q -p no:cov" in text


def test_coverage_target_includes_core_cli_surface_and_runtime_security_tests() -> None:
    policy = _load_coverage_policy()
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    for pattern in (
        "tests/cli/test_core_command_surface.py",
        "tests/cli/test_execution_mode.py",
        "tests/cli/test_removed_command_migrations.py",
        "tests/cli/test_python_m_invarlock.py",
        "tests/cli/test_container_default_contract.py",
        "tests/cli/test_container_delegation.py",
    ):
        assert pattern in text
    assert "invarlock/runtime_security.py" in policy.coverage_include()


def test_coverage_target_includes_adapter_auto_runtime_suite() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    assert "tests/adapters/test_adapter_auto_runtime.py" in text


def test_coverage_policy_is_shared_with_makefile_and_expanded_surface() -> None:
    policy = _load_coverage_policy()
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    assert "COVERAGE_POLICY := $(PYTHON) scripts/coverage_policy.py" in text
    assert "COVERAGE_MODULES := \\" in text
    assert "\t$(shell $(COVERAGE_POLICY) coverage-modules)" in text
    assert "COVERAGE_INCLUDE := $(shell $(COVERAGE_POLICY) coverage-include)" in text

    assert "src/invarlock/observability/" in policy.CORE_PREFIXES
    assert "src/invarlock/config.py" in policy.CORE_FILES
    assert "src/invarlock/adapters/auto.py" in policy.CORE_FILES

    assert policy.COVERAGE_MODULE_FLAGS == ("--cov",)

    assert not any(
        flag.startswith("--cov=src/invarlock/") for flag in policy.COVERAGE_MODULE_FLAGS
    )

    for pattern in (
        "src/invarlock/observability/*",
        "src/invarlock/config.py",
        "src/invarlock/adapters/auto.py",
        "invarlock/observability/*",
        "invarlock/config.py",
        "invarlock/adapters/auto.py",
    ):
        assert pattern in policy.COVERAGE_INCLUDE_PATTERNS


def test_makefile_exposes_marker_based_fast_and_integration_lanes() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    assert "test-fast:" in text
    assert "test-integration:" in text
    assert '-m "not integration and not slow and not manual"' in text
    assert "-m integration tests/integration" in text


def test_makefile_exposes_actionlint_and_minimal_packaging_smoke_targets() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    assert "actionlint:" in text
    assert "workflow-lint: actionlint" in text
    assert "command -v actionlint" in text
    assert "actionlint .github/workflows/*.yml" in text

    assert "packaging-smoke-minimal:" in text
    assert (
        "tests/integration/packaging/test_wheel_evidence_pack_verify.py::"
        "test_wheel_install_exposes_core_cli_contracts_outside_repo_tree"
    ) in text
    assert (
        "tests/integration/packaging/test_wheel_evidence_pack_verify.py::"
        "test_wheel_install_can_verify_report_runtime_and_evidence_pack_outside_repo_tree"
    ) in text
    assert (
        "tests/integration/packaging/test_wheel_evidence_pack_verify.py::"
        "test_wheel_install_verify_rejects_ambiguous_directory_outside_repo_tree"
    ) in text
    assert (
        "tests/integration/packaging/test_wheel_evidence_pack_verify.py::"
        "test_wheel_install_runtime_verify_failure_json_outside_repo_tree"
    ) in text
    assert (
        "tests/integration/packaging/test_wheel_evidence_pack_verify.py::"
        "test_wheel_install_evidence_pack_verify_reports_integrity_failure_outside_repo_tree"
    ) in text
    assert "docs-live-fast:" in text
    assert "docs-live:" in text
    assert "scripts/verify_live_examples.py" in text
    docs_live_fast_block = text.split("docs-live-fast:", 1)[1].split("docs-live:", 1)[0]
    docs_live_block = text.split("docs-live:", 1)[1].split("docs-check-build:", 1)[0]

    assert "--markdown-execution-mode host" in docs_live_fast_block
    assert "--skip-markdown-model-loading" in docs_live_fast_block
    assert "--skip-notebook-model-loading" in docs_live_fast_block
    assert "--markdown-execution-mode host" in docs_live_block
    assert "--skip-markdown-model-loading" not in docs_live_block
    assert "--skip-notebook-model-loading" not in docs_live_block


def test_makefile_exposes_front_door_packaging_smoke_target() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    assert "packaging-smoke-front-door:" in text
    assert (
        "tests/integration/packaging/test_wheel_front_door_contract.py::"
        "test_wheel_install_runs_front_door_evaluate_verify_report_html_outside_repo_tree"
    ) in text
    front_door_block = text.split("packaging-smoke-front-door:", 1)[1].split(
        "model-evidence-list:", 1
    )[0]
    assert "INVARLOCK_LIGHT_IMPORT=1" not in front_door_block


def test_makefile_exposes_container_default_smoke_target() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    assert ".PHONY: container-default-smoke container-default-smoke-podman" in text
    assert "container-default-smoke:" in text
    assert "container-default-smoke: runtime-image" in text
    assert "tests/integration/test_container_default_smoke.py" in text
    target_block = text.split("container-default-smoke:", 1)[1].split(
        "container-default-smoke-podman:", 1
    )[0]
    assert "INVARLOCK_ALLOW_NETWORK=1" in target_block
    assert "INVARLOCK_CONTAINER_DEFAULT_SMOKE=1" in target_block
    assert "INVARLOCK_RUNTIME_IMAGE=$(RUNTIME_IMAGE)" in target_block
    assert "INVARLOCK_CONTAINER_ENGINE=$(CONTAINER_ENGINE)" in target_block


def test_makefile_exposes_container_front_door_smoke_target() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    assert "container-front-door-smoke container-front-door-smoke-podman" in text
    assert "container-front-door-smoke:" in text
    assert "container-front-door-smoke: runtime-image" in text
    target_block = text.split("container-front-door-smoke:", 1)[1].split(
        "container-front-door-smoke-podman:", 1
    )[0]
    assert "INVARLOCK_ALLOW_NETWORK=1" in target_block
    assert "INVARLOCK_CONTAINER_DEFAULT_SMOKE=1" in target_block
    assert "INVARLOCK_RUNTIME_IMAGE=$(RUNTIME_IMAGE)" in target_block
    assert (
        "tests/integration/test_container_default_smoke.py::test_container_default_front_door_smoke_runs_evaluate_verify_and_report_html"
        in target_block
    )


def test_makefile_exposes_typed_surface_target() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    assert "mypy-typed-surface:" in text
    for path in (
        "src/invarlock/observability/metrics.py",
        "src/invarlock/config.py",
        "src/invarlock/adapters/auto.py",
        "src/invarlock/core/builtin_plugin_catalog.py",
        "src/invarlock/core/config_loader.py",
        "src/invarlock/core/metric_provider_resolution.py",
        "src/invarlock/core/run_orchestrator_execute_seed.py",
        "src/invarlock/core/run_orchestrator_execute_environment.py",
        "src/invarlock/core/run_orchestrator_execute_dataset.py",
        "src/invarlock/core/run_orchestrator_execute_attempts.py",
        "src/invarlock/core/run_orchestrator_execute_execution.py",
        "src/invarlock/core/run_orchestrator_execute_helpers.py",
        "src/invarlock/cli/app.py",
        "src/invarlock/cli/runtime_verify.py",
        "src/invarlock/eval/probes/mi.py",
        "src/invarlock/reporting/report_schema.py",
        "src/invarlock/runtime_security_helpers.py",
    ):
        assert path in text


def test_makefile_exposes_lockfile_sync_target() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    assert "lock-sync:" in text
    assert "UV_NO_CACHE=1 uv lock --check" in text


def test_makefile_marks_release_helper_targets_phony() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    for target in (
        "workflow-lint",
        "container-default-smoke",
        "container-default-smoke-podman",
        "container-front-door-smoke",
        "container-front-door-smoke-podman",
        "ci-matrix",
        "eval-loop",
        "ci-local-precommit",
        "ci-local-verbose",
    ):
        assert target in text

    assert ".PHONY: ci-matrix eval-loop" in text
    assert ".PHONY: ci-local-precommit ci-local-verbose" in text


def test_makefile_prefers_workspace_python_selector_for_local_targets() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    assert "PYTHON ?= $(shell bash scripts/select_workspace_python.sh)" in text
    assert "PYTHON ?= $(shell bash scripts/select_python.sh)" not in text


def test_coverage_include_does_not_embed_space_prefixed_cli_patterns() -> None:
    policy = _load_coverage_policy()
    include = policy.coverage_include()

    assert ", src/invarlock/cli/*" not in include
    assert ", invarlock/cli/*" not in include
    assert "src/invarlock/cli/*" in include
    assert "src/invarlock/cli/commands/*" in include
    assert "src/invarlock/public_contracts.py" in include
    assert "src/invarlock/evidence_pack.py" in include
    assert "src/invarlock/runtime_security.py" in include
    assert "invarlock/cli/commands/*" in include
