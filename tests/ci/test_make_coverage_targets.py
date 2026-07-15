from __future__ import annotations

import importlib.util
import re
import sys
import tomllib
from pathlib import Path


def _load_coverage_policy():
    policy_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "coverage"
        / "check_coverage_thresholds.py"
    )
    spec = importlib.util.spec_from_file_location("coverage_policy", policy_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_wildcard_test_selector_resolves() -> None:
    root = Path(__file__).resolve().parents[2]
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    selectors = set(re.findall(r"tests/[^\s\\]+[*?][^\s\\]*", makefile))

    unresolved = sorted(
        selector for selector in selectors if not list(root.glob(selector))
    )

    assert unresolved == []


def test_coverage_target_uses_the_complete_test_surface() -> None:
    text = (Path(__file__).resolve().parents[2] / "Makefile").read_text(
        encoding="utf-8"
    )

    assert "COVERAGE_TESTS := tests" in text
    assert "COVERAGE_MARKERS := not integration" in text
    assert "COVERAGE_TESTS_" not in text


def test_coverage_target_does_not_enumerate_movable_test_owners() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")
    start = text.index("coverage:  ## Run tests with coverage and generate XML")
    end = text.index("coverage-enforce:", start)
    recipe = text[start:end]

    assert "tests/cli/verify" not in recipe
    assert "tests/eval/metrics" not in recipe
    assert "tests/evidence_packs/test_" not in recipe


def test_coverage_target_excludes_only_the_separate_integration_lane() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    assert "COVERAGE_MARKERS := not integration" in text
    assert '-m "$(COVERAGE_MARKERS)" $(COVERAGE_TESTS)' in text
    assert "COVERAGE_MARKERS := not integration and" not in text


def test_coverage_target_has_no_separate_tail_allowlists() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    assert "COVERAGE_TAIL_SHARDS" not in text
    assert "COVERAGE_FILE=.coverage.tail" not in text
    assert "tail_pid_" not in text


def test_coverage_policy_includes_runtime_security() -> None:
    policy = _load_coverage_policy()
    assert "invarlock/runtime_security.py" in policy.coverage_include()


def test_coverage_enforces_global_floor_after_the_complete_test_run() -> None:
    text = (Path(__file__).resolve().parents[2] / "Makefile").read_text(
        encoding="utf-8"
    )
    start = text.index("coverage:  ## Run tests with coverage and generate XML")
    end = text.index("coverage-enforce:", start)
    recipe = text[start:end]

    assert "--cov-fail-under" not in recipe
    assert (
        '$(COVERAGE) report --include="$(COVERAGE_INCLUDE)" --fail-under=90' in recipe
    )
    assert recipe.index("$(PYTEST) $(PYTEST_WORKER_ARGS)") < recipe.index(
        "--fail-under=90"
    )


def test_coverage_policy_is_shared_with_makefile_and_expanded_surface() -> None:
    policy = _load_coverage_policy()
    root = Path(__file__).resolve().parents[2]
    makefile = (root / "Makefile").read_text(encoding="utf-8")

    assert (
        "COVERAGE_POLICY := $(PYTHON) scripts/coverage/check_coverage_thresholds.py"
        in makefile
    )
    assert "\t$(shell $(COVERAGE_POLICY) coverage-modules)" in makefile
    assert (
        "COVERAGE_INCLUDE := $(shell $(COVERAGE_POLICY) coverage-include)" in makefile
    )
    assert "--allow-missing-threshold-files" not in makefile

    maintained = policy.maintained_coverage_files(root)
    assert maintained
    assert all(
        len(policy._classification_matches(path, root)) == 1 for path in maintained
    )
    assert policy.REPOSITORY_FLOOR == policy.CoverageFloor(line=0.90, branch=0.80)
    assert policy.TIER_FLOORS == {
        "compact_contract": policy.CoverageFloor(line=1.00, branch=1.00),
        "behavioral": policy.CoverageFloor(line=0.95, branch=0.90),
        "live_backend": policy.CoverageFloor(line=0.85, branch=0.75),
    }
    assert policy.COVERAGE_RATCHETS == {}
    assert set(policy.LIVE_HARDWARE_CLOSURE_REQUIREMENTS) == {
        "src/invarlock/guards/exact_svd.py"
    }
    assert policy.COVERAGE_MODULE_FLAGS == ("--cov",)
    assert set(policy.MAINTAINED_ASSURANCE_FILES).issubset(maintained)


def test_checkpoint_identity_coverage_runs_in_complete_parallel_surface() -> None:
    policy = _load_coverage_policy()
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")
    checkpoint_tests = Path(__file__).resolve().parents[1] / "core"

    assert "COVERAGE_TESTS := tests" in text
    assert "coverage-enforce-parallel: PYTEST_WORKERS = auto" in text
    assert "$(MAKE) coverage-enforce PYTEST_WORKERS=$(PYTEST_WORKERS)" in text
    assert (checkpoint_tests / "test_checkpoint_identity.py").is_file()
    assert (
        checkpoint_tests / "test_checkpoint_identity_failure_contracts.py"
    ).is_file()
    checkpoint_path = "src/invarlock/core/checkpoint_identity.py"
    assert checkpoint_path in policy.maintained_coverage_files()
    assert policy._classification_matches(
        checkpoint_path, Path(__file__).resolve().parents[2]
    ) == ("behavioral",)
    assert policy._effective_floor(
        checkpoint_path, "behavioral"
    ) == policy.CoverageFloor(line=0.95, branch=0.90)

    for pattern in (
        "src/invarlock/observability/*",
        "src/invarlock/__init__.py",
        "src/invarlock/calibration.py",
        "src/invarlock/adapters/auto.py",
        "scripts/release/*.py",
        "scripts/evidence_packs/python/editing/streaming_pruning.py",
        "scripts/evidence_packs/python/editing/validate_artifact.py",
        "invarlock/observability/*",
        "invarlock/__init__.py",
        "invarlock/calibration.py",
        "invarlock/adapters/auto.py",
        "evidence_packs/python/editing/streaming_pruning.py",
        "evidence_packs/python/editing/validate_artifact.py",
    ):
        assert pattern in policy.COVERAGE_INCLUDE_PATTERNS


def test_pruning_replay_verifiers_are_bound_to_behavioral_policy() -> None:
    root = Path(__file__).resolve().parents[2]
    policy = _load_coverage_policy()
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    report_include = set(pyproject["tool"]["coverage"]["report"]["include"])
    expected = {
        "scripts/evidence_packs/python/editing/streaming_pruning.py",
        "scripts/evidence_packs/python/editing/validate_artifact.py",
    }

    assert expected.issubset(report_include)
    assert expected.issubset(policy.COVERAGE_INCLUDE_PATTERNS)
    assert expected.issubset(policy.MAINTAINED_ASSURANCE_FILES)
    for path in expected:
        assert policy._classification_matches(path, root) == ("behavioral",)
        assert policy._effective_floor(path, "behavioral") == policy.CoverageFloor(
            line=0.95,
            branch=0.90,
        )


def test_makefile_exposes_marker_based_fast_and_integration_lanes() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    assert "test-fast:" in text
    assert "test-parallel:" in text
    assert "test-integration:" in text
    assert "PYTEST_WORKERS ?= 0" in text
    assert "PYTEST_WORKER_ARGS :=" in text
    assert "test-parallel: PYTEST_WORKERS = auto" in text
    assert "$(MAKE) test-fast PYTEST_WORKERS=$(PYTEST_WORKERS)" in text
    assert '-m "not integration and not slow and not manual"' in text
    assert "$(PYTEST) $(PYTEST_WORKER_ARGS) -q -m" in text
    assert "$(PYTEST) $(PYTEST_WORKER_ARGS) -q tests/$(TEST_DIR)" in text
    assert "-m integration tests/integration" in text
    assert "coverage-enforce-parallel: PYTEST_WORKERS = auto" in text
    assert '-m "$(COVERAGE_MARKERS)" $(COVERAGE_TESTS)' in text
    assert "$(MAKE) coverage-enforce PYTEST_WORKERS=$(PYTEST_WORKERS)" in text


def test_makefile_exposes_actionlint_and_minimal_packaging_smoke_targets() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    assert "actionlint:" in text
    assert "workflow-lint: actionlint" in text
    assert "command -v actionlint" in text
    assert "go install github.com/rhysd/actionlint/cmd/actionlint@v1.7.7" in text
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
    assert "scripts/docs/verify_live_examples.py" in text
    docs_live_fast_block = text.split("docs-live-fast:", 1)[1].split("docs-live:", 1)[0]
    docs_live_block = text.split("docs-live:", 1)[1].split("docs-check-build:", 1)[0]

    assert "--markdown-execution-mode host" in docs_live_fast_block
    assert "--skip-markdown-model-loading" in docs_live_fast_block
    assert "--skip-notebook-model-loading" in docs_live_fast_block
    assert "--markdown-execution-mode host" in docs_live_block
    assert "--skip-markdown-model-loading" not in docs_live_block
    assert "--skip-notebook-model-loading" not in docs_live_block


def test_makefile_exposes_offline_local_hf_pipeline_smoke_target() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    assert "local-hf-pipeline-smoke:" in text
    assert "local-hf-env-check:" in text
    assert "local-hf-env-refresh:" in text
    assert "local-hf-pipeline-smoke-locked:" in text
    assert "scripts/checks/check_local_hf_runtime.py" in text
    assert "uv sync --locked --extra hf --extra ci" in text
    assert "tests/integration/test_local_hf_pipeline_smoke.py" in text
    assert "INVARLOCK_ALLOW_NETWORK=0" in text
    assert "HF_HUB_OFFLINE=1" in text
    assert "TRANSFORMERS_OFFLINE=1" in text
    assert "TOKENIZERS_PARALLELISM=false" in text
    assert "uv run --isolated --locked --extra hf --extra ci" in text


def test_makefile_exposes_mutation_smoke_target() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    assert "mutation-smoke:" in text
    assert "scripts/coverage/mutation_smoke.py" in text


def test_makefile_exposes_slow_statistical_calibration_target() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    assert "statistical-calibration-fast:" in text
    assert "test_logloss_ci_empirical_coverage_smoke" in text
    assert "test_paired_delta_log_ci_property_strict_identity" in text
    assert "statistical-calibration-slow:" in text
    assert "-m slow tests/core/test_bootstrap_calibration_slow.py" in text


def test_makefile_exposes_front_door_packaging_smoke_target() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    assert "packaging-smoke-front-door:" in text
    assert (
        "tests/integration/packaging/test_wheel_front_door_contract.py::"
        "test_wheel_install_verifies_strict_report_bundle_outside_repo_tree"
    ) in text
    assert (
        "tests/integration/packaging/test_wheel_front_door_contract.py::"
        "test_wheel_install_runs_front_door_evaluate_verify_report_html_outside_repo_tree"
    ) in text
    front_door_block = text.split("packaging-smoke-front-door:", 1)[1].split(
        "runtime-image:", 1
    )[0]
    assert "INVARLOCK_LIGHT_IMPORT=1" not in front_door_block


def test_makefile_assurance_lane_includes_strict_assurance_tests() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")
    target_block = text.split("test-assurance:", 1)[1].split("\n\n", 1)[0]

    for expected in (
        "tests/core/test_assurance_contract.py",
        "tests/reporting/validation/test_verify_assurance_guard_chain.py",
        "tests/core/test_bootstrap.py::test_paired_delta_log_ci_property_strict_identity",
        "tests/core/test_bootstrap.py::test_paired_delta_log_ci_property_rejects_mismatched_lengths",
        "tests/guards/contracts/test_unsupported_assurance_shape.py",
    ):
        assert expected in target_block


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
        "src/invarlock/__init__.py",
        "src/invarlock/adapters/auto.py",
        "src/invarlock/core/assurance_contract.py",
        "src/invarlock/core/bootstrap.py",
        "src/invarlock/core/builtin_plugin_catalog.py",
        "src/invarlock/core/config_loader.py",
        "src/invarlock/core/evaluate_plan.py",
        "src/invarlock/core/metric_provider_resolution.py",
        "src/invarlock/core/runner_runtime/eval_metrics_stats.py",
        "src/invarlock/clean_pruning_selection_contracts/snapshot.py",
        "src/invarlock/core/run_orchestrator.py",
        "src/invarlock/core/orchestration/execute.py",
        "src/invarlock/core/orchestration/environment.py",
        "src/invarlock/core/orchestration/attempts.py",
        "src/invarlock/core/orchestration/attempt_results.py",
        "src/invarlock/core/orchestration/execution.py",
        "src/invarlock/core/orchestration/helpers.py",
        "src/invarlock/cli/app.py",
        "src/invarlock/eval/probes/importance.py",
        "src/invarlock/reporting/report_builder_telemetry.py",
        "src/invarlock/reporting/report_builder_support.py",
        "src/invarlock/reporting/report_make.py",
        "src/invarlock/reporting/report_primary_metric_policy.py",
        "src/invarlock/reporting/report_schema.py",
        "src/invarlock/reporting/verify_contract.py",
        "src/invarlock/reporting/verify_strict_accuracy.py",
        "src/invarlock/reporting/verify_strict_ppl.py",
        "src/invarlock/reporting/verify_strict_schedule.py",
        "src/invarlock/runtime_security_helpers.py",
    ):
        assert path in text
    assert "src/invarlock/clean_pruning_selection_evidence.py" not in text


def test_makefile_exposes_lockfile_sync_target() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    assert "lock-sync:" in text
    assert "UV_NO_CACHE=1 uv lock --check" in text


def test_makefile_exposes_isolated_security_gate() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    assert "SECURITY_ARTIFACT_DIR ?= artifacts/supply-chain" in text
    assert "SECURITY_RUN ?= uv run --isolated --locked --extra security-ci" in text
    assert ".PHONY: security supply-chain-security" in text
    assert "security: supply-chain-security" in text
    assert "command -v uv" in text
    assert "scripts/security/generate_sbom.sh --scope tool-environment" in text
    assert "$(SECURITY_ARTIFACT_DIR)/sbom.json" in text
    assert "python scripts/security/run_pip_audit.py" in text


def test_makefile_marks_release_helper_targets_phony() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")

    for target in (
        "workflow-lint",
        "security",
        "supply-chain-security",
        "container-default-smoke",
        "container-default-smoke-podman",
        "container-front-door-smoke",
        "container-front-door-smoke-podman",
        "release-evidence-check",
        "guard-validation-smoke",
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
