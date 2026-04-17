# InvarLock Development Makefile
# Optional development shortcuts

.PHONY: help install dev-install lock-sync test test-fast test-integration test-assurance lint mypy-typed-surface format clean docsclean deepclean docs docs-ci verify verify-ruff cli-smoke-core cli-smoke-advanced coverage coverage-enforce docs-serve docs-deploy pre-commit pre-commit-install docs-check docs-live docs-live-fast docs-lint docs-check-build docs-check-links docs-lint-markdown docs-lint-spell ci-local ci-local-list ci-local-job ci-local-dry contracts-check contracts-sync model-evidence-list model-evidence-sweep runtime-image runtime-image-podman runtime-image-cuda runtime-image-cuda-podman runtime-smoke runtime-smoke-podman runtime-smoke-cuda runtime-smoke-cuda-podman runtime-verify actionlint packaging-smoke-minimal packaging-smoke-front-door ensure-mypy

PYTHON ?= $(shell bash scripts/select_workspace_python.sh)
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy
COVERAGE := $(PYTHON) -m coverage
MKDOCS := $(PYTHON) -m mkdocs
PRE_COMMIT := $(PYTHON) -m pre_commit
MODEL_EVIDENCE_ARGS ?=
CONTAINER_ENGINE ?= $(shell if command -v docker >/dev/null 2>&1; then echo docker; elif command -v podman >/dev/null 2>&1; then echo podman; fi)
RUNTIME_IMAGE ?= invarlock-runtime:local
RUNTIME_IMAGE_CUDA ?= invarlock-runtime:cuda-local
RUNTIME_IMAGE_CUDA_REQUIREMENTS ?= requirements/workflows/runtime-image-py312-cu128.txt
RUNTIME_IMAGE_CUDA_INDEX_URL ?= https://download.pytorch.org/whl/cu128
RUNTIME_IMAGE_DIGEST ?= sha256:local-runtime-image
COVERAGE_POLICY := $(PYTHON) scripts/coverage_policy.py

# Keep repo-wide coverage practical while still exercising the CLI command
# surface that would otherwise pull the project floor below the real trust core.
COVERAGE_TESTS_CORE := \
	tests/core tests/guards tests/reporting tests/calibration tests/scripts

COVERAGE_TESTS_RUN := \
	tests/cli/run

COVERAGE_TESTS_VERIFY := \
	tests/cli/test_verify*.py tests/cli/test_cli_command_help_smoke.py \
	tests/cli/test_policy_commands.py tests/cli/test_evidence_pack_commands.py

COVERAGE_TESTS_CONFIG := \
	tests/cli/test_config_failfast.py tests/cli/test_error_codes.py \
	tests/cli/test_config.py tests/cli/test_config_validation.py \
	tests/cli/test_config_runtime_loader.py tests/cli/test_config_schema_and_loader.py \
	tests/cli/test_device.py tests/cli/test_config_and_device.py

COVERAGE_TESTS_EVAL := \
	tests/eval/test_metrics*.py tests/eval/test_baseline_artifacts.py \
	tests/eval/test_validate_module*.py tests/eval/test_bench*.py \
	tests/eval/test_metric_gate_summary_inputs.py \
	tests/eval/test_primary_metric*.py \
	tests/eval/test_eval_import_safety.py \
	tests/eval/test_determinism.py tests/eval/test_mask_parity_fail.py \
	tests/eval/test_task_metrics.py tests/eval/test_eval_bootstrap_wrapper.py \
	tests/eval/test_data*.py tests/eval/test_hf_text_provider*.py \
	tests/eval/test_local_jsonl*.py tests/eval/test_synthetic_provider_paths.py \
	tests/eval/test_wikitext2_fast_capacity.py \
	tests/eval/test_provider_deterministic_loader_paths.py \
	tests/eval/test_difficulty_scorer_modes.py \
	tests/eval/providers

COVERAGE_TESTS_EVAL_PROBES := \
	tests/eval/test_fft.py tests/eval/test_fft_probe_paths.py \
	tests/eval/test_mi*.py \
	tests/eval/test_post_attention_probes.py tests/eval/test_post_attention_probe_paths.py

COVERAGE_TESTS_CLI_COMMANDS := \
	tests/cli/test_doctor*.py tests/cli/test_plugins*.py tests/cli/test_evaluate*.py \
	tests/cli/test_export_html*.py tests/cli/test_app*.py \
	tests/cli/test_core_command_surface.py tests/cli/test_execution_mode.py \
	tests/cli/test_removed_command_migrations.py tests/cli/test_python_m_invarlock.py \
	tests/cli/test_explain_gates*.py tests/cli/test_report*.py \
	tests/cli/test_calibrate_harness_*.py tests/cli/test_determinism_preset.py

COVERAGE_TESTS_CLI_HELPERS := \
	tests/cli/test_adapter_auto*.py tests/cli/test_no_color.py \
	tests/cli/test_internal_config_run.py tests/cli/test_json_helpers.py \
	tests/cli/test_runtime_launch_plan_contract.py \
	tests/cli/test_overhead_extraction.py

COVERAGE_TESTS_OBSERVABILITY := \
	tests/observability

COVERAGE_TESTS_ADAPTERS := \
	tests/adapters/test_adapter_contracts.py \
	tests/adapters/test_adapter_auto_runtime.py \
	tests/adapters/test_hf_loading_helpers.py \
	tests/adapters/test_hf_multimodal_adapter.py \
	tests/adapters/test_adapter_errors.py \
	tests/adapters/test_adapters_hf_and_integration.py

COVERAGE_TESTS_RUNTIME := \
	tests/cli/test_container_default_contract.py \
	tests/cli/test_container_delegation.py \
	tests/runtime/test_network_policy.py \
	tests/runtime/test_runtime_image_contract.py \
	tests/runtime/test_runtime_manifest_contract.py \
	tests/runtime/test_runtime_security_container.py \
	tests/runtime/test_runtime_security_core.py \
	tests/runtime/test_runtime_security_facade.py \
	tests/runtime/test_runtime_security_manifest.py \
	tests/runtime/test_runtime_security_paths.py \
	tests/runtime/test_runtime_security_decision_matrix.py

COVERAGE_TESTS := \
	$(COVERAGE_TESTS_CORE) \
	$(COVERAGE_TESTS_RUN) \
	$(COVERAGE_TESTS_VERIFY) \
	$(COVERAGE_TESTS_CONFIG) \
	$(COVERAGE_TESTS_EVAL) \
	$(COVERAGE_TESTS_CLI_COMMANDS) \
	$(COVERAGE_TESTS_CLI_HELPERS) \
	$(COVERAGE_TESTS_OBSERVABILITY)

COVERAGE_MODULES := \
	$(shell $(COVERAGE_POLICY) coverage-modules)

COVERAGE_INCLUDE := $(shell $(COVERAGE_POLICY) coverage-include)
MYPY_TYPED_SURFACE := \
	src/invarlock/observability/alerting.py \
	src/invarlock/observability/core.py \
	src/invarlock/observability/exporters.py \
	src/invarlock/observability/health.py \
	src/invarlock/observability/metrics.py \
	src/invarlock/observability/utils.py \
	src/invarlock/config.py \
	src/invarlock/adapters/auto.py \
	src/invarlock/core/config_loader.py \
	src/invarlock/core/config_runtime.py \
	src/invarlock/core/metric_kind_contract.py \
	src/invarlock/core/metric_provider_resolution.py \
	src/invarlock/core/registry.py \
	src/invarlock/core/runner_eval_metrics_multimodal.py \
	src/invarlock/core/builtin_plugin_catalog.py \
	src/invarlock/core/run_orchestrator_execute_seed.py \
	src/invarlock/core/run_orchestrator_execute_environment.py \
	src/invarlock/core/run_orchestrator_execute_dataset.py \
	src/invarlock/core/run_orchestrator_execute_attempts.py \
	src/invarlock/core/run_orchestrator_execute_execution.py \
	src/invarlock/core/run_orchestrator_execute_events.py \
	src/invarlock/core/run_orchestrator_execute_helpers.py \
	src/invarlock/core/run_orchestrator_execute_outcome.py \
	src/invarlock/core/run_orchestrator_execute_pipeline.py \
	src/invarlock/cli/__init__.py \
	src/invarlock/cli/__main__.py \
	src/invarlock/cli/_json.py \
	src/invarlock/cli/app.py \
	src/invarlock/cli/evaluate_output.py \
	src/invarlock/cli/evaluate_phases.py \
	src/invarlock/cli/commands/evaluate.py \
	src/invarlock/cli/commands/verify.py \
	src/invarlock/cli/runtime_verify.py \
	src/invarlock/eval/probes/mi.py \
	src/invarlock/reporting/report_confidence.py \
	src/invarlock/reporting/report_schema.py \
	src/invarlock/reporting/report_types.py \
	src/invarlock/reporting/verify_check_helpers.py \
	src/invarlock/runtime_security.py \
	src/invarlock/runtime_security_helpers.py \
	src/invarlock/runtime_security_container.py \
	src/invarlock/runtime_security_manifest.py

TEST_DIR_TARGETS := adapters calibration ci cli core docs edits eval fuzzing guards integration lint observability plugins evidence_packs reporting runtime scripts
GROUPED_TEST_DIR_TARGETS := $(filter-out integration,$(TEST_DIR_TARGETS))

help:  ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Installation
install:  ## Install package
	$(MAKE) ensure-python
	$(PIP) install -e .

dev-install:  ## Install package with development dependencies
	$(MAKE) ensure-python
	$(PIP) install -e ".[dev]"

lock-sync:  ## Check uv.lock is in sync with pyproject.toml
	UV_NO_CACHE=1 uv lock --check

##@ Development
test:  ## Run tests
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTEST) tests/ -v

test-fast:  ## Run the fast lane with marker selection
	$(MAKE) ensure-python
	INVARLOCK_LIGHT_IMPORT=1 INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS=0 \
		PYTHONPATH=src $(PYTEST) -q -m "not integration and not slow and not manual" tests

test-integration:  ## Run the integration lane
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTEST) -q -m integration tests/integration

##@ Coverage
coverage:  ## Run tests with coverage and generate XML
	$(MAKE) ensure-python
	$(COVERAGE) erase
	rm -f .coverage.*
	PYTHONPATH=src $(PYTEST) -q $(COVERAGE_TESTS) \
		$(COVERAGE_MODULES) \
		--cov-branch \
		--cov-report=term --cov-report=xml:reports/cov.xml --cov-fail-under=90
	PYTHONPATH=src $(COVERAGE) run --append -m pytest -q -p no:cov \
		$(COVERAGE_TESTS_EVAL_PROBES)
	PYTHONPATH=src $(COVERAGE) run --append -m pytest -q -p no:cov \
		$(COVERAGE_TESTS_RUNTIME)
	PYTHONPATH=src $(COVERAGE) run --append -m pytest -q -p no:cov \
		$(COVERAGE_TESTS_ADAPTERS)
	$(COVERAGE) report --include="$(COVERAGE_INCLUDE)" --fail-under=90
	$(COVERAGE) xml --include="$(COVERAGE_INCLUDE)" -o reports/cov.xml

coverage-enforce:  ## Run coverage and enforce per-file thresholds
	$(MAKE) coverage
	$(PYTHON) scripts/check_coverage_thresholds.py --coverage reports/cov.xml --json reports/thresholds.json

# Grouped test targets
.PHONY: $(addprefix test-,$(GROUPED_TEST_DIR_TARGETS))
test-core: TEST_DIR = core
test-core: ## Run tests/core
test-cli: TEST_DIR = cli
test-cli: ## Run tests/cli
test-eval: TEST_DIR = eval
test-eval: ## Run tests/eval
test-guards: TEST_DIR = guards
test-guards: ## Run tests/guards
test-edits: TEST_DIR = edits
test-edits: ## Run tests/edits
test-adapters: TEST_DIR = adapters
test-adapters: ## Run tests/adapters
test-calibration: TEST_DIR = calibration
test-calibration: ## Run tests/calibration
test-docs: TEST_DIR = docs
test-docs: ## Run tests/docs
test-fuzzing: TEST_DIR = fuzzing
test-fuzzing: ## Run tests/fuzzing
test-lint: TEST_DIR = lint
test-lint: ## Run tests/lint
test-observability: TEST_DIR = observability
test-observability: ## Run tests/observability
test-plugins: TEST_DIR = plugins
test-plugins: ## Run tests/plugins
test-evidence_packs: TEST_DIR = evidence_packs
test-evidence_packs: ## Run tests/evidence_packs
test-reporting: TEST_DIR = reporting
test-reporting: ## Run tests/reporting
test-runtime: TEST_DIR = runtime
test-runtime: ## Run tests/runtime
test-scripts: TEST_DIR = scripts
test-scripts: ## Run tests/scripts
test-ci: TEST_DIR = ci
test-ci: ## Run tests/ci
$(addprefix test-,$(GROUPED_TEST_DIR_TARGETS)):
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTEST) -q tests/$(TEST_DIR)

test-assurance:  ## Run assurance-related tests only
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTEST) -q \
		tests/ci/test_golden_runs_offline.py \
		tests/ci/test_support_matrix_consistency.py \
		tests/adapters/test_adapter_capability_contract.py \
		tests/eval/test_assurance_contracts.py \
		tests/docs/test_claim_surface_consistency.py \
		tests/docs/test_assurance_xref_linter.py \
		tests/reporting/test_public_contracts.py \
		tests/reporting/test_evidence_pack_contract.py \
		tests/reporting/test_policy_pack_contract.py \
		tests/reporting/test_policy_utils.py::test_compute_policy_digest_matches_assurance_spec

lint:  ## Run linting
	$(MAKE) ensure-ruff
	$(MAKE) ensure-mypy
	$(RUFF) check src/ tests/ scripts/
	$(MYPY) src/

mypy-typed-surface:  ## Run mypy on the enforced typed surface
	$(MAKE) ensure-python
	$(MAKE) ensure-mypy
	PYTHONPATH=src $(MYPY) $(MYPY_TYPED_SURFACE)

format:  ## Format code
	$(MAKE) ensure-ruff
	$(RUFF) format src/ tests/ scripts/
	$(RUFF) check --fix src/ tests/ scripts/

verify:  ## Run verification (pytest -q, runtime verifier, lint, format, markdown + spell docs lint)
	@echo "Running verification..."
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTEST) -q
	OMP_NUM_THREADS=1 SKIP_RUFF=1 INVARLOCK_PYTHON="$(PYTHON)" bash scripts/run_smoke_regression.sh
	$(MAKE) cli-smoke-core
	$(MAKE) cli-smoke-advanced
	$(MAKE) runtime-verify
	$(MAKE) verify-ruff
	$(MAKE) contracts-check
	$(MAKE) docs-lint
	@if [ -n "$$VERIFY_DOCS_API" ]; then \
		$(PYTHON) scripts/validate_docs_api_refs.py; \
	fi
	@echo "Verification completed successfully"

verify-ruff:  ## Run the Ruff checks used by make verify
	$(MAKE) ensure-ruff
	$(RUFF) check src/ tests/ scripts/
	$(RUFF) format --check src/ tests/ scripts/

cli-smoke-core:  ## Smoke the simplified core CLI surface
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTHON) -m invarlock --help >/dev/null
	PYTHONPATH=src $(PYTHON) -m invarlock evaluate --help >/dev/null
	PYTHONPATH=src $(PYTHON) -m invarlock verify --help >/dev/null
	PYTHONPATH=src $(PYTHON) -m invarlock report html --help >/dev/null
	PYTHONPATH=src $(PYTHON) -m invarlock doctor --json >/dev/null

cli-smoke-advanced:  ## Smoke the advanced CLI namespace
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTHON) -m invarlock advanced --help >/dev/null
	PYTHONPATH=src $(PYTHON) -m invarlock advanced evidence-pack --help >/dev/null
	PYTHONPATH=src $(PYTHON) -m invarlock advanced policy --help >/dev/null
	PYTHONPATH=src $(PYTHON) -m invarlock advanced plugins --help >/dev/null
	PYTHONPATH=src $(PYTHON) -m invarlock advanced calibrate --help >/dev/null

actionlint:  ## Lint GitHub Actions workflow files
	@command -v actionlint >/dev/null 2>&1 || { \
		echo "❌ actionlint is required but not installed; add it to PATH first."; \
		exit 1; \
	}
	actionlint .github/workflows/*.yml

packaging-smoke-minimal:  ## Smoke the minimal wheel install around the public contract and evidence-pack verify path
	$(MAKE) ensure-python
	@PYTHON="$$(if [ -x .venv/bin/python ]; then printf '%s' .venv/bin/python; else printf '%s' "$(PYTHON)"; fi)"; \
	INVARLOCK_LIGHT_IMPORT=1 PYTHONPATH=src "$$PYTHON" -m pytest -q \
		tests/integration/packaging/test_wheel_evidence_pack_verify.py::test_wheel_install_exposes_core_cli_contracts_outside_repo_tree \
		tests/integration/packaging/test_wheel_evidence_pack_verify.py::test_wheel_install_can_verify_report_runtime_and_evidence_pack_outside_repo_tree \
		tests/integration/packaging/test_wheel_evidence_pack_verify.py::test_wheel_install_verify_rejects_ambiguous_directory_outside_repo_tree \
		tests/integration/packaging/test_wheel_evidence_pack_verify.py::test_wheel_install_runtime_verify_failure_json_outside_repo_tree \
		tests/integration/packaging/test_wheel_evidence_pack_verify.py::test_wheel_install_evidence_pack_verify_reports_integrity_failure_outside_repo_tree

packaging-smoke-front-door:  ## Smoke installed-wheel evaluate -> verify -> report html from outside the repo tree
	$(MAKE) ensure-python
	@PYTHON="$$(if [ -x .venv/bin/python ]; then printf '%s' .venv/bin/python; else printf '%s' "$(PYTHON)"; fi)"; \
	PYTHONPATH=src "$$PYTHON" -m pytest -q \
		tests/integration/packaging/test_wheel_front_door_contract.py::test_wheel_install_runs_front_door_evaluate_verify_report_html_outside_repo_tree

model-evidence-list:  ## Print the maintained shipped-model evidence manifest
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTHON) scripts/model_evidence_sweep.py --list-json $(MODEL_EVIDENCE_ARGS)

model-evidence-sweep:  ## Run the maintained shipped-model evidence sweep
	$(MAKE) ensure-python
	PYTHONPATH=src INVARLOCK_ALLOW_NETWORK=1 $(PYTHON) scripts/model_evidence_sweep.py $(MODEL_EVIDENCE_ARGS)

runtime-image:  ## Build the local container runtime image used for default execution
	@test -n "$(CONTAINER_ENGINE)" || { echo "❌ An OCI container engine (Docker or Podman) is required."; exit 1; }
	@if $(CONTAINER_ENGINE) image inspect $(RUNTIME_IMAGE) >/dev/null 2>&1; then $(CONTAINER_ENGINE) image rm -f $(RUNTIME_IMAGE) >/dev/null 2>&1 || true; fi
	$(CONTAINER_ENGINE) build -f runtime/Dockerfile -t $(RUNTIME_IMAGE) .

runtime-image-podman: CONTAINER_ENGINE=podman
runtime-image-podman: runtime-image  ## Build the local container runtime image with Podman

runtime-image-cuda:  ## Build the local CUDA container runtime image for GPU-backed default execution
	@test -n "$(CONTAINER_ENGINE)" || { echo "❌ An OCI container engine (Docker or Podman) is required."; exit 1; }
	@if $(CONTAINER_ENGINE) image inspect $(RUNTIME_IMAGE_CUDA) >/dev/null 2>&1; then $(CONTAINER_ENGINE) image rm -f $(RUNTIME_IMAGE_CUDA) >/dev/null 2>&1 || true; fi
	$(CONTAINER_ENGINE) build \
		--build-arg RUNTIME_REQUIREMENTS_AMD64=$(RUNTIME_IMAGE_CUDA_REQUIREMENTS) \
		--build-arg RUNTIME_REQUIREMENTS_ARM64=requirements/workflows/runtime-image-py312-aarch64.txt \
		--build-arg PYTORCH_EXTRA_INDEX_URL=$(RUNTIME_IMAGE_CUDA_INDEX_URL) \
		-f runtime/Dockerfile \
		-t $(RUNTIME_IMAGE_CUDA) .

runtime-image-cuda-podman: CONTAINER_ENGINE=podman
runtime-image-cuda-podman: runtime-image-cuda  ## Build the local CUDA container runtime image with Podman

runtime-smoke:  ## Smoke the local container runtime image
	@test -n "$(CONTAINER_ENGINE)" || { echo "❌ An OCI container engine (Docker or Podman) is required."; exit 1; }
	$(CONTAINER_ENGINE) run --rm \
		--entrypoint python \
		$(RUNTIME_IMAGE) \
		-c "import datasets, safetensors, torch, transformers; print('runtime image imports ok')"

runtime-smoke-podman: CONTAINER_ENGINE=podman
runtime-smoke-podman: runtime-smoke  ## Smoke the local container runtime image with Podman

runtime-smoke-cuda: RUNTIME_IMAGE=$(RUNTIME_IMAGE_CUDA)
runtime-smoke-cuda: runtime-smoke  ## Smoke the local CUDA container runtime image

runtime-smoke-cuda-podman: CONTAINER_ENGINE=podman
runtime-smoke-cuda-podman: RUNTIME_IMAGE=$(RUNTIME_IMAGE_CUDA)
runtime-smoke-cuda-podman: runtime-smoke  ## Smoke the local CUDA container runtime image with Podman

runtime-verify:  ## Smoke the Python runtime verifier on the fixture bundle
	PYTHONPATH=src $(PYTHON) -m invarlock.cli.runtime_verify \
		--report tests/fixtures/runtime_provenance/evaluation.report.json \
		--manifest tests/fixtures/runtime_provenance/runtime.manifest.json \
		--json

##@ CI/Build
clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docsclean: ## Remove local MkDocs site build
	rm -rf site/

deepclean: ## Remove all generated artifacts, caches, and run outputs (destructive)
	rm -rf \
		build/ dist/ *.egg-info .eggs/ \
		site/ \
		data/ \
		node_modules/ \
		reports/ reports_*/ \
		runs/ runs_cfg/ \
		pip-wheel-metadata/ \
		__pycache__/ */__pycache__/ \
		.pytest_cache/ .mypy_cache/ .ruff_cache/ .pre-commit-cache/ .npm-cache/ .npm-prefix/ \
		.hypothesis/ .evaluate_tmp/ tmp/ tmp_*/ \
		.tox/ .nox/ \
		.coverage coverage.xml htmlcov/ \
		test_config.yaml tmp_cfg.yaml \
		*.pyc *.pyo

docs-serve: ## Serve documentation locally
	$(MAKE) ensure-python
	$(MKDOCS) serve -a 127.0.0.1:8000

docs-deploy: ## Build and publish docs to gh-pages (local)
	$(MAKE) ensure-python
	$(MKDOCS) gh-deploy --clean --force

pre-commit-install: ## Install pre-commit hooks locally
	$(MAKE) ensure-python
	$(PIP) install -U pre-commit
	$(PRE_COMMIT) install

pre-commit: ## Run pre-commit on all files
	$(MAKE) ensure-python
	$(PRE_COMMIT) run --all-files --show-diff-on-failure

docs:  ## Build docs with default mkdocs.yml (CI/networked)
	$(MAKE) ensure-python
	$(MKDOCS) build --strict

docs-ci:  ## Build documentation and run link checker
	$(MAKE) ensure-python
	$(MKDOCS) build --strict
	$(PYTHON) scripts/check_docs_links.py

## (Consolidated) Single docs-serve target defined above

##@ Evaluation
eval-loop:  ## Run automated evaluation loop (baseline + quant8 quickstart)
	@echo "Running automated evaluation workflow..."
	@rm -rf runs/eval_loop reports/eval/eval_loop
	@INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate \
		--source sshleifer/tiny-gpt2 --edited sshleifer/tiny-gpt2 --adapter auto \
		--profile release --tier balanced \
		--preset configs/presets/causal_lm/wikitext2_512.yaml \
		--edit-config configs/overlays/edits/quant_rtn/8bit_attn.yaml \
		--out runs/eval_loop \
		--report-out reports/eval/eval_loop
	@echo "Evaluation complete. Artifacts: runs/eval_loop/, reports/eval/eval_loop/"

##@ Utilities
ci-matrix:  ## Verify CI matrix
	bash scripts/verify_ci_matrix.sh

contracts-check:  ## Ensure packaged contracts match the repo contract source
	$(MAKE) ensure-python
	$(PYTHON) scripts/sync_packaged_contracts.py --check

contracts-sync:  ## Copy repo contracts into src/invarlock/_data/contracts
	$(MAKE) ensure-python
	$(PYTHON) scripts/sync_packaged_contracts.py --write

## (manual-tests target removed)


.PHONY: ensure-ruff
ensure-python:
	@$(PYTHON) -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" || { \
		printf '%s\n' "Python 3.12+ required. Selected: $(PYTHON) ($$($(PYTHON) --version 2>&1))" >&2; \
		exit 1; \
	}

ensure-ruff:
	@$(MAKE) ensure-python
	@if $(PYTHON) -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('ruff') else 1)"; then \
		:; \
	else \
		printf '%s\n' "ruff is required but not installed; install it in the selected environment (e.g. '$(PYTHON) -m pip install ruff')" >&2; \
		exit 1; \
	fi

ensure-mypy:
	@$(MAKE) ensure-python
	@if $(PYTHON) -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('mypy') else 1)"; then \
		:; \
	else \
		printf '%s\n' "mypy is required but not installed; install it in the selected environment (e.g. '$(PYTHON) -m pip install mypy')" >&2; \
		exit 1; \
	fi

## (verify-ci and verify-release targets removed)

.PHONY: docs-check docs-live docs-live-fast docs-lint docs-check-build docs-check-links docs-lint-markdown docs-lint-spell
docs-check: ## Run consolidated docs validation plus curated live examples
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTHON) scripts/docs_check.py --all

docs-live-fast: ## Live-run the curated deterministic docs and notebook subset
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTHON) scripts/verify_live_examples.py \
		--markdown-execution-mode host \
		--skip-markdown-model-loading \
		--skip-notebook-model-loading \
		--paths \
		README.md \
		docs/user-guide/getting-started.md \
		docs/user-guide/quickstart.md \
		notebooks/invarlock_python_api.ipynb \
		notebooks/invarlock_policy_tiers.ipynb

docs-live: ## Live-run runnable markdown CLI examples and notebooks
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTHON) scripts/verify_live_examples.py \
		--markdown-execution-mode host

docs-check-build: ## Build docs strictly and run link checks
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTHON) scripts/docs_check.py --build --links

docs-check-links: ## Run docs link checks only
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTHON) scripts/docs_check.py --links

docs-lint: ## Lint docs (markdown + spell)
	$(MAKE) ensure-python
	$(PYTHON) scripts/docs_lint.py --all

docs-lint-markdown: ## Lint docs markdown style only
	$(MAKE) ensure-python
	$(PYTHON) scripts/docs_lint.py --markdown

docs-lint-spell: ## Spell-check docs only
	$(MAKE) ensure-python
	$(PYTHON) scripts/docs_lint.py --spell

.PHONY: config-check
config-check: ## Verify config includes and adapter availability
	$(MAKE) ensure-python
	$(PYTHON) scripts/check_config_integrity.py configs

##@ Local CI (act)
# Run GitHub Actions workflows locally using nektos/act
# Install: brew install act (macOS) or see https://github.com/nektos/act

ci-local:  ## Run all CI workflows locally (requires act; this helper currently assumes Docker)
	@command -v act >/dev/null 2>&1 || { echo "❌ 'act' not found. Install: brew install act"; exit 1; }
	@command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required for this local act helper. Start Docker Desktop first."; exit 1; }
	act push --job tests-docs --env INVARLOCK_LIGHT_IMPORT=1 --env INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS=0

ci-local-list:  ## List available workflows and jobs
	@command -v act >/dev/null 2>&1 || { echo "❌ 'act' not found. Install: brew install act"; exit 1; }
	act --list

ci-local-job:  ## Run a specific job: make ci-local-job JOB=tests-docs
	@command -v act >/dev/null 2>&1 || { echo "❌ 'act' not found. Install: brew install act"; exit 1; }
	@if [ -z "$(JOB)" ]; then echo "Usage: make ci-local-job JOB=<job-name>"; act --list; exit 1; fi
	act push --job $(JOB) --env INVARLOCK_LIGHT_IMPORT=1

ci-local-dry:  ## Dry-run CI locally (no execution, just shows plan)
	@command -v act >/dev/null 2>&1 || { echo "❌ 'act' not found. Install: brew install act"; exit 1; }
	act push --dryrun

ci-local-precommit:  ## Run pre-commit workflow locally
	@command -v act >/dev/null 2>&1 || { echo "❌ 'act' not found. Install: brew install act"; exit 1; }
	act push --workflows .github/workflows/pre-commit.yml

ci-local-verbose:  ## Run CI locally with verbose output for debugging
	@command -v act >/dev/null 2>&1 || { echo "❌ 'act' not found. Install: brew install act"; exit 1; }
	act push --job tests-docs --verbose --env INVARLOCK_LIGHT_IMPORT=1
