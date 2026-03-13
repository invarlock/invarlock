# InvarLock Development Makefile
# Optional development shortcuts

.PHONY: help install dev-install test test-assurance lint format clean docsclean deepclean docs docs-ci verify coverage coverage-enforce docs-serve docs-deploy pre-commit pre-commit-install docs-check docs-lint docs-check-build docs-check-links docs-lint-markdown docs-lint-spell ci-local ci-local-list ci-local-job ci-local-dry

PYTHON ?= $(shell bash scripts/select_python.sh)
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy
COVERAGE := $(PYTHON) -m coverage
MKDOCS := $(PYTHON) -m mkdocs
PRE_COMMIT := $(PYTHON) -m pre_commit

# Keep repo-wide coverage practical while still exercising the CLI command
# surface that would otherwise pull the project floor below the real trust core.
COVERAGE_TESTS_CORE := \
	tests/core tests/guards tests/reporting tests/calibration tests/scripts

COVERAGE_TESTS_RUN := \
	tests/cli/run

COVERAGE_TESTS_VERIFY := \
	tests/cli/test_verify*.py tests/cli/test_cli_command_help_smoke.py tests/cli/test_policy_commands.py

COVERAGE_TESTS_CONFIG := \
	tests/cli/test_config_failfast.py tests/cli/test_error_codes.py \
	tests/cli/test_config.py tests/cli/test_config_cases.py \
	tests/cli/test_config_runtime_loader.py tests/cli/test_config_schema_and_loader.py \
	tests/cli/test_device.py tests/cli/test_config_and_device.py

COVERAGE_TESTS_EVAL := \
	tests/eval/test_metrics*.py tests/eval/test_report*.py \
	tests/eval/test_validate_module.py tests/eval/test_baseline_artifacts.py \
	tests/eval/test_bench.py tests/eval/test_primary_metric*.py \
	tests/eval/test_determinism.py tests/eval/test_mask_parity_fail.py

COVERAGE_TESTS_CLI_COMMANDS := \
	tests/cli/test_doctor*.py tests/cli/test_plugins*.py tests/cli/test_evaluate*.py \
	tests/cli/test_export_html*.py tests/cli/test_app*.py \
	tests/cli/test_explain_gates*.py tests/cli/test_report*.py \
	tests/cli/test_calibrate_harness_artifacts.py tests/cli/test_determinism_preset.py

COVERAGE_TESTS_CLI_HELPERS := \
	tests/cli/test_adapter_auto*.py tests/cli/test_no_color.py \
	tests/cli/test_json_helpers.py tests/unit/test_overhead_extraction.py

COVERAGE_TESTS := \
	$(COVERAGE_TESTS_CORE) \
	$(COVERAGE_TESTS_RUN) \
	$(COVERAGE_TESTS_VERIFY) \
	$(COVERAGE_TESTS_CONFIG) \
	$(COVERAGE_TESTS_EVAL) \
	$(COVERAGE_TESTS_CLI_COMMANDS) \
	$(COVERAGE_TESTS_CLI_HELPERS)

COVERAGE_MODULES := \
	--cov=src/invarlock/eval --cov=src/invarlock/guards --cov=src/invarlock/calibration \
	--cov=src/invarlock/cli --cov=src/invarlock/core --cov=src/invarlock/reporting \
	--cov=invarlock.public_contracts --cov=invarlock.policy_pack

help:  ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Installation
install:  ## Install package
	$(MAKE) ensure-python
	$(PIP) install -e .

dev-install:  ## Install package with development dependencies
	$(MAKE) ensure-python
	$(PIP) install -e ".[dev]"

##@ Development
test:  ## Run tests
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTEST) tests/ -v

##@ Coverage
coverage:  ## Run tests with coverage and generate XML
	$(MAKE) ensure-python
	$(COVERAGE) erase
	PYTHONPATH=src $(PYTEST) -q $(COVERAGE_TESTS) \
		$(COVERAGE_MODULES) \
		--cov-branch \
		--cov-report=term --cov-report=xml:reports/cov.xml --cov-fail-under=90

coverage-enforce:  ## Run coverage and enforce per-file thresholds
	$(MAKE) coverage
	$(PYTHON) scripts/check_coverage_thresholds.py --coverage reports/cov.xml --json reports/thresholds.json

# Grouped test targets
.PHONY: test-core test-cli test-eval test-guards test-edits test-adapters test-plugins test-scripts test-ci
test-core:
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTEST) -q tests/core
test-cli:
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTEST) -q tests/cli
test-eval:
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTEST) -q tests/eval
test-guards:
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTEST) -q tests/guards
test-edits:
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTEST) -q tests/edits
test-adapters:
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTEST) -q tests/adapters
test-plugins:
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTEST) -q tests/plugins
test-scripts:
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTEST) -q tests/scripts
test-ci:
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTEST) -q tests/ci

test-assurance:  ## Run assurance-related tests only
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTEST) -q \
		tests/api/test_assurance_facade.py \
		tests/ci/test_golden_runs_offline.py \
		tests/ci/test_support_matrix_consistency.py \
		tests/adapters/test_adapter_capability_contract.py \
		tests/eval/test_assurance_contracts.py \
		tests/docs/test_claim_surface_consistency.py \
		tests/docs/test_assurance_xref_linter.py \
		tests/reporting/test_public_contracts.py \
		tests/reporting/test_policy_pack_contract.py \
		tests/reporting/test_policy_utils.py::test_compute_policy_digest_matches_assurance_spec

lint:  ## Run linting
	$(MAKE) ensure-ruff
	$(RUFF) check src/ tests/ scripts/
	$(MYPY) src/

format:  ## Format code
	$(MAKE) ensure-ruff
	$(RUFF) format src/ tests/ scripts/
	$(RUFF) check --fix src/ tests/ scripts/

verify:  ## Run verification (pytest -q, lint, format, markdownlint)
	@echo "Running verification..."
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTEST) -q
	OMP_NUM_THREADS=1 SKIP_RUFF=1 INVARLOCK_PYTHON="$(PYTHON)" bash scripts/run_smoke_regression.sh
	$(MAKE) ensure-ruff
	$(RUFF) check src/ tests/ scripts/
	$(RUFF) format --check src/ tests/ scripts/
	$(PYTHON) scripts/docs_lint.py --markdown
	@if [ -n "$$VERIFY_DOCS_API" ]; then \
		$(PYTHON) scripts/validate_docs_api_refs.py; \
	fi
	@echo "Verification completed successfully"

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
		reports/ reports_*/ reports_report/ \
		runs/ runs_cfg/ run1/ run2/ \
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

## (verify-ci and verify-release targets removed)

.PHONY: docs-check docs-lint docs-check-build docs-check-links docs-lint-markdown docs-lint-spell
docs-check: ## Run consolidated docs validation (build, links, refs, examples, consistency)
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTHON) scripts/docs_check.py --all

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

ci-local:  ## Run all CI workflows locally (requires Docker + act)
	@command -v act >/dev/null 2>&1 || { echo "❌ 'act' not found. Install: brew install act"; exit 1; }
	@command -v docker >/dev/null 2>&1 || { echo "❌ Docker not running. Start Docker Desktop first."; exit 1; }
	act push --job tests-docs --env INVARLOCK_LIGHT_IMPORT=1 --env INVARLOCK_DISABLE_PLUGIN_DISCOVERY=1

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
