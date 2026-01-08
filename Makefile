# InvarLock Development Makefile
# Optional development shortcuts

.PHONY: help install dev-install test lint format clean docsclean deepclean docs docs-ci verify coverage coverage-enforce docs-serve docs-deploy pre-commit pre-commit-install docs-check docs-lint docs-check-build docs-check-links docs-lint-markdown docs-lint-spell ci-local ci-local-list ci-local-job ci-local-dry

help:  ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Installation
install:  ## Install package
	pip install -e .

dev-install:  ## Install package with development dependencies
	pip install -e ".[dev]"

##@ Development
test:  ## Run tests
	PYTHONPATH=src pytest tests/ -v

##@ Coverage
coverage:  ## Run tests with coverage and generate XML
	coverage erase
	PYTHONPATH=src pytest -q \
		tests/core tests/guards tests/reporting tests/cli/run tests/cli/test_run_*.py tests/calibration tests/scripts \
		tests/cli/test_run_command_*.py tests/cli/test_config_failfast.py tests/cli/test_error_codes.py \
		tests/cli/test_verify*.py tests/cli/test_cli_command_help_smoke.py \
		tests/cli/test_calibrate_harness_artifacts.py tests/cli/test_determinism_preset.py tests/cli/test_json_helpers.py \
		tests/cli/test_config.py tests/cli/test_config_more.py tests/cli/test_config_runtime_loader.py tests/cli/test_config_schema_and_loader.py \
		tests/eval/test_metrics*.py tests/eval/test_report*.py tests/eval/test_validate_module.py tests/eval/test_baseline_artifacts.py tests/eval/test_bench.py tests/eval/test_primary_metric*.py \
		tests/eval/test_determinism.py tests/eval/test_mask_parity_fail.py tests/eval/test_certificate*.py \
		--cov=src/invarlock/eval --cov=src/invarlock/guards --cov=src/invarlock/calibration \
		--cov=src/invarlock/cli --cov=src/invarlock/core --cov=src/invarlock/reporting \
		--cov-branch \
		--cov-report=term --cov-report=xml:reports/cov.xml

coverage-enforce:  ## Run coverage and enforce per-file thresholds
	$(MAKE) coverage
	python scripts/check_coverage_thresholds.py --coverage reports/cov.xml --json reports/thresholds.json

# Grouped test targets
.PHONY: test-core test-cli test-eval test-guards test-edits test-adapters test-plugins test-scripts test-ci
test-core:
	PYTHONPATH=src pytest -q tests/core
test-cli:
	PYTHONPATH=src pytest -q tests/cli
test-eval:
	PYTHONPATH=src pytest -q tests/eval
test-guards:
	PYTHONPATH=src pytest -q tests/guards
test-edits:
	PYTHONPATH=src pytest -q tests/edits
test-adapters:
	PYTHONPATH=src pytest -q tests/adapters
test-plugins:
	PYTHONPATH=src pytest -q tests/plugins
test-scripts:
	PYTHONPATH=src pytest -q tests/scripts
test-ci:
	PYTHONPATH=src pytest -q tests/ci

test-assurance:  ## Run assurance-related tests only
	pytest \
		tests/unit/test_assurance_contracts.py \
		tests/unit/test_metrics_masked_lm.py \
		tests/unit/test_structured_edit.py::test_structured_prune_mask_determinism \
		tests/unit/test_cli.py::test_run_command_successful_execution

lint:  ## Run linting
	$(MAKE) ensure-ruff
	python -m ruff check src/ tests/ scripts/
	mypy src/

format:  ## Format code
	$(MAKE) ensure-ruff
	python -m ruff format src/ tests/ scripts/
	python -m ruff check --fix src/ tests/ scripts/

verify:  ## Run verification (pytest -q, lint, format, markdownlint)
	@echo "Running verification..."
	PYTHONPATH=src pytest -q
	OMP_NUM_THREADS=1 SKIP_RUFF=1 bash scripts/run_smoke_regression.sh
	$(MAKE) ensure-ruff
	python -m ruff check src/ tests/ scripts/
	python -m ruff format --check src/ tests/ scripts/
	python scripts/docs_lint.py --markdown
	@if [ -n "$$VERIFY_DOCS_API" ]; then \
		python scripts/validate_docs_api_refs.py; \
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
		data/ contracts/ \
		node_modules/ \
		reports/ reports_*/ reports_report/ \
		runs/ runs_cfg/ run1/ run2/ \
		pip-wheel-metadata/ \
		__pycache__/ */__pycache__/ \
		.pytest_cache/ .mypy_cache/ .ruff_cache/ .pre-commit-cache/ .npm-cache/ .npm-prefix/ \
		.hypothesis/ .certify_tmp/ tmp/ tmp_*/ \
		.tox/ .nox/ \
		.coverage coverage.xml htmlcov/ \
		test_config.yaml tmp_cfg.yaml \
		*.pyc *.pyo

docs-serve: ## Serve documentation locally
	mkdocs serve -a 127.0.0.1:8000

docs-deploy: ## Build and publish docs to gh-pages (local)
	mkdocs gh-deploy --clean --force

pre-commit-install: ## Install pre-commit hooks locally
	python -m pip install -U pre-commit
	pre-commit install

pre-commit: ## Run pre-commit on all files
	pre-commit run --all-files --show-diff-on-failure

docs:  ## Build docs with default mkdocs.yml (CI/networked)
	mkdocs build --strict

docs-ci:  ## Build documentation and run link checker
	mkdocs build --strict
	python scripts/check_docs_links.py

## (Consolidated) Single docs-serve target defined above

##@ Certification
cert-loop:  ## Run automated certification loop (baseline + quant8)
	@echo "Running automated certification workflow..."
	@rm -rf runs/cert_loop reports/cert/cert_loop
	@INVARLOCK_ALLOW_NETWORK=1 INVARLOCK_DEDUP_TEXTS=1 invarlock certify \
		--baseline sshleifer/tiny-gpt2 --subject sshleifer/tiny-gpt2 --adapter auto \
		--profile ci --tier balanced \
		--preset configs/presets/causal_lm/wikitext2_512.yaml \
		--edit-config configs/overlays/edits/quant_rtn/8bit_attn.yaml \
		--out runs/cert_loop \
		--cert-out reports/cert/cert_loop
	@echo "Certification complete. Artifacts: runs/cert_loop/, reports/cert/cert_loop/"

##@ Utilities
ci-matrix:  ## Verify CI matrix
	bash scripts/verify_ci_matrix.sh

## (manual-tests target removed)


.PHONY: ensure-ruff
ensure-ruff:
	@python -c "import importlib.util, subprocess, sys; spec = importlib.util.find_spec('ruff'); \
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ruff>=0.1.0']) if spec is None else None"

## (verify-ci and verify-release targets removed)

.PHONY: docs-check docs-lint docs-check-build docs-check-links docs-lint-markdown docs-lint-spell
docs-check: ## Run consolidated docs validation (build, links, refs, examples, consistency)
	python scripts/docs_check.py --all

docs-check-build: ## Build docs strictly and run link checks
	python scripts/docs_check.py --build --links

docs-check-links: ## Run docs link checks only
	python scripts/docs_check.py --links

docs-lint: ## Lint docs (markdown + spell)
	python scripts/docs_lint.py --all

docs-lint-markdown: ## Lint docs markdown style only
	python scripts/docs_lint.py --markdown

docs-lint-spell: ## Spell-check docs only
	python scripts/docs_lint.py --spell

.PHONY: config-check
config-check: ## Verify config includes and adapter availability
	python scripts/check_config_integrity.py configs

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
