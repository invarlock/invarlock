# InvarLock developer and release gates

PYTHON ?= $(shell bash scripts/select_workspace_python.sh)
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy
MKDOCS := $(PYTHON) -m mkdocs
PYTEST_WORKERS ?= 0
PYTEST_WORKER_ARGS := $(if $(filter-out 0,$(PYTEST_WORKERS)),-n $(PYTEST_WORKERS),)
CONTAINER_ENGINE ?= $(shell if command -v docker >/dev/null 2>&1; then echo docker; elif command -v podman >/dev/null 2>&1; then echo podman; fi)
RUNTIME_IMAGE ?= invarlock-runtime:local
RUNTIME_IMAGE_CUDA ?= invarlock-runtime:hf-cuda-local
RUNTIME_CUDA_DEVICE_ARGS = $(if $(filter podman,$(CONTAINER_ENGINE)),--device nvidia.com/gpu=all,--gpus all)
RUNTIME_SOURCE_DATE_EPOCH ?= $(shell git log -1 --pretty=%ct 2>/dev/null)
SECURITY_ARTIFACT_DIR ?= artifacts/supply-chain
SECURITY_RUN ?= uv run --isolated --locked --extra security-ci
DIST_RUN ?= uv run --isolated --locked --extra release-ci
RELEASE_PREFLIGHT_ARGS ?=

MYPY_TYPED_SURFACE := \
	src/invarlock/engine.py \
	src/invarlock/cli/app.py \
	src/invarlock/core/evaluation_request.py \
	src/invarlock/core/runtime_provider \
	src/invarlock/evaluation_run.py \
	src/invarlock/evaluation_runtime.py \
	src/invarlock/evaluation_transaction.py \
	src/invarlock/evidence_pack.py \
	src/invarlock/evidence_receipt.py \
	src/invarlock/evidence_reporting.py \
	src/invarlock/evidence_verification.py

.PHONY: help install dev-install lock-sync test test-fast test-parallel test-integration addins-test
.PHONY: coverage coverage-enforce coverage-enforce-parallel
.PHONY: trust-smoke mutation-smoke trust-boundary-demo
.PHONY: lint typecheck mypy-typed-surface format verify verify-fast verify-ruff
.PHONY: cli-smoke-core hf-provider-smoke local-hf-pipeline-smoke local-hf-pipeline-smoke-locked
.PHONY: actionlint workflow-lint docs docs-ci docs-serve docs-check docs-live-fast docs-live
.PHONY: docs-lint docs-lint-markdown docs-lint-spell docs-lint-strict docs-check-build docs-check-links
.PHONY: security supply-chain-security cve-audit dist-check addins-install-smoke packaging-smoke-minimal packaging-smoke-front-door
.PHONY: runtime-image runtime-image-podman runtime-image-cuda runtime-image-cuda-podman
.PHONY: runtime-smoke runtime-smoke-podman runtime-smoke-cuda runtime-smoke-cuda-podman container-front-door-smoke
.PHONY: release-preflight contracts-check contracts-sync repo-cruft-check public-evidence-audit public-evidence-sync
.PHONY: clean docsclean deepclean pre-commit pre-commit-install ensure-python ensure-ruff ensure-mypy

help:  ## Show maintained targets
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make <target>\n"} /^[a-zA-Z0-9_-]+:.*?##/ {printf "  %-28s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

##@ Install
install:  ## Install the core package
	$(MAKE) ensure-python
	$(PIP) install -e .

dev-install:  ## Install development dependencies
	$(MAKE) ensure-python
	$(PIP) install -e ".[dev]"

lock-sync:  ## Check that uv.lock matches pyproject.toml
	UV_NO_CACHE=1 uv lock --check

##@ Test
test:  ## Run the complete test suite
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTEST) $(PYTEST_WORKER_ARGS) -q tests

test-fast:  ## Run tests that need no network, GPU, or long-lived runtime
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTEST) $(PYTEST_WORKER_ARGS) -q \
		-m "not integration and not slow and not manual and not gpu" tests

test-parallel: PYTEST_WORKERS = auto
test-parallel:  ## Run the fast suite with pytest-xdist
	$(MAKE) test-fast PYTEST_WORKERS=$(PYTEST_WORKERS)

test-integration:  ## Run integration tests
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTEST) -q -m integration tests/integration

addins-test:  ## Test every first-party optional package
	PYTHONPATH=src:addins/diagnostics/src:addins/gguf/src:addins/multimodal/src:addins/tensorrt_llm/src \
		$(PYTEST) $(PYTEST_WORKER_ARGS) -q \
		addins/diagnostics/tests addins/gguf/tests addins/multimodal/tests addins/tensorrt_llm/tests

test-%:  ## Run one tests/<name> directory
	$(MAKE) ensure-python
	@test -d tests/$* || { echo "tests/$* does not exist" >&2; exit 2; }
	PYTHONPATH=src $(PYTEST) $(PYTEST_WORKER_ARGS) -q tests/$*

coverage:  ## Run the fast suite with branch coverage
	$(MAKE) ensure-python
	PYTHONPATH=src $(PYTEST) $(PYTEST_WORKER_ARGS) -q \
		-m "not integration and not slow and not manual and not gpu" tests \
		--cov=src/invarlock --cov-branch --cov-report=term-missing \
		--cov-report=xml:reports/cov.xml --cov-fail-under=90

coverage-enforce: PYTEST_WORKERS = auto
coverage-enforce:  ## Enforce branch coverage in parallel by default
	$(MAKE) coverage PYTEST_WORKERS=$(PYTEST_WORKERS)

coverage-enforce-parallel: PYTEST_WORKERS = auto
coverage-enforce-parallel:  ## Enforce coverage with pytest-xdist
	$(MAKE) coverage PYTEST_WORKERS=$(PYTEST_WORKERS)

trust-smoke:  ## Exercise pack tamper rejection and signed receipt verification
	PYTHONPATH=src $(PYTEST) -q tests/evidence_packs

mutation-smoke: trust-smoke  ## CI alias for the trust-critical adversarial smoke

trust-boundary-demo:  ## Run the isolated evidence-signing/verifier example transaction
	@if test -e examples/artifacts/trust-boundary-demo; then \
		chmod -R u+w examples/artifacts/trust-boundary-demo; \
	fi
	rm -rf examples/artifacts/trust-boundary-demo
	PYTHONPATH=src $(PYTHON) examples/run_trust_boundary_demo.py \
		--workspace examples/artifacts/trust-boundary-demo

##@ Static analysis
lint: verify-ruff typecheck  ## Run Ruff and mypy

verify-ruff:  ## Check Python lint and formatting
	$(MAKE) ensure-ruff
	$(RUFF) check src tests scripts addins
	$(RUFF) format --check src tests scripts addins

typecheck:  ## Type-check the core package
	$(MAKE) ensure-mypy
	$(MYPY) src/invarlock
	PYTHONPATH=src:addins/diagnostics/src $(MYPY) -p invarlock_addins.diagnostics
	PYTHONPATH=src:addins/gguf/src $(MYPY) -p invarlock_addins.gguf
	PYTHONPATH=src:addins/multimodal/src $(MYPY) -p invarlock_addins.multimodal
	PYTHONPATH=src:addins/tensorrt_llm/src $(MYPY) -p invarlock_addins.tensorrt_llm

mypy-typed-surface:  ## Type-check the public transaction and evidence surface
	$(MAKE) ensure-mypy
	PYTHONPATH=src $(MYPY) $(MYPY_TYPED_SURFACE)

format:  ## Format Python sources and tests
	$(MAKE) ensure-ruff
	$(RUFF) format src tests scripts addins
	$(RUFF) check --fix src tests scripts addins

##@ Product smoke
cli-smoke-core:  ## Check the evaluate, verify, and report command surface
	PYTHONPATH=src $(PYTHON) -m invarlock --help >/dev/null
	PYTHONPATH=src $(PYTHON) -m invarlock --version >/dev/null
	PYTHONPATH=src $(PYTHON) -m invarlock evaluate --help >/dev/null
	PYTHONPATH=src $(PYTHON) -m invarlock verify --help >/dev/null
	PYTHONPATH=src $(PYTHON) -m invarlock report --help >/dev/null

hf-provider-smoke:  ## Exercise the canonical built-in Hugging Face provider
	INVARLOCK_ALLOW_NETWORK=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
		PYTHONPATH=src $(PYTEST) -q tests/runtime_providers/test_hf_transformers.py \
		tests/runtime_providers/test_hf_transformers_strict.py \
		tests/cli/test_import_journey.py

local-hf-pipeline-smoke: hf-provider-smoke  ## CI alias for the built-in provider smoke

local-hf-pipeline-smoke-locked:  ## Run the built-in provider smoke in the locked environment
	uv run --isolated --locked --extra hf --extra ci $(MAKE) hf-provider-smoke

container-front-door-smoke: runtime-image  ## Run the host-to-container evaluation smoke
	INVARLOCK_RUN_CONTAINER_SMOKE=1 INVARLOCK_CONTAINER_ENGINE=$(CONTAINER_ENGINE) \
		INVARLOCK_RUNTIME_IMAGE=$(RUNTIME_IMAGE) \
		PYTHONPATH=src $(PYTEST) -q -m integration tests/integration/test_container_front_door_journey.py

##@ Verification
verify: PYTEST_WORKERS = auto
verify:  ## Run repository, product, docs, and contract gates in parallel by default
	$(MAKE) repo-cruft-check
	$(MAKE) public-evidence-audit
	$(MAKE) contracts-check
	$(MAKE) test PYTEST_WORKERS=$(PYTEST_WORKERS)
	$(MAKE) addins-test PYTEST_WORKERS=$(PYTEST_WORKERS)
	$(MAKE) cli-smoke-core
	$(MAKE) lint
	$(MAKE) docs-check-build

verify-fast: PYTEST_WORKERS = auto
verify-fast:  ## Run local gates in parallel without network, GPU, or downloads
	$(MAKE) repo-cruft-check
	$(MAKE) public-evidence-audit
	$(MAKE) contracts-check
	$(MAKE) test-fast PYTEST_WORKERS=$(PYTEST_WORKERS)
	$(MAKE) addins-test PYTEST_WORKERS=$(PYTEST_WORKERS)
	$(MAKE) cli-smoke-core
	$(MAKE) lint

contracts-check:  ## Check that packaged contracts match repository contracts
	PYTHONPATH=src $(PYTHON) scripts/checks/sync_packaged_contracts.py --check

contracts-sync:  ## Refresh packaged contracts
	PYTHONPATH=src $(PYTHON) scripts/checks/sync_packaged_contracts.py --write

repo-cruft-check:  ## Reject transport and generated artifacts in source paths
	$(PYTHON) scripts/checks/check_repo_cruft.py

public-evidence-audit:  ## Validate the canonical public evidence index
	PYTHONPATH=src $(PYTHON) scripts/checks/check_public_evidence.py
	PYTHONPATH=src $(PYTHON) scripts/checks/sync_packaged_public_evidence.py --check

public-evidence-sync:  ## Refresh the packaged public evidence index
	PYTHONPATH=src $(PYTHON) scripts/checks/sync_packaged_public_evidence.py --write

##@ Documentation
docs:  ## Build documentation strictly
	$(MKDOCS) build --strict

docs-ci: docs-check-build  ## Run documentation CI

docs-serve:  ## Serve documentation locally
	$(MKDOCS) serve -a 127.0.0.1:8000

docs-check: docs-check-build  ## Run documentation validation

docs-live-fast: cli-smoke-core docs-check-build  ## Check documented command surface and docs build

docs-live: docs-live-fast  ## Run the maintained documentation checks

docs-check-build: docs-lint-strict  ## Lint and build documentation
	$(MKDOCS) build --strict

docs-check-links: docs-check-build  ## Link checking is part of the strict MkDocs build

docs-lint: docs-lint-markdown docs-lint-spell  ## Run established documentation linters

docs-lint-strict: docs-lint  ## Strict documentation lint alias

docs-lint-markdown:  ## Run markdownlint-cli2
	npx --no-install markdownlint-cli2 README.md CODE_OF_CONDUCT.md \
		CONTRIBUTING.md SECURITY.md SUPPORT.md THIRD_PARTY_NOTICES.md \
		".github/**/*.md" \
		"docs/**/*.md" "scripts/**/*.md" "public_evidence/**/*.md" \
		"examples/**/*.md" "requirements/**/*.md" "tests/README.md" \
		"addins/**/*.md"

docs-lint-spell:  ## Run cspell
	npx --no-install cspell --no-progress README.md CODE_OF_CONDUCT.md \
		CONTRIBUTING.md SECURITY.md SUPPORT.md THIRD_PARTY_NOTICES.md \
		".github/**/*.md" \
		"docs/**/*.md" "scripts/**/*.md" "public_evidence/**/*.md" \
		"examples/**/*.md" "requirements/**/*.md" "tests/README.md" \
		"addins/**/*.md"

##@ Packaging and security
actionlint:  ## Lint GitHub Actions workflows
	@command -v actionlint >/dev/null 2>&1 || { echo "actionlint is required" >&2; exit 1; }
	actionlint .github/workflows/*.yml

workflow-lint: actionlint  ## Run workflow linting

security: supply-chain-security cve-audit  ## Run supply-chain security gates

supply-chain-security:  ## Generate an SBOM and audit the isolated tool environment
	@command -v uv >/dev/null 2>&1 || { echo "uv is required" >&2; exit 1; }
	$(SECURITY_RUN) bash -c 'scripts/security/generate_sbom.sh --scope tool-environment --python "$$(command -v python)" "$(SECURITY_ARTIFACT_DIR)/sbom.json"'
	$(SECURITY_RUN) python scripts/security/run_pip_audit.py

cve-audit:  ## Audit locked dependencies against OSV
	@command -v uv >/dev/null 2>&1 || { echo "uv is required" >&2; exit 1; }
	$(SECURITY_RUN) python scripts/security/cve_audit.py \
		--out-json "$(SECURITY_ARTIFACT_DIR)/cve-audit.json" \
		--out-md "$(SECURITY_ARTIFACT_DIR)/cve-audit.md"

dist-check:  ## Build and validate the core and first-party add-in distributions
	rm -rf build dist addins/*/build addins/*/src/*.egg-info
	$(DIST_RUN) python -m build --no-isolation
	$(DIST_RUN) python -m build --no-isolation --outdir dist/addins addins/diagnostics
	$(DIST_RUN) python -m build --no-isolation --outdir dist/addins addins/gguf
	$(DIST_RUN) python -m build --no-isolation --outdir dist/addins addins/multimodal
	$(DIST_RUN) python -m build --no-isolation --outdir dist/addins addins/tensorrt_llm
	$(DIST_RUN) python -m twine check dist/*.whl dist/*.tar.gz
	$(DIST_RUN) python -m twine check dist/addins/*

addins-install-smoke: dist-check  ## Install and discover every add-in from built wheels
	rm -rf .addins-smoke-site
	uv pip install --no-deps --target .addins-smoke-site dist/*.whl dist/addins/*.whl
	PYTHONPATH=.addins-smoke-site $(PYTHON) -m invarlock_addins.gguf.conformance
	PYTHONPATH=.addins-smoke-site $(PYTHON) -m invarlock_addins.multimodal.conformance
	PYTHONPATH=.addins-smoke-site $(PYTHON) -m invarlock_addins.tensorrt_llm.conformance
	PYTHONPATH=.addins-smoke-site $(PYTHON) -c "from pathlib import Path; import invarlock; assert Path(invarlock.__file__).resolve().is_relative_to(Path('.addins-smoke-site').resolve())"
	PYTHONPATH=.addins-smoke-site $(PYTHON) -c "from invarlock_addins.diagnostics import spectral_observation; assert spectral_observation([[1.0]])['status'] == 'observation'"
	PYTHONPATH=.addins-smoke-site $(PYTHON) -c "from importlib.metadata import entry_points; assert {'hf_vision_text', 'llama_cpp', 'tensorrt_llm'} <= {item.name for item in entry_points(group='invarlock.runtime_providers')}"

packaging-smoke-minimal: addins-install-smoke  ## Validate distributable artifacts

packaging-smoke-front-door: addins-install-smoke cli-smoke-core  ## Validate artifacts and CLI entry point

release-preflight:  ## Validate a clean exact release checkout and distributions
	@test -n "$(RELEASE_PREFLIGHT_ARGS)" || { echo "RELEASE_PREFLIGHT_ARGS is required" >&2; exit 2; }
	$(PYTHON) scripts/release/release_preflight.py $(RELEASE_PREFLIGHT_ARGS)

##@ Runtime image
runtime-image:  ## Build the canonical Hugging Face runtime image
	@test -n "$(CONTAINER_ENGINE)" || { echo "Docker or Podman is required" >&2; exit 1; }
	@test -n "$(RUNTIME_SOURCE_DATE_EPOCH)" || { echo "RUNTIME_SOURCE_DATE_EPOCH is required" >&2; exit 1; }
	SOURCE_DATE_EPOCH=$(RUNTIME_SOURCE_DATE_EPOCH) $(CONTAINER_ENGINE) build \
		--build-arg SOURCE_DATE_EPOCH=$(RUNTIME_SOURCE_DATE_EPOCH) \
		-f runtime/Dockerfile -t $(RUNTIME_IMAGE) .

runtime-image-podman: CONTAINER_ENGINE=podman
runtime-image-podman: runtime-image  ## Build the runtime image with Podman

runtime-image-cuda:  ## Build the x86_64 CUDA Hugging Face runtime image
	@test -n "$(CONTAINER_ENGINE)" || { echo "Docker or Podman is required" >&2; exit 1; }
	@test -n "$(RUNTIME_SOURCE_DATE_EPOCH)" || { echo "RUNTIME_SOURCE_DATE_EPOCH is required" >&2; exit 1; }
	SOURCE_DATE_EPOCH=$(RUNTIME_SOURCE_DATE_EPOCH) $(CONTAINER_ENGINE) build \
		--platform linux/amd64 \
		--build-arg SOURCE_DATE_EPOCH=$(RUNTIME_SOURCE_DATE_EPOCH) \
		-f runtime/Dockerfile.cuda -t $(RUNTIME_IMAGE_CUDA) .

runtime-image-cuda-podman: CONTAINER_ENGINE=podman
runtime-image-cuda-podman: runtime-image-cuda  ## Build the CUDA runtime image with Podman

runtime-smoke:  ## Check the canonical runtime image imports
	$(CONTAINER_ENGINE) run --rm --network none --entrypoint python $(RUNTIME_IMAGE) \
		-c "import torch, transformers, safetensors; print('runtime image imports ok')"

runtime-smoke-podman: CONTAINER_ENGINE=podman
runtime-smoke-podman: runtime-smoke  ## Smoke the runtime image with Podman

runtime-smoke-cuda:  ## Confirm the CUDA runtime imports and sees an NVIDIA GPU
	$(CONTAINER_ENGINE) run --rm --network none $(RUNTIME_CUDA_DEVICE_ARGS) \
		--entrypoint python $(RUNTIME_IMAGE_CUDA) \
		-c "import torch, transformers, safetensors; assert torch.version.cuda == '12.8'; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

runtime-smoke-cuda-podman: CONTAINER_ENGINE=podman
runtime-smoke-cuda-podman: runtime-smoke-cuda  ## Smoke the CUDA runtime image with Podman

##@ Housekeeping
pre-commit:  ## Run configured pre-commit hooks
	$(PYTHON) -m pre_commit run --all-files --show-diff-on-failure

pre-commit-install:  ## Install configured pre-commit hooks
	$(PYTHON) -m pre_commit install

clean:  ## Remove build and Python cache artifacts
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache .addins-smoke-site .addins-smoke-venv
	rm -rf addins/*/build addins/*/.pytest_cache addins/*/.mypy_cache addins/*/src/*.egg-info
	find . -type d -name __pycache__ ! -path './.git/*' -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '.DS_Store' -o -name '._*' \) ! -path './.git/*' -delete

docsclean:  ## Remove the rendered documentation site
	rm -rf site

deepclean: clean docsclean  ## Remove generated reports and local run outputs
	rm -rf reports runs artifacts .coverage coverage.xml htmlcov

ensure-python:
	@$(PYTHON) -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" || { echo "Python 3.12+ required" >&2; exit 1; }

ensure-ruff: ensure-python
	@$(PYTHON) -c "import ruff" 2>/dev/null || { echo "ruff is required" >&2; exit 1; }

ensure-mypy: ensure-python
	@$(PYTHON) -c "import mypy" 2>/dev/null || { echo "mypy is required" >&2; exit 1; }
