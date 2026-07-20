# InvarLock developer and release gates

PYTHON ?= $(shell bash scripts/select_workspace_python.sh)
QUALIFICATION_DRIVER_PYTHON ?= $(PYTHON)
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
RUNTIME_SOURCE_COMMIT ?= $(shell git rev-parse HEAD 2>/dev/null)
RUNTIME_SOURCE_BUNDLE ?=
RUNTIME_SOURCE_BUNDLE_SHA256 ?=
RUNTIME_BUILD_STATEMENT ?=
SOURCE_BUNDLE_OUTPUT ?=
QUALIFICATION_DEVICE ?=
SECURITY_ARTIFACT_DIR ?= artifacts/supply-chain
SECURITY_RUN ?= uv run --isolated --locked --extra security-ci
DIST_RUN ?= uv run --isolated --locked --extra release-ci
RELEASE_PREFLIGHT_ARGS ?=
EXAMPLE_ARGS ?=
ADDINS_SMOKE_PYTHON_TAG := $(shell $(PYTHON) -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')
ADDINS_SMOKE_RELEASE_LOCK ?= requirements/workflows/release-install-py$(ADDINS_SMOKE_PYTHON_TAG).txt

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
.PHONY: coverage coverage-addins coverage-qualification coverage-release coverage-examples coverage-enforce coverage-enforce-parallel
.PHONY: trust-smoke mutation-smoke trust-boundary-demo example-evidence-handoff example-hf-transformers example-peft-lora
.PHONY: example-torchao-int8 example-gguf-llama-cpp example-lm-evaluation-harness example-tensorrt-llm example-tensorrt-llm-prepared
.PHONY: lint typecheck mypy-typed-surface format verify verify-fast verify-ruff
.PHONY: cli-smoke-core hf-provider-smoke local-hf-pipeline-smoke local-hf-pipeline-smoke-locked
.PHONY: actionlint workflow-lint docs docs-ci docs-serve docs-check docs-live-fast docs-live
.PHONY: docs-lint docs-lint-markdown docs-lint-spell docs-lint-strict docs-check-build docs-check-links
.PHONY: security supply-chain-security cve-audit dist-check addins-install-smoke packaging-smoke-minimal packaging-smoke-front-door
.PHONY: runtime-image runtime-image-podman runtime-image-cuda runtime-image-cuda-podman
.PHONY: runtime-smoke runtime-smoke-podman runtime-smoke-cuda runtime-smoke-cuda-podman container-front-door-smoke
.PHONY: qualification-source-bundle runtime-qualification-canary runtime-qualification-readiness runtime-qualification-evidence
.PHONY: release-preflight contracts-check contracts-sync repo-cruft-check public-evidence-audit public-evidence-sync examples-check
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
	@git ls-files 'src/invarlock/**/*.py' 'src/invarlock/*.py' | \
		grep -v '/__init__.py$$' | \
		while IFS= read -r source; do \
			$(PYTHON) -m coverage report --include="$$source" --fail-under=80 || exit $$?; \
		done

coverage-linux-check:
	@test "$$(uname -s)" = Linux || { \
		echo "the complete add-in coverage gate requires Linux descriptor execution" >&2; \
		exit 2; \
	}

coverage-addins: coverage-linux-check  ## Enforce the branch-coverage ratchet for optional packages
	PYTHONPATH=src:addins/diagnostics/src:addins/gguf/src:addins/multimodal/src:addins/tensorrt_llm/src \
		$(PYTEST) $(PYTEST_WORKER_ARGS) -q \
		addins/diagnostics/tests addins/gguf/tests addins/multimodal/tests addins/tensorrt_llm/tests \
		--cov=addins/diagnostics/src/invarlock_addins/diagnostics \
		--cov=addins/gguf/src/invarlock_addins/gguf \
		--cov=addins/multimodal/src/invarlock_addins/multimodal \
		--cov=addins/tensorrt_llm/src/invarlock_addins/tensorrt_llm \
		--cov-branch --cov-report=term-missing \
		--cov-report=xml:reports/addins-cov.xml \
		--cov-fail-under=80
	$(PYTHON) -m coverage report \
		--include='addins/diagnostics/src/*' \
		--fail-under=80
	$(PYTHON) -m coverage report \
		--include='addins/gguf/src/*' \
		--fail-under=80
	$(PYTHON) -m coverage report \
		--include='addins/multimodal/src/*' \
		--fail-under=80
	$(PYTHON) -m coverage report \
		--include='addins/tensorrt_llm/src/*' \
		--fail-under=80
	@git ls-files 'addins/*/src/**/*.py' | \
		grep -v '/__init__.py$$' | \
		while IFS= read -r source; do \
			$(PYTHON) -m coverage report --include="$$source" --fail-under=80 || exit $$?; \
		done

coverage-qualification:  ## Enforce branch coverage for the maintained qualification transaction
	PYTHONPATH=src:addins/tensorrt_llm/src $(PYTEST) $(PYTEST_WORKER_ARGS) -q \
		tests/scripts/test_runtime_qualification.py \
		tests/scripts/test_runtime_qualification_security.py \
		tests/ci/test_qualification_precheck.py \
		tests/scripts/test_qualification_candidate_wheels.py \
		tests/scripts/test_qualification_receipt_check.py \
		tests/scripts/test_qualification_render_preflight.py \
		tests/scripts/test_qualification_source.py \
		tests/scripts/test_authenticated_runtime_build.py \
		addins/tensorrt_llm/tests/test_tensorrt_llm_canary_preflight.py \
		--cov --cov-config=scripts/qualification.coveragerc --cov-branch \
		--cov-report=term-missing \
		--cov-report=xml:reports/qualification-cov.xml \
		--cov-fail-under=80
	$(PYTHON) -m coverage report --rcfile=scripts/qualification.coveragerc \
		--include='scripts/authenticated_runtime_build.py' --fail-under=80
	$(PYTHON) -m coverage report --rcfile=scripts/qualification.coveragerc \
		--include='scripts/qualification_precheck.py' --fail-under=80
	$(PYTHON) -m coverage report --rcfile=scripts/qualification.coveragerc \
		--include='scripts/qualification_candidate_wheels.py' --fail-under=80
	$(PYTHON) -m coverage report --rcfile=scripts/qualification.coveragerc \
		--include='scripts/qualification_receipt_check.py' --fail-under=80
	$(PYTHON) -m coverage report --rcfile=scripts/qualification.coveragerc \
		--include='scripts/qualification_render_preflight.py' --fail-under=80
	$(PYTHON) -m coverage report --rcfile=scripts/qualification.coveragerc \
		--include='scripts/qualification_source.py' --fail-under=80
	$(PYTHON) -m coverage report --rcfile=scripts/qualification.coveragerc \
		--include='scripts/runtime_qualification.py' --fail-under=80
	$(PYTHON) -m coverage report --rcfile=scripts/qualification.coveragerc \
		--include='scripts/tensorrt_llm_canary_preflight.py' --fail-under=80

coverage-release:  ## Enforce branch coverage for maintained release helpers
	PYTHONPATH=src $(PYTEST) $(PYTEST_WORKER_ARGS) -q \
		tests/scripts/test_first_party_distribution_validation.py \
		tests/scripts/test_release_preflight.py \
		tests/scripts/test_release_preflight_adversarial.py \
		tests/scripts/test_release_preflight_edges.py \
		tests/scripts/test_verify_hosted_distributions.py \
		tests/scripts/test_testpypi_promotion.py \
		--cov --cov-config=scripts/release.coveragerc --cov-branch \
		--cov-report=term-missing \
		--cov-report=xml:reports/release-cov.xml \
		--cov-fail-under=80
	$(PYTHON) -m coverage report --rcfile=scripts/release.coveragerc \
		--include='scripts/release/first_party_distribution_validation.py' --fail-under=80
	$(PYTHON) -m coverage report --rcfile=scripts/release.coveragerc \
		--include='scripts/release/release_distribution_validation.py' --fail-under=80
	$(PYTHON) -m coverage report --rcfile=scripts/release.coveragerc \
		--include='scripts/release/release_preflight.py' --fail-under=80
	$(PYTHON) -m coverage report --rcfile=scripts/release.coveragerc \
		--include='scripts/release/testpypi_promotion.py' --fail-under=80
	$(PYTHON) -m coverage report --rcfile=scripts/release.coveragerc \
		--include='scripts/release/verify_hosted_distributions.py' --fail-under=80

coverage-examples:  ## Enforce branch coverage for maintained example launchers
	PYTHONPATH=src:. $(PYTEST) $(PYTEST_WORKER_ARGS) -q \
		tests/examples \
		--cov=examples.integrations.launch \
		--cov=examples.integrations.run \
		--cov=examples.integrations.gguf_llama_cpp \
		--cov=examples.integrations.qwen3_profile \
		--cov=examples/integrations/lm-evaluation-harness \
		--cov=examples/integrations/tensorrt-llm \
		--cov-branch \
		--cov-report=term-missing \
		--cov-report=xml:reports/examples-cov.xml \
		--cov-fail-under=80
	@find examples/integrations -type f -name '*.py' \
		! -name '__init__.py' | sort | \
		while IFS= read -r source; do \
			$(PYTHON) -m coverage report --include="$$source" --fail-under=80 || exit $$?; \
		done

coverage-enforce: PYTEST_WORKERS = auto
coverage-enforce: coverage-linux-check  ## Enforce branch coverage in parallel by default
	$(MAKE) coverage PYTEST_WORKERS=$(PYTEST_WORKERS)
	$(MAKE) coverage-addins PYTEST_WORKERS=$(PYTEST_WORKERS)
	$(MAKE) coverage-qualification PYTEST_WORKERS=$(PYTEST_WORKERS)
	$(MAKE) coverage-release PYTEST_WORKERS=$(PYTEST_WORKERS)
	$(MAKE) coverage-examples PYTEST_WORKERS=$(PYTEST_WORKERS)

coverage-enforce-parallel: PYTEST_WORKERS = auto
coverage-enforce-parallel:  ## Enforce coverage with pytest-xdist
	$(MAKE) coverage-enforce PYTEST_WORKERS=$(PYTEST_WORKERS)

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

example-evidence-handoff: trust-boundary-demo  ## Run signed acceptance, rejection, and tamper handoff

example-hf-transformers:  ## Run a real one-command Hugging Face comparison
	PYTHONPATH=src uv run --isolated --locked --extra hf python \
		examples/integrations/launch.py hf-transformers $(EXAMPLE_ARGS)

example-peft-lora:  ## Train and merge with PEFT, then evaluate, verify, and report
	PYTHONPATH=src uv run --isolated --locked --extra hf --group example-peft python \
		examples/integrations/launch.py peft-lora $(EXAMPLE_ARGS)

example-torchao-int8:  ## Quantize with TorchAO, then evaluate, verify, and report
	PYTHONPATH=src uv run --isolated --locked --extra hf --group example-torchao python \
		examples/integrations/launch.py torchao-int8 $(EXAMPLE_ARGS)

example-gguf-llama-cpp:  ## Compare two pinned GGUF quantizations with llama.cpp
	PYTHONPATH=src:addins/gguf/src uv run --isolated --locked --with . \
		--with ./addins/gguf python -m examples.integrations.gguf_llama_cpp $(EXAMPLE_ARGS)

example-lm-evaluation-harness:  ## Import real per-record LM Evaluation Harness output
	PYTHONPATH=src:. uv run --isolated --locked --extra hf python \
		examples/integrations/lm-evaluation-harness/launch.py $(EXAMPLE_ARGS)

example-tensorrt-llm:  ## Compare BF16 and calibrated FP8 Qwen3 TensorRT-LLM engines
	PYTHONPATH=src:addins/tensorrt_llm/src:. uv run --isolated --locked --extra hf \
		--with . --with ./addins/tensorrt_llm python \
		examples/integrations/tensorrt-llm/showcase.py $(EXAMPLE_ARGS)

example-tensorrt-llm-prepared:  ## Compare caller-prepared TensorRT-LLM engines
	@test -n "$(EXAMPLE_ARGS)" || { \
		echo 'set EXAMPLE_ARGS to the immutable image, prepared inputs, and locators' >&2; \
		exit 2; \
	}
	PYTHONPATH=src:addins/tensorrt_llm/src uv run --isolated --locked --with . \
		--with ./addins/tensorrt_llm python \
		examples/integrations/tensorrt-llm/run.py $(EXAMPLE_ARGS)

##@ Static analysis
lint: verify-ruff typecheck  ## Run Ruff and mypy

verify-ruff:  ## Check Python lint and formatting
	$(MAKE) ensure-ruff
	$(RUFF) check src tests scripts addins examples
	$(RUFF) format --check src tests scripts addins examples

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
	$(RUFF) format src tests scripts addins examples
	$(RUFF) check --fix src tests scripts addins examples

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

qualification-source-bundle:  ## Create the exact Git archive used by runtime qualification
	$(foreach variable,SOURCE_BUNDLE_OUTPUT,$(if $(strip $($(variable))),,$(error $(variable) is required)))
	"$(PYTHON)" scripts/qualification_source.py create \
		--repository "$(CURDIR)" \
		--commit "$(RUNTIME_SOURCE_COMMIT)" \
		--output "$(SOURCE_BUNDLE_OUTPUT)"

runtime-qualification-canary:  ## Bootstrap one signed exact-image canary qualification
	$(foreach variable,REQUEST SIGNING_KEY IMAGE IMAGE_DIGEST EVIDENCE TRUST_PROFILE RECEIPT SUMMARY SOURCE_COMMIT SOURCE_BUNDLE SOURCE_BUNDLE_SHA256 CANDIDATE_WHEEL_MANIFEST QUALIFICATION_DEVICE,$(if $(strip $($(variable))),,$(error $(variable) is required)))
	"$(QUALIFICATION_DRIVER_PYTHON)" -I -S scripts/runtime_qualification.py canary \
		--python "$(PYTHON)" \
		--request "$(REQUEST)" \
		--signing-key "$(SIGNING_KEY)" \
		--runtime-image "$(IMAGE)" \
		--runtime-image-digest "$(IMAGE_DIGEST)" \
		--evidence "$(EVIDENCE)" \
		--trust-profile "$(TRUST_PROFILE)" \
		--receipt "$(RECEIPT)" \
		--summary "$(SUMMARY)" \
		--source-commit "$(SOURCE_COMMIT)" \
		--source-bundle "$(SOURCE_BUNDLE)" \
		--source-bundle-sha256 "$(SOURCE_BUNDLE_SHA256)" \
		--candidate-wheel-manifest "$(CANDIDATE_WHEEL_MANIFEST)" \
		--container-engine "$(CONTAINER_ENGINE)" \
		--runtime-device "$(QUALIFICATION_DEVICE)" \
		$(if $(strip $(QUALIFICATION_CPUS)),--runtime-cpus "$(QUALIFICATION_CPUS)") \
		$(if $(strip $(QUALIFICATION_MEMORY_MIB)),--runtime-memory-mib "$(QUALIFICATION_MEMORY_MIB)") \
		$(if $(strip $(QUALIFICATION_USER)),--runtime-user "$(QUALIFICATION_USER)") \
		$(if $(strip $(REPORT)),--report "$(REPORT)")

runtime-qualification-readiness:  ## Validate one frozen runtime qualification without execution
	$(foreach variable,REQUEST SIGNING_KEY IMAGE IMAGE_DIGEST EVIDENCE TRUST_PROFILE RECEIPT CANARY_EVIDENCE CANARY_RECEIPT CANARY_TRUST_PROFILE SOURCE_COMMIT SOURCE_BUNDLE SOURCE_BUNDLE_SHA256 CANDIDATE_WHEEL_MANIFEST QUALIFICATION_DEVICE,$(if $(strip $($(variable))),,$(error $(variable) is required)))
	"$(QUALIFICATION_DRIVER_PYTHON)" -I -S scripts/runtime_qualification.py readiness \
		--python "$(PYTHON)" \
		--request "$(REQUEST)" \
		--signing-key "$(SIGNING_KEY)" \
		--runtime-image "$(IMAGE)" \
		--runtime-image-digest "$(IMAGE_DIGEST)" \
		--evidence "$(EVIDENCE)" \
		--trust-profile "$(TRUST_PROFILE)" \
		--receipt "$(RECEIPT)" \
		--canary-evidence "$(CANARY_EVIDENCE)" \
		--canary-receipt "$(CANARY_RECEIPT)" \
		--canary-trust-profile "$(CANARY_TRUST_PROFILE)" \
		--source-commit "$(SOURCE_COMMIT)" \
		--source-bundle "$(SOURCE_BUNDLE)" \
		--source-bundle-sha256 "$(SOURCE_BUNDLE_SHA256)" \
		--candidate-wheel-manifest "$(CANDIDATE_WHEEL_MANIFEST)" \
		--container-engine "$(CONTAINER_ENGINE)" \
		--runtime-device "$(QUALIFICATION_DEVICE)" \
		$(if $(strip $(QUALIFICATION_CPUS)),--runtime-cpus "$(QUALIFICATION_CPUS)") \
		$(if $(strip $(QUALIFICATION_MEMORY_MIB)),--runtime-memory-mib "$(QUALIFICATION_MEMORY_MIB)") \
		$(if $(strip $(QUALIFICATION_USER)),--runtime-user "$(QUALIFICATION_USER)")

runtime-qualification-evidence:  ## Evaluate, verify, report, and summarize one frozen runtime qualification
	$(foreach variable,REQUEST SIGNING_KEY IMAGE IMAGE_DIGEST EVIDENCE TRUST_PROFILE RECEIPT CANARY_EVIDENCE CANARY_RECEIPT CANARY_TRUST_PROFILE SUMMARY SOURCE_COMMIT SOURCE_BUNDLE SOURCE_BUNDLE_SHA256 CANDIDATE_WHEEL_MANIFEST QUALIFICATION_DEVICE,$(if $(strip $($(variable))),,$(error $(variable) is required)))
	"$(QUALIFICATION_DRIVER_PYTHON)" -I -S scripts/runtime_qualification.py run \
		--python "$(PYTHON)" \
		--request "$(REQUEST)" \
		--signing-key "$(SIGNING_KEY)" \
		--runtime-image "$(IMAGE)" \
		--runtime-image-digest "$(IMAGE_DIGEST)" \
		--evidence "$(EVIDENCE)" \
		--trust-profile "$(TRUST_PROFILE)" \
		--receipt "$(RECEIPT)" \
		--canary-evidence "$(CANARY_EVIDENCE)" \
		--canary-receipt "$(CANARY_RECEIPT)" \
		--canary-trust-profile "$(CANARY_TRUST_PROFILE)" \
		--summary "$(SUMMARY)" \
		--source-commit "$(SOURCE_COMMIT)" \
		--source-bundle "$(SOURCE_BUNDLE)" \
		--source-bundle-sha256 "$(SOURCE_BUNDLE_SHA256)" \
		--candidate-wheel-manifest "$(CANDIDATE_WHEEL_MANIFEST)" \
		--container-engine "$(CONTAINER_ENGINE)" \
		--runtime-device "$(QUALIFICATION_DEVICE)" \
		$(if $(strip $(QUALIFICATION_CPUS)),--runtime-cpus "$(QUALIFICATION_CPUS)") \
		$(if $(strip $(QUALIFICATION_MEMORY_MIB)),--runtime-memory-mib "$(QUALIFICATION_MEMORY_MIB)") \
		$(if $(strip $(QUALIFICATION_USER)),--runtime-user "$(QUALIFICATION_USER)") \
		$(if $(strip $(REPORT)),--report "$(REPORT)")

##@ Verification
verify: PYTEST_WORKERS = auto
verify:  ## Run repository, product, docs, and contract gates in parallel by default
	$(MAKE) repo-cruft-check
	$(MAKE) public-evidence-audit
	$(MAKE) examples-check
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
	$(MAKE) examples-check
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

examples-check:  ## Test the maintained one-command integration journeys
	PYTHONPATH=src:. $(PYTEST) $(PYTEST_WORKER_ARGS) -q tests/examples

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
	rm -rf build dist src/*.egg-info addins/*/build addins/*/src/*.egg-info
	$(DIST_RUN) python -m build --no-isolation
	$(DIST_RUN) python -m build --no-isolation --outdir dist/addins addins/diagnostics
	$(DIST_RUN) python -m build --no-isolation --outdir dist/addins addins/gguf
	$(DIST_RUN) python -m build --no-isolation --outdir dist/addins addins/multimodal
	$(DIST_RUN) python -m build --no-isolation --outdir dist/addins addins/tensorrt_llm
	$(DIST_RUN) python -m twine check dist/*.whl dist/*.tar.gz
	$(DIST_RUN) python -m twine check dist/addins/*
	$(DIST_RUN) python scripts/release/first_party_distribution_validation.py \
		--repo-root . --core-dist-dir dist --addin-dist-dir dist/addins

addins-install-smoke: dist-check  ## Install and discover all five wheels in a disposable environment
	@test -f $(ADDINS_SMOKE_RELEASE_LOCK) || { echo "No coordinated release lock for $(PYTHON); expected $(ADDINS_SMOKE_RELEASE_LOCK)" >&2; exit 2; }
	@set -eu; \
		smoke_venv="$$(mktemp -d "$${TMPDIR:-/tmp}/invarlock-addins-smoke.XXXXXX")"; \
		cleanup_smoke_venv() { rm -rf "$$smoke_venv"; }; \
		trap cleanup_smoke_venv EXIT; \
		trap 'exit 129' HUP; \
		trap 'exit 130' INT; \
		trap 'exit 143' TERM; \
		$(PYTHON) -m venv "$$smoke_venv"; \
		"$$smoke_venv/bin/python" -m pip install --require-hashes -r requirements/workflows/pip-bootstrap.txt; \
		"$$smoke_venv/bin/python" -m pip install --require-hashes -r $(ADDINS_SMOKE_RELEASE_LOCK); \
		PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH= "$$smoke_venv/bin/python" -m pip install --no-deps --force-reinstall dist/*.whl dist/addins/*.whl; \
		"$$smoke_venv/bin/python" -m pip check; \
		PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH= "$$smoke_venv/bin/python" -m invarlock_addins.gguf.conformance; \
		PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH= "$$smoke_venv/bin/python" -m invarlock_addins.multimodal.conformance; \
		PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH= "$$smoke_venv/bin/python" -m invarlock_addins.tensorrt_llm.conformance; \
		PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH= "$$smoke_venv/bin/python" -c "from pathlib import Path; import invarlock; import sysconfig; site = Path(sysconfig.get_path('purelib')).resolve(); assert Path(invarlock.__file__).resolve().is_relative_to(site)"; \
		PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH= "$$smoke_venv/bin/python" -c "from invarlock_addins.diagnostics import spectral_observation; assert spectral_observation([[1.0]])['status'] == 'observation'"; \
		PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH= "$$smoke_venv/bin/python" -c "from importlib.metadata import entry_points; assert {'hf_vision_text', 'llama_cpp', 'tensorrt_llm'} <= {item.name for item in entry_points(group='invarlock.runtime_providers')}"; \
		PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH= "$$smoke_venv/bin/python" -c "from importlib import import_module; from pathlib import Path; import sysconfig; from invarlock import __version__; from invarlock.core.registry import CoreRegistry; from invarlock.core.runtime_provider import INVARLOCK_RUNTIME_PROVIDER_ABI; registry = CoreRegistry(); expected = {'hf_vision_text': 'invarlock-runtime-hf-vision-text', 'llama_cpp': 'invarlock-runtime-gguf', 'tensorrt_llm': 'invarlock-runtime-tensorrt-llm'}; providers = {name: registry.get_runtime_provider(name) for name in expected}; assert all(provider.name == name and provider.abi_version == INVARLOCK_RUNTIME_PROVIDER_ABI for name, provider in providers.items()); assert all(registry.get_plugin_info(name, 'runtime_providers')['package'] == package and registry.get_plugin_info(name, 'runtime_providers')['version'] == __version__ and registry.get_plugin_info(name, 'runtime_providers')['entry_point'] == name for name, package in expected.items()); site = Path(sysconfig.get_path('purelib')).resolve(); assert all(Path(import_module(provider.__class__.__module__).__file__).resolve().is_relative_to(site) for provider in providers.values())"

packaging-smoke-minimal: addins-install-smoke  ## Validate distributable artifacts

packaging-smoke-front-door: addins-install-smoke cli-smoke-core  ## Validate artifacts and CLI entry point

release-preflight:  ## Validate a clean exact release checkout and distributions
	@test -n "$(RELEASE_PREFLIGHT_ARGS)" || { echo "RELEASE_PREFLIGHT_ARGS is required" >&2; exit 2; }
	$(PYTHON) scripts/release/release_preflight.py $(RELEASE_PREFLIGHT_ARGS)

##@ Runtime image
runtime-image:  ## Build the canonical Hugging Face runtime image
	@test -n "$(CONTAINER_ENGINE)" || { echo "Docker or Podman is required" >&2; exit 1; }
	@test -n "$(RUNTIME_SOURCE_DATE_EPOCH)" || { echo "RUNTIME_SOURCE_DATE_EPOCH is required" >&2; exit 1; }
	$(foreach variable,RUNTIME_SOURCE_COMMIT RUNTIME_SOURCE_BUNDLE RUNTIME_SOURCE_BUNDLE_SHA256,$(if $(strip $($(variable))),,$(error $(variable) is required)))
	"$(PYTHON)" scripts/authenticated_runtime_build.py \
		--repository "$(CURDIR)" \
		--source-commit "$(RUNTIME_SOURCE_COMMIT)" \
		--source-bundle "$(RUNTIME_SOURCE_BUNDLE)" \
		--source-bundle-sha256 "$(RUNTIME_SOURCE_BUNDLE_SHA256)" \
		--container-engine "$(CONTAINER_ENGINE)" \
		--dockerfile runtime/Dockerfile \
		--image "$(RUNTIME_IMAGE)" \
		--build-arg "SOURCE_DATE_EPOCH=$(RUNTIME_SOURCE_DATE_EPOCH)" $(if $(strip $(RUNTIME_BUILD_STATEMENT)),--statement "$(RUNTIME_BUILD_STATEMENT)")

runtime-image-podman: CONTAINER_ENGINE=podman
runtime-image-podman: runtime-image  ## Build the runtime image with Podman

runtime-image-cuda:  ## Build the x86_64 CUDA Hugging Face runtime image
	@test -n "$(CONTAINER_ENGINE)" || { echo "Docker or Podman is required" >&2; exit 1; }
	@test -n "$(RUNTIME_SOURCE_DATE_EPOCH)" || { echo "RUNTIME_SOURCE_DATE_EPOCH is required" >&2; exit 1; }
	$(foreach variable,RUNTIME_SOURCE_COMMIT RUNTIME_SOURCE_BUNDLE RUNTIME_SOURCE_BUNDLE_SHA256,$(if $(strip $($(variable))),,$(error $(variable) is required)))
	"$(PYTHON)" scripts/authenticated_runtime_build.py \
		--repository "$(CURDIR)" \
		--source-commit "$(RUNTIME_SOURCE_COMMIT)" \
		--source-bundle "$(RUNTIME_SOURCE_BUNDLE)" \
		--source-bundle-sha256 "$(RUNTIME_SOURCE_BUNDLE_SHA256)" \
		--container-engine "$(CONTAINER_ENGINE)" \
		--dockerfile runtime/Dockerfile.cuda \
		--image "$(RUNTIME_IMAGE_CUDA)" \
		--platform linux/amd64 \
		--build-arg "SOURCE_DATE_EPOCH=$(RUNTIME_SOURCE_DATE_EPOCH)" $(if $(strip $(RUNTIME_BUILD_STATEMENT)),--statement "$(RUNTIME_BUILD_STATEMENT)")

runtime-image-cuda-podman: CONTAINER_ENGINE=podman
runtime-image-cuda-podman: runtime-image-cuda  ## Build the CUDA runtime image with Podman

runtime-smoke:  ## Check the canonical runtime image imports
	$(CONTAINER_ENGINE) run --rm --network none \
		--pull=never --read-only --cap-drop=ALL \
		--security-opt no-new-privileges --pids-limit 1024 \
		--user 65532:65532 \
		--tmpfs "/tmp:rw,noexec,nosuid,nodev,size=4g" \
		--env HOME=/tmp --env PYTHONDONTWRITEBYTECODE=1 \
		--entrypoint python $(RUNTIME_IMAGE) \
		-c "import accelerate, safetensors, torch, transformers; assert accelerate.__version__ == '1.14.0'; assert safetensors.__version__ == '0.8.0'; assert transformers.__version__ == '5.14.1'; print('runtime image imports ok')"

runtime-smoke-podman: CONTAINER_ENGINE=podman
runtime-smoke-podman: runtime-smoke  ## Smoke the runtime image with Podman

runtime-smoke-cuda:  ## Confirm the CUDA runtime imports and sees an NVIDIA GPU
	$(CONTAINER_ENGINE) run --rm --network none $(RUNTIME_CUDA_DEVICE_ARGS) \
		--pull=never --read-only --cap-drop=ALL \
		--security-opt no-new-privileges --pids-limit 1024 \
		--user 65532:65532 \
		--tmpfs "/tmp:rw,noexec,nosuid,nodev,size=4g" \
		--env HOME=/tmp --env PYTHONDONTWRITEBYTECODE=1 \
		--entrypoint python $(RUNTIME_IMAGE_CUDA) \
		-c "import accelerate, safetensors, torch, transformers; assert accelerate.__version__ == '1.14.0'; assert safetensors.__version__ == '0.8.0'; assert transformers.__version__ == '5.14.1'; assert torch.version.cuda == '12.8'; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

runtime-smoke-cuda-podman: CONTAINER_ENGINE=podman
runtime-smoke-cuda-podman: runtime-smoke-cuda  ## Smoke the CUDA runtime image with Podman

##@ Housekeeping
pre-commit:  ## Run configured pre-commit hooks
	$(PYTHON) -m pre_commit run --all-files --show-diff-on-failure

pre-commit-install:  ## Install configured pre-commit hooks
	$(PYTHON) -m pre_commit install

clean:  ## Remove build and Python cache artifacts
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .mypy_cache .ruff_cache .addins-smoke-site .addins-smoke-venv
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
