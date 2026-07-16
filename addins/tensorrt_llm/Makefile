REPOSITORY_ROOT := $(abspath $(CURDIR)/../..)
PYTHON ?= $(REPOSITORY_ROOT)/.venv/bin/python
CONTAINER_ENGINE ?= docker
IMAGE ?= invarlock-runtime:tensorrt-llm-local
IMAGE_DIGEST ?= $(shell $(CONTAINER_ENGINE) image inspect --format '{{.Id}}' $(IMAGE) 2>/dev/null)
SOURCE_DATE_EPOCH ?= $(shell git -C $(REPOSITORY_ROOT) log -1 --pretty=%ct)

.PHONY: build smoke conformance canary qualify-evidence

build:
	cd $(REPOSITORY_ROOT) && $(CONTAINER_ENGINE) build \
		--file addins/tensorrt_llm/runtime/Dockerfile \
		--build-arg SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) \
		--tag $(IMAGE) .

smoke:
	$(CONTAINER_ENGINE) run --rm --network none --gpus all \
		--entrypoint /opt/invarlock/cli-venv/bin/python $(IMAGE) \
		-c "import torch; assert torch.cuda.is_available(); from invarlock_addins.tensorrt_llm.conformance import main; raise SystemExit(main([]))"

conformance:
	PYTHONPATH=$(REPOSITORY_ROOT)/src:$(REPOSITORY_ROOT)/addins/tensorrt_llm/src \
		$(PYTHON) -m invarlock_addins.tensorrt_llm.conformance

canary:
	@for variable in INPUT_ROOT ENGINE_BUNDLE TOKENIZER_CONTRACT EXPECTED_ENGINE_TREE_SHA256 EXPECTED_TOKENIZER_SHA256 EXPECTED_OUTPUT_SHA256; do \
		eval "value=\$$$${variable}"; \
		test -n "$$value" || { echo "$$variable is required" >&2; exit 2; }; \
	done
	@test -n "$(IMAGE_DIGEST)" || { echo "IMAGE_DIGEST is required" >&2; exit 2; }
	$(CONTAINER_ENGINE) run --rm --network none --gpus all \
		--env INVARLOCK_CONTAINER_EXECUTION=1 \
		--env INVARLOCK_RUNTIME_IMAGE=$(IMAGE_DIGEST) \
		--env INVARLOCK_RUNTIME_IMAGE_DIGEST=$(IMAGE_DIGEST) \
		--mount type=bind,src=$(INPUT_ROOT),dst=/inputs,readonly \
		--entrypoint /opt/invarlock/cli-venv/bin/python $(IMAGE) \
		-m invarlock_addins.tensorrt_llm.canary \
		--engine-bundle /inputs/$(ENGINE_BUNDLE) \
		--tokenizer-contract /inputs/$(TOKENIZER_CONTRACT) \
		--runner /opt/invarlock/bin/tensorrt-llm-runner \
		--expected-engine-tree-sha256 $(EXPECTED_ENGINE_TREE_SHA256) \
		--expected-tokenizer-sha256 $(EXPECTED_TOKENIZER_SHA256) \
		--expected-output-sha256 $(EXPECTED_OUTPUT_SHA256)

qualify-evidence:
	@for variable in REQUEST SIGNING_KEY EVIDENCE POLICY BASELINE_ARTIFACT SUBJECT_ARTIFACT SCHEDULE SIGNER VERIFIER_KEY VERIFIER_IDENTITY RECEIPT; do \
		eval "value=\$$$${variable}"; \
		test -n "$$value" || { echo "$$variable is required" >&2; exit 2; }; \
	done
	@test -n "$(IMAGE_DIGEST)" || { echo "IMAGE_DIGEST is required" >&2; exit 2; }
	cd $(REPOSITORY_ROOT) && PYTHONPATH=src:addins/tensorrt_llm/src $(PYTHON) -m invarlock evaluate $(REQUEST) \
		--signing-key $(SIGNING_KEY) \
		--runtime-image $(IMAGE) \
		--runtime-image-digest $(IMAGE_DIGEST) \
		--container-engine $(CONTAINER_ENGINE) \
		--runtime-device cuda
	cd $(REPOSITORY_ROOT) && PYTHONPATH=src:addins/tensorrt_llm/src $(PYTHON) -m invarlock verify $(EVIDENCE) \
		--policy $(POLICY) \
		--expected-baseline-artifact $(BASELINE_ARTIFACT) \
		--expected-subject-artifact $(SUBJECT_ARTIFACT) \
		--expected-schedule $(SCHEDULE) \
		--expected-baseline-runtime $(IMAGE_DIGEST) \
		--expected-subject-runtime $(IMAGE_DIGEST) \
		--expected-signer $(SIGNER) \
		--receipt $(RECEIPT) \
		--verifier-signing-key $(VERIFIER_KEY) \
		--verifier-identity $(VERIFIER_IDENTITY)
