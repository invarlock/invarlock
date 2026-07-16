REPOSITORY_ROOT := $(abspath $(CURDIR)/../..)
PYTHON ?= $(REPOSITORY_ROOT)/.venv/bin/python
CONTAINER_ENGINE ?= docker
IMAGE ?= invarlock-runtime:gguf-local
IMAGE_DIGEST ?= $(shell $(CONTAINER_ENGINE) image inspect --format '{{.Id}}' $(IMAGE) 2>/dev/null)
SOURCE_DATE_EPOCH ?= $(shell git -C $(REPOSITORY_ROOT) log -1 --pretty=%ct)
LLAMA_CPP_APT_SNAPSHOT ?=
LLAMA_CPP_BUILD_JOBS ?= 4

.PHONY: build smoke conformance qualify-evidence

build:
	@test -n "$(LLAMA_CPP_APT_SNAPSHOT)" || { echo "LLAMA_CPP_APT_SNAPSHOT is required" >&2; exit 2; }
	cd $(REPOSITORY_ROOT) && $(CONTAINER_ENGINE) build \
		--file addins/gguf/runtime/Dockerfile \
		--build-arg SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) \
		--build-arg LLAMA_CPP_APT_SNAPSHOT=$(LLAMA_CPP_APT_SNAPSHOT) \
		--build-arg LLAMA_CPP_BUILD_JOBS=$(LLAMA_CPP_BUILD_JOBS) \
		--tag $(IMAGE) .

smoke:
	$(CONTAINER_ENGINE) run --rm --network none --entrypoint python $(IMAGE) \
		-c "import shutil; from invarlock_addins.gguf.conformance import main; assert shutil.which('llama-completion'); raise SystemExit(main([]))"

conformance:
	PYTHONPATH=$(REPOSITORY_ROOT)/src:$(REPOSITORY_ROOT)/addins/gguf/src \
		$(PYTHON) -m invarlock_addins.gguf.conformance

qualify-evidence:
	@for variable in REQUEST SIGNING_KEY EVIDENCE POLICY BASELINE_ARTIFACT SUBJECT_ARTIFACT SCHEDULE SIGNER VERIFIER_KEY VERIFIER_IDENTITY RECEIPT; do \
		eval "value=\$$$${variable}"; \
		test -n "$$value" || { echo "$$variable is required" >&2; exit 2; }; \
	done
	@test -n "$(IMAGE_DIGEST)" || { echo "IMAGE_DIGEST is required" >&2; exit 2; }
	cd $(REPOSITORY_ROOT) && PYTHONPATH=src:addins/gguf/src $(PYTHON) -m invarlock evaluate $(REQUEST) \
		--signing-key $(SIGNING_KEY) \
		--runtime-image $(IMAGE) \
		--runtime-image-digest $(IMAGE_DIGEST) \
		--container-engine $(CONTAINER_ENGINE)
	cd $(REPOSITORY_ROOT) && PYTHONPATH=src:addins/gguf/src $(PYTHON) -m invarlock verify $(EVIDENCE) \
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
