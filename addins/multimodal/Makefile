REPOSITORY_ROOT := $(abspath $(CURDIR)/../..)
PYTHON ?= $(REPOSITORY_ROOT)/.venv/bin/python
CONTAINER_ENGINE ?= docker
IMAGE ?=
IMAGE_DIGEST ?=
BASE_IMAGE ?=
RUNTIME_DEVICE ?= cuda
SOURCE_DATE_EPOCH ?= $(shell git -C $(REPOSITORY_ROOT) log -1 --pretty=%ct)

.PHONY: build smoke conformance qualify-evidence

build:
	@test -n "$(BASE_IMAGE)" || { echo "BASE_IMAGE is required" >&2; exit 2; }
	@case "$(BASE_IMAGE)" in *@sha256:*|sha256:*) ;; *) echo "BASE_IMAGE must embed a sha256 digest" >&2; exit 2;; esac
	@test -n "$(IMAGE)" || { echo "IMAGE is required" >&2; exit 2; }
	cd $(REPOSITORY_ROOT) && $(CONTAINER_ENGINE) build \
		--file addins/multimodal/runtime/Dockerfile \
		--build-arg RUNTIME_BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) \
		--tag $(IMAGE) .

smoke:
	@test -n "$(IMAGE)" || { echo "IMAGE is required" >&2; exit 2; }
	$(CONTAINER_ENGINE) run --rm --network none --gpus all \
		--entrypoint python $(IMAGE) \
		-c "import torch; from PIL import Image; assert torch.cuda.is_available(); assert Image.__version__ == '12.3.0'; from invarlock_addins.multimodal.conformance import main; raise SystemExit(main())"

conformance:
	PYTHONPATH=$(REPOSITORY_ROOT)/src:$(REPOSITORY_ROOT)/addins/multimodal/src \
		$(PYTHON) -m invarlock_addins.multimodal.conformance

qualify-evidence:
	@for variable in REQUEST SIGNING_KEY EVIDENCE POLICY BASELINE_ARTIFACT SUBJECT_ARTIFACT SCHEDULE SIGNER VERIFIER_KEY VERIFIER_IDENTITY RECEIPT REPORT IMAGE IMAGE_DIGEST RESOURCE_ROOT CONTENT_STORE; do \
		eval "value=\$$$${variable}"; \
		test -n "$$value" || { echo "$$variable is required" >&2; exit 2; }; \
	done
	cd $(REPOSITORY_ROOT) && \
		INVARLOCK_HF_VISION_TEXT_RESOURCE_ROOT=$(RESOURCE_ROOT) \
		INVARLOCK_HF_VISION_TEXT_CONTENT_STORE=$(CONTENT_STORE) \
		PYTHONPATH=src:addins/multimodal/src $(PYTHON) -m invarlock evaluate $(REQUEST) \
		--signing-key $(SIGNING_KEY) \
		--runtime-image $(IMAGE) \
		--runtime-image-digest $(IMAGE_DIGEST) \
		--container-engine $(CONTAINER_ENGINE) \
		--runtime-device $(RUNTIME_DEVICE)
	cd $(REPOSITORY_ROOT) && \
		PYTHONPATH=src:addins/multimodal/src $(PYTHON) -m invarlock verify $(EVIDENCE) \
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
	cd $(REPOSITORY_ROOT) && \
		PYTHONPATH=src:addins/multimodal/src $(PYTHON) -m invarlock report $(EVIDENCE) \
		--html $(REPORT) \
		--explain
