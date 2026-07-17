REPOSITORY_ROOT := $(abspath $(CURDIR)/../..)
PYTHON ?= $(REPOSITORY_ROOT)/.venv/bin/python
QUALIFICATION_DRIVER_PYTHON ?= $(PYTHON)
CONTAINER_ENGINE ?= docker
IMAGE ?=
IMAGE_DIGEST ?=
QUALIFICATION_IMAGE ?= $(IMAGE_DIGEST)
BASE_IMAGE ?=
RUNTIME_DEVICE ?= cuda
QUALIFICATION_DEVICE ?= $(RUNTIME_DEVICE)
QUALIFICATION_CPUS ?=
QUALIFICATION_MEMORY_MIB ?=
QUALIFICATION_USER ?=
SOURCE_DATE_EPOCH ?= $(shell git -C $(REPOSITORY_ROOT) log -1 --pretty=%ct)
BUILD_STATEMENT ?=

.PHONY: build smoke qualification-host-check conformance qualify-canary qualify-preflight qualify-evidence

build:
	$(foreach variable,SOURCE_COMMIT SOURCE_BUNDLE SOURCE_BUNDLE_SHA256,$(if $(strip $($(variable))),,$(error $(variable) is required)))
	@test -n "$(BASE_IMAGE)" || { echo "BASE_IMAGE is required" >&2; exit 2; }
	@case "$(BASE_IMAGE)" in ?*@sha256:*) ;; *) echo "BASE_IMAGE must use a named repository@sha256 reference; a local config ID is not valid in Dockerfile FROM" >&2; exit 2;; esac
	@test -n "$(IMAGE)" || { echo "IMAGE is required" >&2; exit 2; }
	"$(PYTHON)" "$(REPOSITORY_ROOT)/scripts/authenticated_runtime_build.py" \
		--repository "$(REPOSITORY_ROOT)" \
		--source-commit "$(SOURCE_COMMIT)" \
		--source-bundle "$(SOURCE_BUNDLE)" \
		--source-bundle-sha256 "$(SOURCE_BUNDLE_SHA256)" \
		--container-engine "$(CONTAINER_ENGINE)" \
		--dockerfile addins/multimodal/runtime/Dockerfile \
		--image "$(IMAGE)" \
		--require-base-source-labels "$(BASE_IMAGE)" \
		--build-arg "RUNTIME_BASE_IMAGE=$(BASE_IMAGE)" \
		--build-arg "SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH)" $(if $(strip $(BUILD_STATEMENT)),--statement "$(BUILD_STATEMENT)")

smoke:
	@test -n "$(IMAGE)" || { echo "IMAGE is required" >&2; exit 2; }
	$(CONTAINER_ENGINE) run --rm --network none --gpus all \
		--pull=never --read-only --cap-drop=ALL \
		--security-opt no-new-privileges --pids-limit 1024 \
		--user 65532:65532 \
		--tmpfs "/tmp:rw,noexec,nosuid,nodev,size=4g" \
		--env HOME=/tmp --env PYTHONDONTWRITEBYTECODE=1 \
		--entrypoint python "$(IMAGE)" \
		-c "import torch, torchvision; from PIL import Image; from transformers import Qwen2VLImageProcessor, Qwen2VLVideoProcessor; assert torch.cuda.is_available(); assert Image.__version__ == '12.3.0'; assert torchvision.__version__ == '0.26.0+cu128'; assert Qwen2VLImageProcessor().to_dict(); assert Qwen2VLVideoProcessor().to_dict(); from invarlock_addins.multimodal.conformance import main; raise SystemExit(main())"

qualification-host-check:
	@PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH= "$(PYTHON)" -c "from invarlock.core.registry import CoreRegistry; from invarlock.core.runtime_provider import INVARLOCK_RUNTIME_PROVIDER_ABI; provider = CoreRegistry().get_runtime_provider('hf_vision_text'); assert provider.name == 'hf_vision_text' and provider.abi_version == INVARLOCK_RUNTIME_PROVIDER_ABI" || { \
		printf '%s\n' "qualification host check failed for PYTHON=$(PYTHON): install matching core (invarlock) and add-in (invarlock-runtime-hf-vision-text) wheels into that interpreter; PYTHONPATH/importability alone is insufficient because provider authorization requires installed distribution metadata." >&2; \
		exit 1; \
	}

conformance: qualification-host-check
	PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH= "$(PYTHON)" -m invarlock_addins.multimodal.conformance

qualify-canary:
	$(foreach variable,REQUEST SIGNING_KEY QUALIFICATION_IMAGE IMAGE_DIGEST RESOURCE_ROOT CONTENT_STORE EVIDENCE TRUST_PROFILE RECEIPT SOURCE_COMMIT SOURCE_BUNDLE SOURCE_BUNDLE_SHA256 CANDIDATE_WHEEL_MANIFEST SUMMARY QUALIFICATION_DEVICE,$(if $(strip $($(variable))),,$(error $(variable) is required)))
	cd "$(REPOSITORY_ROOT)" && \
		INVARLOCK_HF_VISION_TEXT_RESOURCE_ROOT="$(RESOURCE_ROOT)" \
		INVARLOCK_HF_VISION_TEXT_CONTENT_STORE="$(CONTENT_STORE)" \
		"$(QUALIFICATION_DRIVER_PYTHON)" -I -S scripts/runtime_qualification.py canary \
		--python "$(PYTHON)" \
		--request "$(REQUEST)" \
		--signing-key "$(SIGNING_KEY)" \
		--runtime-image "$(QUALIFICATION_IMAGE)" \
		--runtime-image-digest "$(IMAGE_DIGEST)" \
		--evidence "$(EVIDENCE)" \
		--trust-profile "$(TRUST_PROFILE)" \
		--receipt "$(RECEIPT)" \
		--source-commit "$(SOURCE_COMMIT)" \
		--source-bundle "$(SOURCE_BUNDLE)" \
		--source-bundle-sha256 "$(SOURCE_BUNDLE_SHA256)" \
		--candidate-wheel-manifest "$(CANDIDATE_WHEEL_MANIFEST)" \
		--container-engine "$(CONTAINER_ENGINE)" \
		--runtime-device "$(QUALIFICATION_DEVICE)" \
		$(if $(strip $(QUALIFICATION_CPUS)),--runtime-cpus "$(QUALIFICATION_CPUS)") \
		$(if $(strip $(QUALIFICATION_MEMORY_MIB)),--runtime-memory-mib "$(QUALIFICATION_MEMORY_MIB)") \
		$(if $(strip $(QUALIFICATION_USER)),--runtime-user "$(QUALIFICATION_USER)") \
		$(if $(strip $(REPORT)),--report "$(REPORT)") \
		--summary "$(SUMMARY)"

qualify-preflight:
	$(foreach variable,REQUEST SIGNING_KEY QUALIFICATION_IMAGE IMAGE_DIGEST RESOURCE_ROOT CONTENT_STORE EVIDENCE TRUST_PROFILE RECEIPT CANARY_EVIDENCE CANARY_RECEIPT CANARY_TRUST_PROFILE SOURCE_COMMIT SOURCE_BUNDLE SOURCE_BUNDLE_SHA256 CANDIDATE_WHEEL_MANIFEST QUALIFICATION_DEVICE,$(if $(strip $($(variable))),,$(error $(variable) is required)))
	cd "$(REPOSITORY_ROOT)" && \
		INVARLOCK_HF_VISION_TEXT_RESOURCE_ROOT="$(RESOURCE_ROOT)" \
		INVARLOCK_HF_VISION_TEXT_CONTENT_STORE="$(CONTENT_STORE)" \
		"$(QUALIFICATION_DRIVER_PYTHON)" -I -S scripts/runtime_qualification.py readiness \
		--python "$(PYTHON)" \
		--request "$(REQUEST)" \
		--signing-key "$(SIGNING_KEY)" \
		--runtime-image "$(QUALIFICATION_IMAGE)" \
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

qualify-evidence:
	$(foreach variable,REQUEST SIGNING_KEY QUALIFICATION_IMAGE IMAGE_DIGEST RESOURCE_ROOT CONTENT_STORE EVIDENCE TRUST_PROFILE RECEIPT CANARY_EVIDENCE CANARY_RECEIPT CANARY_TRUST_PROFILE SOURCE_COMMIT SOURCE_BUNDLE SOURCE_BUNDLE_SHA256 CANDIDATE_WHEEL_MANIFEST SUMMARY QUALIFICATION_DEVICE,$(if $(strip $($(variable))),,$(error $(variable) is required)))
	cd "$(REPOSITORY_ROOT)" && \
		INVARLOCK_HF_VISION_TEXT_RESOURCE_ROOT="$(RESOURCE_ROOT)" \
		INVARLOCK_HF_VISION_TEXT_CONTENT_STORE="$(CONTENT_STORE)" \
		"$(QUALIFICATION_DRIVER_PYTHON)" -I -S scripts/runtime_qualification.py run \
		--python "$(PYTHON)" \
		--request "$(REQUEST)" \
		--signing-key "$(SIGNING_KEY)" \
		--runtime-image "$(QUALIFICATION_IMAGE)" \
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
		$(if $(strip $(QUALIFICATION_USER)),--runtime-user "$(QUALIFICATION_USER)") \
		$(if $(strip $(REPORT)),--report "$(REPORT)") \
		--summary "$(SUMMARY)"
