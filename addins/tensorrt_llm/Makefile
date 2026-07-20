REPOSITORY_ROOT := $(abspath $(CURDIR)/../..)
PYTHON ?= $(REPOSITORY_ROOT)/.venv/bin/python
QUALIFICATION_DRIVER_PYTHON ?= $(PYTHON)
CONTAINER_ENGINE ?= docker
IMAGE ?= invarlock-runtime:tensorrt-llm-local
IMAGE_DIGEST ?= $(shell $(CONTAINER_ENGINE) image inspect --format '{{.Id}}' $(IMAGE) 2>/dev/null)
QUALIFICATION_IMAGE ?= $(IMAGE_DIGEST)
RUNTIME_DEVICE ?= cuda
QUALIFICATION_DEVICE ?= $(RUNTIME_DEVICE)
QUALIFICATION_CPUS ?=
QUALIFICATION_MEMORY_MIB ?=
QUALIFICATION_USER ?=
SOURCE_DATE_EPOCH ?= $(shell git -C $(REPOSITORY_ROOT) log -1 --pretty=%ct)
CANARY_TMPFS_GIB ?= 8
BUILD_STATEMENT ?=

.PHONY: build smoke qualification-host-check conformance canary qualify-canary qualify-preflight qualify-evidence

build:
	$(foreach variable,SOURCE_COMMIT SOURCE_BUNDLE SOURCE_BUNDLE_SHA256,$(if $(strip $($(variable))),,$(error $(variable) is required)))
	"$(PYTHON)" "$(REPOSITORY_ROOT)/scripts/authenticated_runtime_build.py" \
		--repository "$(REPOSITORY_ROOT)" \
		--source-commit "$(SOURCE_COMMIT)" \
		--source-bundle "$(SOURCE_BUNDLE)" \
		--source-bundle-sha256 "$(SOURCE_BUNDLE_SHA256)" \
		--container-engine "$(CONTAINER_ENGINE)" \
		--dockerfile addins/tensorrt_llm/runtime/Dockerfile \
		--image "$(IMAGE)" \
		--build-arg "SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH)" $(if $(strip $(BUILD_STATEMENT)),--statement "$(BUILD_STATEMENT)")

smoke:
	$(CONTAINER_ENGINE) run --rm --network none --gpus all \
		--pull=never --read-only --cap-drop=ALL \
		--security-opt no-new-privileges --pids-limit 1024 \
		--user 65532:65532 \
		--tmpfs "/tmp:rw,noexec,nosuid,nodev,size=4g" \
		--env HOME=/tmp --env PYTHONDONTWRITEBYTECODE=1 \
		--env USER=65532 --env LOGNAME=65532 \
		--env LD_LIBRARY_PATH=/usr/local/tensorrt/lib \
		--entrypoint /opt/invarlock/bin/vendor-python "$(IMAGE)" \
		-c "import ctypes; ctypes.CDLL('libnvonnxparser.so.10'); import tensorrt_llm; import torch; assert torch.cuda.is_available(); from invarlock_addins.tensorrt_llm.conformance import main; raise SystemExit(main())"

qualification-host-check:
	@PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH= "$(PYTHON)" -c "from invarlock.core.registry import CoreRegistry; from invarlock.core.runtime_provider import INVARLOCK_RUNTIME_PROVIDER_ABI; provider = CoreRegistry().get_runtime_provider('tensorrt_llm'); assert provider.name == 'tensorrt_llm' and provider.abi_version == INVARLOCK_RUNTIME_PROVIDER_ABI" || { \
		printf '%s\n' "qualification host check failed for PYTHON=$(PYTHON): install matching core (invarlock) and add-in (invarlock-runtime-tensorrt-llm) wheels into that interpreter; PYTHONPATH/importability alone is insufficient because provider authorization requires installed distribution metadata." >&2; \
		exit 1; \
	}

conformance: qualification-host-check
	PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH= "$(PYTHON)" -m invarlock_addins.tensorrt_llm.conformance

canary:
	@set -eu; \
	canonical_input_root="$$(PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
		PYTHONPATH="$(REPOSITORY_ROOT)/src:$(REPOSITORY_ROOT)/addins/tensorrt_llm/src" \
		"$(PYTHON)" "$(REPOSITORY_ROOT)/scripts/tensorrt_llm_canary_preflight.py" \
			--image "$(IMAGE)" \
			--image-digest "$(IMAGE_DIGEST)" \
			--input-root "$(INPUT_ROOT)" \
			--engine-bundle "$(ENGINE_BUNDLE)" \
			--tokenizer-contract "$(TOKENIZER_CONTRACT)" \
			--expected-engine-tree-sha256 "$(EXPECTED_ENGINE_TREE_SHA256)" \
			--expected-tokenizer-sha256 "$(EXPECTED_TOKENIZER_SHA256)" \
			--expected-output-sha256 "$(EXPECTED_OUTPUT_SHA256)" \
			--tmpfs-gib "$(CANARY_TMPFS_GIB)")"; \
	"$(CONTAINER_ENGINE)" run --rm --network none --gpus all --read-only \
		--cap-drop=ALL --security-opt no-new-privileges --pids-limit 1024 \
		--user 65532:65532 \
		--tmpfs "/tmp:rw,noexec,nosuid,nodev,size=$(CANARY_TMPFS_GIB)g" \
		--env INVARLOCK_CONTAINER_EXECUTION=1 \
		--env "INVARLOCK_RUNTIME_IMAGE=$(IMAGE_DIGEST)" \
		--env "INVARLOCK_RUNTIME_IMAGE_DIGEST=$(IMAGE_DIGEST)" \
		--mount "type=bind,src=$$canonical_input_root,dst=/inputs,readonly" \
		--entrypoint /opt/invarlock/cli-venv/bin/python "$(IMAGE)" \
		-m invarlock_addins.tensorrt_llm.canary \
		--engine-bundle "/inputs/$(ENGINE_BUNDLE)" \
		--tokenizer-contract "/inputs/$(TOKENIZER_CONTRACT)" \
		--runner /opt/invarlock/bin/tensorrt-llm-runner \
		--expected-engine-tree-sha256 "$(EXPECTED_ENGINE_TREE_SHA256)" \
		--expected-tokenizer-sha256 "$(EXPECTED_TOKENIZER_SHA256)" \
		--expected-output-sha256 "$(EXPECTED_OUTPUT_SHA256)"

qualify-canary:
	$(foreach variable,REQUEST SIGNING_KEY QUALIFICATION_IMAGE IMAGE_DIGEST EVIDENCE TRUST_PROFILE RECEIPT SOURCE_COMMIT SOURCE_BUNDLE SOURCE_BUNDLE_SHA256 CANDIDATE_WHEEL_MANIFEST SUMMARY QUALIFICATION_DEVICE,$(if $(strip $($(variable))),,$(error $(variable) is required)))
	cd "$(REPOSITORY_ROOT)" && \
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
	$(foreach variable,REQUEST SIGNING_KEY QUALIFICATION_IMAGE IMAGE_DIGEST EVIDENCE TRUST_PROFILE RECEIPT CANARY_EVIDENCE CANARY_RECEIPT CANARY_TRUST_PROFILE SOURCE_COMMIT SOURCE_BUNDLE SOURCE_BUNDLE_SHA256 CANDIDATE_WHEEL_MANIFEST QUALIFICATION_DEVICE,$(if $(strip $($(variable))),,$(error $(variable) is required)))
	cd "$(REPOSITORY_ROOT)" && \
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
	$(foreach variable,REQUEST SIGNING_KEY QUALIFICATION_IMAGE IMAGE_DIGEST EVIDENCE TRUST_PROFILE RECEIPT CANARY_EVIDENCE CANARY_RECEIPT CANARY_TRUST_PROFILE SOURCE_COMMIT SOURCE_BUNDLE SOURCE_BUNDLE_SHA256 CANDIDATE_WHEEL_MANIFEST SUMMARY QUALIFICATION_DEVICE,$(if $(strip $($(variable))),,$(error $(variable) is required)))
	cd "$(REPOSITORY_ROOT)" && \
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
