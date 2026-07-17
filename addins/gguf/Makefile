REPOSITORY_ROOT := $(abspath $(CURDIR)/../..)
PYTHON ?= $(REPOSITORY_ROOT)/.venv/bin/python
QUALIFICATION_DRIVER_PYTHON ?= $(PYTHON)
CONTAINER_ENGINE ?= docker
IMAGE ?= invarlock-runtime:gguf-local
IMAGE_DIGEST ?= $(shell $(CONTAINER_ENGINE) image inspect --format '{{.Id}}' $(IMAGE) 2>/dev/null)
QUALIFICATION_IMAGE ?= $(IMAGE_DIGEST)
RUNTIME_DEVICE ?= cpu
QUALIFICATION_DEVICE ?= $(RUNTIME_DEVICE)
QUALIFICATION_CPUS ?=
QUALIFICATION_MEMORY_MIB ?=
QUALIFICATION_USER ?=
SOURCE_DATE_EPOCH ?= $(shell git -C $(REPOSITORY_ROOT) log -1 --pretty=%ct)
LLAMA_CPP_APT_SNAPSHOT ?=
LLAMA_CPP_BUILD_JOBS ?= 4
BUILD_STATEMENT ?=

.PHONY: build smoke qualification-host-check conformance qualify-canary qualify-preflight qualify-evidence

build:
	$(foreach variable,SOURCE_COMMIT SOURCE_BUNDLE SOURCE_BUNDLE_SHA256,$(if $(strip $($(variable))),,$(error $(variable) is required)))
	@test -n "$(LLAMA_CPP_APT_SNAPSHOT)" || { echo "LLAMA_CPP_APT_SNAPSHOT is required" >&2; exit 2; }
	@printf '%s\n' "$(LLAMA_CPP_APT_SNAPSHOT)" | grep -Eq '^[0-9]{8}T[0-9]{6}Z$$' || { echo "LLAMA_CPP_APT_SNAPSHOT must use YYYYMMDDTHHMMSSZ" >&2; exit 2; }
	@case "$(LLAMA_CPP_BUILD_JOBS)" in ''|*[!0-9]*) echo "LLAMA_CPP_BUILD_JOBS must be an integer from 1 to 8" >&2; exit 2;; esac; \
		test "$(LLAMA_CPP_BUILD_JOBS)" -ge 1 && test "$(LLAMA_CPP_BUILD_JOBS)" -le 8 || { echo "LLAMA_CPP_BUILD_JOBS must be an integer from 1 to 8" >&2; exit 2; }
	"$(PYTHON)" "$(REPOSITORY_ROOT)/scripts/authenticated_runtime_build.py" \
		--repository "$(REPOSITORY_ROOT)" \
		--source-commit "$(SOURCE_COMMIT)" \
		--source-bundle "$(SOURCE_BUNDLE)" \
		--source-bundle-sha256 "$(SOURCE_BUNDLE_SHA256)" \
		--container-engine "$(CONTAINER_ENGINE)" \
		--dockerfile addins/gguf/runtime/Dockerfile \
		--image "$(IMAGE)" \
		--build-arg "SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH)" \
		--build-arg "LLAMA_CPP_APT_SNAPSHOT=$(LLAMA_CPP_APT_SNAPSHOT)" \
		--build-arg "LLAMA_CPP_BUILD_JOBS=$(LLAMA_CPP_BUILD_JOBS)" $(if $(strip $(BUILD_STATEMENT)),--statement "$(BUILD_STATEMENT)")

smoke:
	$(CONTAINER_ENGINE) run --rm --network none \
		--pull=never --read-only --cap-drop=ALL \
		--security-opt no-new-privileges --pids-limit 1024 \
		--user 65532:65532 \
		--tmpfs "/tmp:rw,noexec,nosuid,nodev,size=4g" \
		--env HOME=/tmp --env PYTHONDONTWRITEBYTECODE=1 \
		--entrypoint python "$(IMAGE)" \
		-c "import shutil; from invarlock_addins.gguf.conformance import main; assert shutil.which('llama-completion'); raise SystemExit(main())"

qualification-host-check:
	@PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH= "$(PYTHON)" -c "from invarlock.core.registry import CoreRegistry; from invarlock.core.runtime_provider import INVARLOCK_RUNTIME_PROVIDER_ABI; provider = CoreRegistry().get_runtime_provider('llama_cpp'); assert provider.name == 'llama_cpp' and provider.abi_version == INVARLOCK_RUNTIME_PROVIDER_ABI" || { \
		printf '%s\n' "qualification host check failed for PYTHON=$(PYTHON): install matching core (invarlock) and add-in (invarlock-runtime-gguf) wheels into that interpreter; PYTHONPATH/importability alone is insufficient because provider authorization requires installed distribution metadata." >&2; \
		exit 1; \
	}

conformance: qualification-host-check
	PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH= "$(PYTHON)" -m invarlock_addins.gguf.conformance

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
