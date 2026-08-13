from __future__ import annotations

from pathlib import Path

from tests._support_repository_contracts import MakefileContract

MAKE = MakefileContract.read(Path(__file__).resolve().parents[2] / "Makefile")


def test_make_exposes_the_canonical_cpu_runtime_image_build() -> None:
    data = MAKE.text
    block = MAKE.target("runtime-image").text

    assert 'test -n "$(CONTAINER_ENGINE)"' in block
    assert 'test -n "$(RUNTIME_SOURCE_DATE_EPOCH)"' in block
    assert "scripts/authenticated_runtime_build.py" in block
    assert '--source-bundle "$(RUNTIME_SOURCE_BUNDLE)"' in block
    assert "--dockerfile runtime/Dockerfile" in block
    assert '--image "$(RUNTIME_IMAGE)"' in block
    assert '--build-arg "SOURCE_DATE_EPOCH=$(RUNTIME_SOURCE_DATE_EPOCH)"' in block
    assert '--statement "$(RUNTIME_BUILD_STATEMENT)"' in block
    assert "$(CONTAINER_ENGINE) build" not in block
    assert "runtime-image-quant" not in data


def test_make_exposes_a_separate_minimal_cuda_runtime_image_build() -> None:
    data = MAKE.text
    block = MAKE.target("runtime-image-cuda").text

    assert 'test -n "$(CONTAINER_ENGINE)"' in block
    assert 'test -n "$(RUNTIME_SOURCE_DATE_EPOCH)"' in block
    assert "scripts/authenticated_runtime_build.py" in block
    assert '--source-bundle "$(RUNTIME_SOURCE_BUNDLE)"' in block
    assert "--platform linux/amd64" in block
    assert '--build-arg "SOURCE_DATE_EPOCH=$(RUNTIME_SOURCE_DATE_EPOCH)"' in block
    assert '--statement "$(RUNTIME_BUILD_STATEMENT)"' in block
    assert "--dockerfile runtime/Dockerfile.cuda" in block
    assert '--image "$(RUNTIME_IMAGE_CUDA)"' in block
    assert "$(CONTAINER_ENGINE) build" not in block
    assert "runtime-image-cuda-quant" not in data


def test_make_exposes_the_blackwell_capable_cuda_runtime_profile() -> None:
    block = MAKE.target("runtime-image-cuda129").text

    assert "scripts/authenticated_runtime_build.py" in block
    assert "--dockerfile runtime/Dockerfile.cuda" in block
    assert '--build-arg "CUDA_PROFILE=cu129"' in block
    assert '--image "$(RUNTIME_IMAGE_CUDA129)"' in block
    assert "--platform linux/amd64" in block


def test_make_runtime_smoke_uses_the_built_image_offline() -> None:
    block = MAKE.target("runtime-smoke").text

    assert "$(CONTAINER_ENGINE) run --rm --network none" in block
    assert "--pull=never --read-only --cap-drop=ALL" in block
    assert "--security-opt no-new-privileges --pids-limit 1024" in block
    assert "--user 65532:65532" in block
    assert '--tmpfs "/tmp:rw,noexec,nosuid,nodev,size=4g"' in block
    assert "--env HOME=/tmp --env PYTHONDONTWRITEBYTECODE=1" in block
    assert "--entrypoint python $(RUNTIME_IMAGE)" in block
    assert "import accelerate, safetensors, torch, transformers" in block
    assert "accelerate.__version__ == '1.14.0'" in block
    assert "safetensors.__version__ == '0.8.0'" in block
    assert "transformers.__version__ == '5.14.1'" in block


def test_make_cuda_runtime_smoke_requires_a_visible_gpu() -> None:
    data = MAKE.text
    block = MAKE.target("runtime-smoke-cuda").text

    assert (
        "$(CONTAINER_ENGINE) run --rm --network none $(RUNTIME_CUDA_DEVICE_ARGS)"
        in block
    )
    assert "--entrypoint python $(RUNTIME_IMAGE_CUDA)" in block
    assert "import accelerate, safetensors, torch, transformers" in block
    assert "accelerate.__version__ == '1.14.0'" in block
    assert "safetensors.__version__ == '0.8.0'" in block
    assert "transformers.__version__ == '5.14.1'" in block
    assert "assert torch.__version__ == '2.13.0+cu126'" in block
    assert "assert torch.version.cuda == '12.6'" in block
    assert "assert torch.cuda.is_available()" in block
    assert "assert os.environ.get('TORCH_DISABLE_NATIVE_JIT') == '1'" in block
    assert "torch.bmm(left, right)" in block
    assert (
        "RUNTIME_CUDA_DEVICE_ARGS = $(if $(filter podman,$(CONTAINER_ENGINE)),"
        "--device nvidia.com/gpu=all,--gpus all)"
    ) in data


def test_make_cuda129_smoke_executes_a_real_kernel() -> None:
    block = MAKE.target("runtime-smoke-cuda129").text

    assert "$(RUNTIME_CUDA_DEVICE_ARGS)" in block
    assert "--entrypoint python $(RUNTIME_IMAGE_CUDA129)" in block
    assert "torch.__version__ == '2.13.0+cu129'" in block
    assert "torch.version.cuda == '12.9'" in block
    assert "torch.bmm(left, right)" in block


def test_container_front_door_target_runs_the_opt_in_journey() -> None:
    block = MAKE.target("container-front-door-smoke").text

    assert "runtime-image" in MAKE.target("container-front-door-smoke").prerequisites
    assert "INVARLOCK_RUN_CONTAINER_SMOKE=1" in block
    assert "INVARLOCK_CONTAINER_ENGINE=$(CONTAINER_ENGINE)" in block
    assert "INVARLOCK_RUNTIME_IMAGE=$(RUNTIME_IMAGE)" in block
    assert "tests/integration/test_container_front_door_journey.py" in block
