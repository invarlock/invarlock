from __future__ import annotations

import re
from pathlib import Path


def _target_block(text: str, target: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(target)}\s*:[^\n]*$", re.MULTILINE)
    match = pattern.search(text)
    assert match is not None, f"{target} target not found"
    lines: list[str] = []
    for line in text[match.end() :].splitlines():
        if line and re.match(r"^[A-Za-z0-9_.-]+\s*:\s*", line):
            break
        lines.append(line)
    return "\n".join(lines)


def test_make_exposes_the_canonical_cpu_runtime_image_build() -> None:
    data = Path("Makefile").read_text(encoding="utf-8")
    block = _target_block(data, "runtime-image")

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
    data = Path("Makefile").read_text(encoding="utf-8")
    block = _target_block(data, "runtime-image-cuda")

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


def test_make_runtime_smoke_uses_the_built_image_offline() -> None:
    data = Path("Makefile").read_text(encoding="utf-8")
    block = _target_block(data, "runtime-smoke")

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
    data = Path("Makefile").read_text(encoding="utf-8")
    block = _target_block(data, "runtime-smoke-cuda")

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


def test_container_front_door_target_runs_the_opt_in_journey() -> None:
    data = Path("Makefile").read_text(encoding="utf-8")
    block = _target_block(data, "container-front-door-smoke")

    assert (
        "runtime-image"
        in data.split("container-front-door-smoke:", 1)[1].splitlines()[0]
    )
    assert "INVARLOCK_RUN_CONTAINER_SMOKE=1" in block
    assert "INVARLOCK_CONTAINER_ENGINE=$(CONTAINER_ENGINE)" in block
    assert "INVARLOCK_RUNTIME_IMAGE=$(RUNTIME_IMAGE)" in block
    assert "tests/integration/test_container_front_door_journey.py" in block
