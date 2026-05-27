from __future__ import annotations

import re
from pathlib import Path


def _get_make_target_block(text: str, target: str) -> str | None:
    pattern = re.compile(rf"^\s*{re.escape(target)}\s*:\s*(?:##.*)?$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None

    lines: list[str] = []
    for line in text[match.end() :].splitlines():
        if line and re.match(r"^[A-Za-z0-9_.-]+\s*:\s*", line):
            break
        lines.append(line)
    return "\n".join(lines)


def test_runtime_image_target_replaces_existing_local_tag_before_build() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    data = makefile.read_text(encoding="utf-8")
    block = _get_make_target_block(data, "runtime-image")

    assert block is not None, "runtime-image target not found in Makefile"
    assert "$(CONTAINER_ENGINE) image inspect $(RUNTIME_IMAGE)" in block
    assert "$(CONTAINER_ENGINE) image rm -f $(RUNTIME_IMAGE)" in block
    assert (
        "$(CONTAINER_ENGINE) build -f runtime/Dockerfile -t $(RUNTIME_IMAGE) ." in block
    )


def test_runtime_image_cuda_target_builds_cuda_tag_with_cuda_requirements() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    data = makefile.read_text(encoding="utf-8")
    block = _get_make_target_block(data, "runtime-image-cuda")

    assert block is not None, "runtime-image-cuda target not found in Makefile"
    assert "$(CONTAINER_ENGINE) image inspect $(RUNTIME_IMAGE_CUDA)" in block
    assert "$(CONTAINER_ENGINE) image rm -f $(RUNTIME_IMAGE_CUDA)" in block
    assert (
        "--build-arg RUNTIME_REQUIREMENTS_AMD64=$(RUNTIME_IMAGE_CUDA_REQUIREMENTS)"
    ) in block
    assert (
        "--build-arg PYTORCH_EXTRA_INDEX_URL=$(RUNTIME_IMAGE_CUDA_INDEX_URL)"
    ) in block
    assert "-t $(RUNTIME_IMAGE_CUDA) ." in block


def test_runtime_image_cuda_quant_target_builds_quant_tag_with_quant_requirements() -> (
    None
):
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    data = makefile.read_text(encoding="utf-8")
    block = _get_make_target_block(data, "runtime-image-cuda-quant")

    assert block is not None, "runtime-image-cuda-quant target not found in Makefile"
    assert "$(CONTAINER_ENGINE) image inspect $(RUNTIME_IMAGE_CUDA_QUANT)" in block
    assert "$(CONTAINER_ENGINE) image rm -f $(RUNTIME_IMAGE_CUDA_QUANT)" in block
    assert "--build-arg RUNTIME_BASE_IMAGE=$(RUNTIME_IMAGE_CUDA_QUANT_BASE)" in block
    assert (
        "--build-arg RUNTIME_REQUIREMENTS_AMD64=$(RUNTIME_IMAGE_CUDA_QUANT_REQUIREMENTS)"
    ) in block
    assert "--build-arg RUNTIME_CUDA_HOME=/usr/local/cuda" in block
    assert "--build-arg RUNTIME_KEEP_BUILD_TOOLCHAIN=1" in block
    assert "--build-arg RUNTIME_PATH_PREFIX=/usr/local/cuda/bin:" in block
    assert (
        "--build-arg PYTORCH_EXTRA_INDEX_URL=$(RUNTIME_IMAGE_CUDA_INDEX_URL)"
    ) in block
    assert "-t $(RUNTIME_IMAGE_CUDA_QUANT) ." in block
