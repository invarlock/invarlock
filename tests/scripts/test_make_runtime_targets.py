from __future__ import annotations

from pathlib import Path


def test_makefile_exposes_podman_runtime_targets() -> None:
    text = (Path.cwd() / "Makefile").read_text(encoding="utf-8")
    dockerfile_text = (Path.cwd() / "runtime" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    quant_smoke_text = (
        Path.cwd()
        / "examples"
        / "integrations"
        / "_runtime_images"
        / "quant_runtime_image_smoke.py"
    ).read_text(encoding="utf-8")

    assert "runtime-image-podman" in text
    assert "runtime-image-podman: CONTAINER_ENGINE=podman" in text
    assert "runtime-image-cuda-podman" in text
    assert "runtime-image-cuda-podman: CONTAINER_ENGINE=podman" in text
    assert "runtime-image-cuda-quant" in text
    assert "runtime-image-cuda-quant-podman: CONTAINER_ENGINE=podman" in text
    assert "runtime-smoke-podman" in text
    assert "runtime-smoke-podman: CONTAINER_ENGINE=podman" in text
    assert "runtime-smoke-cuda-podman" in text
    assert "runtime-smoke-cuda-podman: CONTAINER_ENGINE=podman" in text
    assert "runtime-smoke-cuda-quant" in text
    assert "runtime-smoke-cuda-quant-podman: CONTAINER_ENGINE=podman" in text
    assert "quant_runtime_image_smoke.py" in text
    assert "require_gptqmodel_runtime" in quant_smoke_text
    assert "require_jit_toolchain=True" in quant_smoke_text
    assert "container-default-smoke:" in text
    assert "container-default-smoke: runtime-image" in text
    assert "container-default-smoke-podman" in text
    assert "container-default-smoke-podman: CONTAINER_ENGINE=podman" in text
    assert "tests/integration/test_container_default_smoke.py" in text
    assert "container-front-door-smoke:" in text
    assert "container-front-door-smoke: runtime-image" in text
    assert "container-front-door-smoke-podman" in text
    assert "container-front-door-smoke-podman: CONTAINER_ENGINE=podman" in text
    assert "apt_get install -y --no-install-recommends build-essential jq" in (
        dockerfile_text
    )
    assert "shutil.which('jq')" in text
    assert "RUNTIME_SOURCE_DATE_EPOCH ?=" in text
    assert text.count("SOURCE_DATE_EPOCH=$(RUNTIME_SOURCE_DATE_EPOCH)") >= 6
    assert text.count("--build-arg SOURCE_DATE_EPOCH=$(RUNTIME_SOURCE_DATE_EPOCH)") == 3
