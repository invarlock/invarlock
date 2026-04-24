from __future__ import annotations

from pathlib import Path


def test_makefile_exposes_podman_runtime_targets() -> None:
    text = (Path.cwd() / "Makefile").read_text(encoding="utf-8")

    assert "runtime-image-podman" in text
    assert "runtime-image-podman: CONTAINER_ENGINE=podman" in text
    assert "runtime-image-cuda-podman" in text
    assert "runtime-image-cuda-podman: CONTAINER_ENGINE=podman" in text
    assert "runtime-smoke-podman" in text
    assert "runtime-smoke-podman: CONTAINER_ENGINE=podman" in text
    assert "runtime-smoke-cuda-podman" in text
    assert "runtime-smoke-cuda-podman: CONTAINER_ENGINE=podman" in text
    assert "container-default-smoke:" in text
    assert "container-default-smoke: runtime-image" in text
    assert "container-default-smoke-podman" in text
    assert "container-default-smoke-podman: CONTAINER_ENGINE=podman" in text
    assert "tests/integration/test_container_default_smoke.py" in text
    assert "container-front-door-smoke:" in text
    assert "container-front-door-smoke: runtime-image" in text
    assert "container-front-door-smoke-podman" in text
    assert "container-front-door-smoke-podman: CONTAINER_ENGINE=podman" in text
