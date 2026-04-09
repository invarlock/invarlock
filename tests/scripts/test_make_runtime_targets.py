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
