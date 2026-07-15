from __future__ import annotations

from pathlib import Path


def test_llama_cpp_runtime_image_pins_source_and_disables_host_tuning() -> None:
    text = (Path.cwd() / "runtime" / "Dockerfile.llama-cpp").read_text(encoding="utf-8")

    assert "b10015.tar.gz" in text
    assert "5ab75e394f4c71425ecce64a213dab3b8e3e9cfe0f19d0dcda4d5a4f7733da83" in text
    assert "12127defda4f41b7679cb2477a4b0d65ee6a0c8f" in text
    assert "ARG LLAMA_CPP_APT_SNAPSHOT" in text
    assert "APT::Snapshot=${LLAMA_CPP_APT_SNAPSHOT}" in text
    assert (
        "else" not in text.split("apt_get()", 1)[1].split("ARG SOURCE_DATE_EPOCH", 1)[0]
    )
    assert "ADD --checksum=sha256:${LLAMA_CPP_SOURCE_SHA256}" in text
    assert "-DGGML_NATIVE=OFF" in text
    assert "-DGGML_OPENMP=OFF" in text
    assert "-DGGML_CCACHE=OFF" in text
    assert "-DBUILD_SHARED_LIBS=OFF" in text
    assert "-DLLAMA_BUILD_EXAMPLES=OFF" in text
    # b10015 nests the completion tool behind this upstream configuration gate;
    # the build target below still compiles and copies only llama-completion.
    assert "-DLLAMA_BUILD_SERVER=ON" in text
    assert "-DLLAMA_BUILD_TOOLS=ON" in text
    assert "--target llama-completion" in text
    assert '--parallel "${LLAMA_CPP_BUILD_JOBS}"' in text
    assert 'test "${LLAMA_CPP_BUILD_JOBS}" -le 8' in text
    assert "/opt/llama.cpp/source/llama.cpp-b10015.tar.gz" in text
    assert "COPY --from=llama-cpp-build" in text
    assert "requirements/workflows/core-py312.txt" in text
    assert "runtime-image-py312.txt" not in text
    assert "runtime-image-py312-aarch64.txt" not in text


def test_docker_context_includes_llama_cpp_dockerfile() -> None:
    text = (Path.cwd() / ".dockerignore").read_text(encoding="utf-8")

    assert "!runtime/Dockerfile.llama-cpp" in text
    assert "!requirements/workflows/core-py312.txt" in text


def test_makefile_exposes_separate_gguf_build_and_offline_smoke() -> None:
    text = (Path.cwd() / "Makefile").read_text(encoding="utf-8")

    assert "runtime-image-gguf:" in text
    assert "runtime-image-gguf: runtime-image" not in text
    assert "-f runtime/Dockerfile.llama-cpp" in text
    assert "runtime-smoke-gguf:" in text
    assert "LLAMA_CPP_BUILD_JOBS ?= 2" in text
    assert "--build-arg LLAMA_CPP_BUILD_JOBS=$(LLAMA_CPP_BUILD_JOBS)" in text
    assert "RUNTIME_IMAGE_GGUF_BUILD ?= $(RUNTIME_IMAGE_GGUF)-candidate" in text
    assert "-t $(RUNTIME_IMAGE_GGUF_BUILD)" in text
    assert "runtime-smoke-gguf RUNTIME_IMAGE_GGUF=$(RUNTIME_IMAGE_GGUF_BUILD)" in text
    assert "Set GGUF_BLACKBOX_MODEL before building" in text
    assert (
        "runtime-blackbox-gguf RUNTIME_IMAGE_GGUF=$(RUNTIME_IMAGE_GGUF_BUILD)" in text
    )
    assert "image tag $(RUNTIME_IMAGE_GGUF_BUILD) $(RUNTIME_IMAGE_GGUF)" in text
    assert "image rm -f $(RUNTIME_IMAGE_GGUF)" not in text

    build = text.index("runtime-image-gguf:")
    preflight = text.index("Set GGUF_BLACKBOX_MODEL before building", build)
    image_build = text.index("-f runtime/Dockerfile.llama-cpp", build)
    smoke = text.index("$(MAKE) runtime-smoke-gguf", image_build)
    blackbox = text.index("$(MAKE) runtime-blackbox-gguf", smoke)
    stable_tag = text.index("image tag $(RUNTIME_IMAGE_GGUF_BUILD)", blackbox)
    assert build < preflight < image_build < smoke < blackbox < stable_tag


def test_makefile_exposes_optional_pinned_gguf_blackbox() -> None:
    text = (Path.cwd() / "Makefile").read_text(encoding="utf-8")

    assert "runtime-blackbox-gguf:" in text
    assert "GGUF_BLACKBOX_MODEL ?=" in text
    assert "scripts/release/gguf_runtime_blackbox.py" in text
    assert "this target never downloads it" in text
    assert "--network none" in text
    assert "--entrypoint /bin/sh" in text
    assert "sha256sum -c -" in text
    assert "import invarlock" in text
    assert "/opt/llama.cpp/llama-completion --version" in text
    assert "RUNTIME_IMAGE_APT_SNAPSHOT ?= 20260712T232152Z" in text
    assert "--build-arg RUNTIME_APT_SNAPSHOT=$(RUNTIME_IMAGE_APT_SNAPSHOT)" in text
