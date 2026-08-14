from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _target_recipe(path: Path, target: str) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(target)}\s*:[^\n]*$", text, re.MULTILINE)
    assert match is not None, f"{target} target not found in {path}"
    recipe: list[str] = []
    for line in text[match.end() :].splitlines():
        if line and not line.startswith(("\t", " ")):
            break
        recipe.append(line)
    return "\n".join(recipe)


def _assert_hardened_container_smoke(recipe: str) -> None:
    for required in (
        "--network none",
        "--pull=never",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt no-new-privileges",
        "--pids-limit 1024",
        "--user 65532:65532",
        '--tmpfs "/tmp:rw,noexec,nosuid,nodev,size=4g"',
        "--env HOME=/tmp",
        "--env PYTHONDONTWRITEBYTECODE=1",
    ):
        assert required in recipe


def test_cuda_runtime_smoke_is_local_only_and_unprivileged() -> None:
    recipe = _target_recipe(ROOT / "Makefile", "runtime-smoke-cuda")

    _assert_hardened_container_smoke(recipe)
    assert "$(RUNTIME_CUDA_DEVICE_ARGS)" in recipe
    assert "torch.cuda.is_available()" in recipe
    assert "TORCH_DISABLE_NATIVE_JIT" in recipe
    assert "torch.bmm" in recipe


def test_cuda129_runtime_smoke_is_local_only_and_unprivileged() -> None:
    recipe = _target_recipe(ROOT / "Makefile", "runtime-smoke-cuda129")

    _assert_hardened_container_smoke(recipe)
    assert "$(RUNTIME_CUDA_DEVICE_ARGS)" in recipe
    assert "torch.cuda.is_available()" in recipe
    assert "TORCH_DISABLE_NATIVE_JIT" in recipe
    assert "torch.bmm" in recipe


def test_multimodal_smoke_is_local_only_and_unprivileged() -> None:
    recipe = _target_recipe(ROOT / "addins/multimodal/Makefile", "smoke")

    _assert_hardened_container_smoke(recipe)
    assert "--gpus all" in recipe
    assert '--entrypoint python "$(IMAGE)"' in recipe
    assert "torch.cuda.is_available()" in recipe
    assert "TORCH_DISABLE_NATIVE_JIT" in recipe
    assert "torch.bmm" in recipe


def test_gguf_smoke_is_local_only_and_unprivileged() -> None:
    recipe = _target_recipe(ROOT / "addins/gguf/Makefile", "smoke")

    _assert_hardened_container_smoke(recipe)
    assert '--entrypoint python "$(IMAGE)"' in recipe
    assert "invarlock_addins.gguf.conformance" in recipe


def test_tensorrt_smoke_is_local_only_and_unprivileged() -> None:
    recipe = _target_recipe(ROOT / "addins/tensorrt_llm/Makefile", "smoke")

    _assert_hardened_container_smoke(recipe)
    assert "--gpus all" in recipe
    assert '--entrypoint /opt/invarlock/bin/vendor-python "$(IMAGE)"' in recipe
    assert "--env LD_LIBRARY_PATH=/usr/local/tensorrt/lib" in recipe
    assert "ctypes.CDLL('libnvonnxparser.so.10')" in recipe
    assert "torch.cuda.is_available()" in recipe
