from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _makefile(addin: str) -> str:
    return ROOT.joinpath("addins", addin, "Makefile").read_text(encoding="utf-8")


def test_gguf_runtime_has_reproducible_build_smoke_and_evidence_targets() -> None:
    makefile = _makefile("gguf")
    assert "LLAMA_CPP_APT_SNAPSHOT is required" in makefile
    assert "addins/gguf/runtime/Dockerfile" in makefile
    assert "--network none" in makefile
    assert "llama-completion" in makefile
    assert "qualify-evidence:" in makefile
    assert "-m invarlock evaluate" in makefile
    assert "-m invarlock verify" in makefile


def test_tensorrt_runtime_has_gpu_build_smoke_canary_and_evidence_targets() -> None:
    makefile = _makefile("tensorrt_llm")
    assert "addins/tensorrt_llm/runtime/Dockerfile" in makefile
    assert "--network none --gpus all" in makefile
    assert "torch.cuda.is_available" in makefile
    assert "-m invarlock_addins.tensorrt_llm.canary" in makefile
    assert "qualify-evidence:" in makefile
    assert "-m invarlock evaluate" in makefile
    assert "-m invarlock verify" in makefile


def test_multimodal_runtime_has_conformance_and_evidence_qualification_targets() -> (
    None
):
    makefile = _makefile("multimodal")
    dockerfile = ROOT.joinpath("addins/multimodal/runtime/Dockerfile").read_text(
        encoding="utf-8"
    )
    assert "addins/multimodal/runtime/Dockerfile" in makefile
    assert "BASE_IMAGE must embed a sha256 digest" in makefile
    assert "--network none --gpus all" in makefile
    assert "torch.cuda.is_available" in makefile
    assert "conformance:" in makefile
    assert "qualify-evidence:" in makefile
    assert "INVARLOCK_HF_VISION_TEXT_RESOURCE_ROOT" in makefile
    assert "INVARLOCK_HF_VISION_TEXT_CONTENT_STORE" in makefile
    assert "-m invarlock evaluate" in makefile
    assert "-m invarlock verify" in makefile
    assert "-m invarlock report" in makefile
    assert "RUNTIME_BASE_IMAGE" in dockerfile
    assert "multimodal-runtime-py312.txt" in dockerfile
    assert "--require-hashes" in dockerfile
    assert 'ENTRYPOINT ["python", "-m", "invarlock"]' in dockerfile


def test_optional_runtime_qualification_remains_addin_owned() -> None:
    root_makefile = ROOT.joinpath("Makefile").read_text(encoding="utf-8")
    assert "runtime-image-gguf" not in root_makefile
    assert "runtime-image-tensorrt-llm" not in root_makefile
