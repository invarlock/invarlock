from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = REPO_ROOT / "examples" / "integrations" / "_runtime_images"
REQ_DIR = RUNTIME_DIR / "requirements"


def test_example_runtime_image_scripts_are_shell_valid_and_executable() -> None:
    for script_name in (
        "build_example_runtime_image.sh",
        "smoke_example_runtime_image.sh",
    ):
        script = RUNTIME_DIR / script_name
        subprocess.run(["bash", "-n", str(script)], check=True)
        assert os.access(script, os.X_OK)


def test_example_runtime_smoke_uses_host_driver_when_available() -> None:
    script_text = (RUNTIME_DIR / "smoke_example_runtime_image.sh").read_text(
        encoding="utf-8"
    )
    smoke_text = (RUNTIME_DIR / "quant_runtime_image_smoke.py").read_text(
        encoding="utf-8"
    )

    assert "--gpus all" in script_text
    assert "nvidia-smi" in script_text
    assert "--require-gpu" in script_text
    assert "torch.cuda.is_available()" in smoke_text


def test_example_runtime_docs_describe_host_driver_and_image_user_space() -> None:
    text = (RUNTIME_DIR / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "host supplies the NVIDIA" in text
    assert "pinned CUDA" in text
    assert "user-space libraries" in text
    assert "do not include a system CUDA toolkit" in normalized


def test_example_runtime_image_locks_are_hash_checked_and_split_by_family() -> None:
    expected = {
        "cuda-bnb": (
            "bitsandbytes==",
            (
                "compressed-tensors==",
                "gptqmodel==",
                "hqq==",
                "optimum-quanto==",
                "torchao==",
            ),
        ),
        "cuda-compressed-tensors": (
            "compressed-tensors==",
            ("bitsandbytes==", "gptqmodel==", "hqq==", "optimum-quanto==", "torchao=="),
        ),
        "cuda-hqq": (
            "hqq==",
            (
                "bitsandbytes==",
                "compressed-tensors==",
                "gptqmodel==",
                "optimum-quanto==",
                "torchao==",
            ),
        ),
        "cuda-quanto": (
            "optimum-quanto==",
            (
                "bitsandbytes==",
                "compressed-tensors==",
                "gptqmodel==",
                "hqq==",
                "torchao==",
            ),
        ),
        "cuda-torchao": (
            "torchao==",
            (
                "bitsandbytes==",
                "compressed-tensors==",
                "gptqmodel==",
                "hqq==",
                "optimum-quanto==",
            ),
        ),
        "cuda-gptqmodel": (
            "gptqmodel==",
            ("bitsandbytes==", "compressed-tensors==", "hqq==", "optimum-quanto=="),
        ),
    }

    for family, (required, forbidden) in expected.items():
        text = (REQ_DIR / f"{family}-py312-cu128.txt").read_text(encoding="utf-8")
        assert "torch==" in text
        assert "+cu128" in text
        assert "--hash=sha256:" in text
        assert required in text
        for package in forbidden:
            assert package not in text, f"{family} should not include {package}"


def test_example_runtime_images_do_not_expand_root_runtime_build_surface() -> None:
    makefile_text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    dockerfile_text = (REPO_ROOT / "runtime" / "Dockerfile").read_text(encoding="utf-8")

    for family in (
        "bnb",
        "compressed-tensors",
        "gptqmodel",
        "hqq",
        "quanto",
        "torchao",
    ):
        assert f"runtime-image-cuda-{family}" not in makefile_text
        assert f"runtime-image-{family}-py312-cu128.txt" not in dockerfile_text


def test_example_runtime_readmes_use_narrowest_matching_runtime_images() -> None:
    integrations = REPO_ROOT / "examples" / "integrations"
    expected_images = {
        "awq": "invarlock-example-runtime:cuda-gptqmodel",
        "compressed_tensors": "invarlock-example-runtime:cuda-compressed-tensors",
        "gptqmodel": "invarlock-example-runtime:cuda-gptqmodel",
        "hf_bnb": "invarlock-example-runtime:cuda-bnb",
        "hqq": "invarlock-example-runtime:cuda-hqq",
        "quanto": "invarlock-example-runtime:cuda-quanto",
        "torchao_int8_runtime": "invarlock-example-runtime:cuda-torchao",
    }

    for example, image in expected_images.items():
        text = (integrations / example / "README.md").read_text(encoding="utf-8")
        assert image in text
        assert "build_example_runtime_image.sh" in text
        assert "smoke_example_runtime_image.sh" in text

    for dense_example in ("peft_lora", "lm_eval_harness"):
        text = (integrations / dense_example / "README.md").read_text(encoding="utf-8")
        assert "invarlock-runtime:cuda-local" in text
        assert "invarlock-runtime:cuda-quant" not in text
