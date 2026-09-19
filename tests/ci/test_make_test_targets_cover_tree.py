from __future__ import annotations

import subprocess
import time
from pathlib import Path

from tests._support_repository_contracts import MakefileContract

MAKE = MakefileContract.read(Path(__file__).resolve().parents[2] / "Makefile")
MAKEFILE = MAKE.text


def test_test_directories_use_one_pattern_target() -> None:
    assert "test-%:" in MAKEFILE
    assert "tests/$*" in MAKEFILE
    assert "TEST_DIR_TARGETS" not in MAKEFILE


def test_v013_compatibility_corpus_is_release_blocking() -> None:
    assert "compatibility-test:" in MAKEFILE
    assert "tests/compatibility" in MAKEFILE
    assert "test: compatibility-test" in MAKEFILE
    assert "test-fast: compatibility-test" in MAKEFILE


def test_root_tooling_has_no_legacy_product_workflows() -> None:
    forbidden = (
        "evidence-pack-v1",
        "scripts/evidence_packs",
        "scripts/model_evidence",
        "guard-validation-smoke",
        "architecture-fragmentation-check",
        "empirical-guard-inventory-check",
        "eval-loop",
        "--edit-config",
        "runtime-image-gguf",
        "runtime-image-tensorrt-llm",
        "runtime-image-cuda-quant",
    )
    assert [value for value in forbidden if value in MAKEFILE] == []


def test_runtime_provider_images_use_the_consolidated_root_context() -> None:
    assert "gguf_runtime_blackbox.py" not in MAKEFILE
    assert "tensorrt_llm_runtime_fixture.py" not in MAKEFILE
    assert "runtime-image-gguf" not in MAKEFILE
    assert "runtime-image-tensorrt-llm" not in MAKEFILE
    assert (Path(__file__).resolve().parents[2] / "runtime/Dockerfile").is_file()


def test_first_party_runtime_modules_share_test_and_distribution_gates() -> None:
    assert "runtime-test:" in MAKEFILE
    assert "install-smoke:" in MAKEFILE
    assert "tests/diagnostics" in MAKEFILE
    assert "tests/runtime" in MAKEFILE
    assert "tests/judge_measurements" in MAKEFILE
    install_smoke = MAKE.target("install-smoke").text
    assert "RELEASE_INSTALL_RELEASE_LOCK" in install_smoke
    assert "RELEASE_INSTALL_OPTIONAL_LOCK" in install_smoke
    assert "--require-hashes" in install_smoke
    assert (
        'mktemp -d "$${TMPDIR:-/tmp}/invarlock-install-smoke.XXXXXX"' in install_smoke
    )
    assert (
        'cleanup_smoke_venv() { rm -rf "$$smoke_venv" "$$optional_venv"; }'
        in install_smoke
    )
    assert "trap cleanup_smoke_venv EXIT" in install_smoke
    assert "trap 'exit 129' HUP" in install_smoke
    assert "trap 'exit 130' INT" in install_smoke
    assert "trap 'exit 143' TERM" in install_smoke
    core_install = (
        "PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH= "
        '"$$smoke_venv/bin/python" -m pip install '
        "--no-deps --force-reinstall dist/*.whl"
    )
    assert core_install in install_smoke
    assert install_smoke.index(core_install) < install_smoke.index(
        "scripts/release/core_wheel_consumers.py"
    )
    assert "-m pip check" in install_smoke
    assert "PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH=" in install_smoke
    assert "CoreRegistry" not in install_smoke
    assert install_smoke.count("scripts/release/core_wheel_consumers.py") == 2
    assert "--mode optional" in install_smoke
    assert (
        "hf_vision_text_conformance"
        not in install_smoke.split('$(PYTHON) -m venv "$$optional_venv"', 1)[0]
    )
    assert ".install-smoke-site" not in install_smoke

    clean = MAKE.target("clean").text
    assert "src/*.egg-info" in clean


def test_install_smoke_signal_trap_exits_and_cleans(tmp_path: Path) -> None:
    smoke_venv = tmp_path / "smoke-venv"
    resumed = tmp_path / "resumed"
    smoke_venv.mkdir()
    script = f"""
        smoke_venv='{smoke_venv}'
        cleanup_smoke_venv() {{ rm -rf "$smoke_venv"; }}
        trap cleanup_smoke_venv EXIT
        trap 'exit 129' HUP
        trap 'exit 130' INT
        trap 'exit 143' TERM
        : > "$smoke_venv/ready"
        while :; do sleep 0.05; done
        : > '{resumed}'
    """
    process = subprocess.Popen(["/bin/sh", "-c", script])
    try:
        deadline = time.monotonic() + 5
        while not (smoke_venv / "ready").exists():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        process.terminate()
        assert process.wait(timeout=5) == 143
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    assert not smoke_venv.exists()
    assert not resumed.exists()


def test_container_smoke_explicitly_enables_the_gated_integration_test() -> None:
    block = MAKE.target("container-front-door-smoke").text

    assert "INVARLOCK_RUN_CONTAINER_SMOKE=1" in block
    assert "INVARLOCK_CONTAINER_ENGINE=$(CONTAINER_ENGINE)" in block
    assert "INVARLOCK_RUNTIME_IMAGE=$(RUNTIME_IMAGE)" in block
    assert "tests/integration/test_container_front_door_journey.py" in block


def test_standard_tools_own_general_repository_gates() -> None:
    for tool in (
        "ruff",
        "mypy",
        "pytest",
        "mkdocs",
        "markdownlint-cli2",
        "cspell",
        "actionlint",
        "python -m build",
        "python -m twine",
    ):
        assert tool in MAKEFILE.lower()
