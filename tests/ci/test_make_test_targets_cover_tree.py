from __future__ import annotations

from pathlib import Path

MAKEFILE = (Path(__file__).resolve().parents[2] / "Makefile").read_text(
    encoding="utf-8"
)


def test_test_directories_use_one_pattern_target() -> None:
    assert "test-%:" in MAKEFILE
    assert "tests/$*" in MAKEFILE
    assert "TEST_DIR_TARGETS" not in MAKEFILE


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


def test_optional_provider_runtime_images_stay_outside_root_tooling() -> None:
    assert "gguf_runtime_blackbox.py" not in MAKEFILE
    assert "tensorrt_llm_runtime_fixture.py" not in MAKEFILE
    assert "runtime-image-gguf" not in MAKEFILE
    assert "runtime-image-tensorrt-llm" not in MAKEFILE
    for addin in ("gguf", "tensorrt_llm"):
        assert (
            Path(__file__).resolve().parents[2] / "addins" / addin / "Makefile"
        ).is_file()


def test_first_party_addins_share_test_and_distribution_gates() -> None:
    assert "addins-test:" in MAKEFILE
    assert "addins-install-smoke:" in MAKEFILE
    for path in (
        "addins/diagnostics",
        "addins/gguf",
        "addins/multimodal",
        "addins/tensorrt_llm",
    ):
        assert path in MAKEFILE
    install_smoke = MAKEFILE.split("addins-install-smoke:", 1)[1].split(
        "packaging-smoke-minimal:", 1
    )[0]
    assert "CoreRegistry" in install_smoke
    assert "get_runtime_provider" in install_smoke
    assert "get_plugin_info" in install_smoke
    assert "ADDINS_SMOKE_RELEASE_LOCK" in install_smoke
    assert "--require-hashes" in install_smoke
    assert 'mktemp -d "$${TMPDIR:-/tmp}/invarlock-addins-smoke.XXXXXX"' in install_smoke
    assert "trap 'rm -rf \"$$smoke_venv\"' EXIT HUP INT TERM" in install_smoke
    assert (
        "PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH= "
        '"$$smoke_venv/bin/python" -m pip install '
        "--no-deps --force-reinstall dist/*.whl dist/addins/*.whl" in install_smoke
    )
    assert "-m pip check" in install_smoke
    assert "PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH=" in install_smoke
    assert ".addins-smoke-site" not in install_smoke

    clean = MAKEFILE.split("clean:", 1)[1].split("docsclean:", 1)[0]
    assert "src/*.egg-info" in clean


def test_container_smoke_explicitly_enables_the_gated_integration_test() -> None:
    block = MAKEFILE.split("container-front-door-smoke:", 1)[1].split(
        "##@ Verification", 1
    )[0]

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
