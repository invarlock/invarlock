from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ADDINS = {
    "inspect_judge": REPO_ROOT / "addins/inspect_judge",
    "diagnostics": REPO_ROOT / "addins/diagnostics",
    "gguf": REPO_ROOT / "addins/gguf",
    "multimodal": REPO_ROOT / "addins/multimodal",
    "tensorrt_llm": REPO_ROOT / "addins/tensorrt_llm",
}


def _project(path: Path) -> dict[str, object]:
    payload = tomllib.loads((path / "pyproject.toml").read_text(encoding="utf-8"))
    project = payload["project"]
    assert isinstance(project, dict)
    return project


def _module_version(path: Path, package: str) -> str:
    source = (path / "src/invarlock_addins" / package / "__init__.py").read_text(
        encoding="utf-8"
    )
    match = re.search(r'^__version__ = "([^"]+)"$', source, flags=re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_first_party_distribution_versions_match_core() -> None:
    core_version = str(_project(REPO_ROOT)["version"])

    for name, path in ADDINS.items():
        assert _project(path)["version"] == core_version
        assert _module_version(path, name) == core_version


def test_all_addins_ship_the_declared_license_text() -> None:
    expected = (REPO_ROOT / "LICENSE").read_bytes()
    for path in ADDINS.values():
        assert _project(path)["license-files"] == ["LICENSE"]
        assert (path / "LICENSE").read_bytes() == expected


def test_local_distribution_gate_validates_all_first_party_source_parity() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    dist_check = makefile.split("dist-check:", 1)[1].split("addins-install-smoke:", 1)[
        0
    ]

    assert "first_party_distribution_validation.py" in dist_check
    assert "--core-dist-dir dist" in dist_check
    assert "--addin-dist-dir dist/addins" in dist_check


def test_provider_addins_require_the_exact_matching_core_release() -> None:
    core_version = str(_project(REPO_ROOT)["version"])
    expected = f"invarlock=={core_version}"

    for name in ("gguf", "tensorrt_llm", "inspect_judge"):
        dependencies = _project(ADDINS[name])["dependencies"]
        assert isinstance(dependencies, list)
        assert dependencies == [expected]

    multimodal_project = _project(ADDINS["multimodal"])
    multimodal_base = multimodal_project["dependencies"]
    assert isinstance(multimodal_base, list)
    assert multimodal_base[0] == expected
    assert any(str(item).startswith("pillow>=") for item in multimodal_base)
    assert not any(
        str(item).startswith(("torch", "transformers")) for item in multimodal_base
    )
    assert "runtime" not in multimodal_project.get("optional-dependencies", {})


def test_multimodal_smoke_exercises_the_current_runtime_surface() -> None:
    makefile = (ADDINS["multimodal"] / "Makefile").read_text(encoding="utf-8")

    for expected in (
        "import accelerate, safetensors, torch, torchvision, transformers",
        "accelerate.__version__ == '1.14.0+invarlock.1'",
        "safetensors.__version__ == '0.8.0'",
        "transformers.__version__ == '5.14.1'",
        "_resolve_vision_text_model_loader",
        "transformers.AutoModelForMultimodalLM",
        "{'gemma4', 'qwen3_5'} <= model_types",
    ):
        assert expected in makefile


def test_provider_images_expose_the_invarlock_front_door() -> None:
    gguf = (ADDINS["gguf"] / "runtime/Dockerfile").read_text(encoding="utf-8")
    multimodal = (ADDINS["multimodal"] / "runtime/Dockerfile").read_text(
        encoding="utf-8"
    )
    tensorrt = (ADDINS["tensorrt_llm"] / "runtime/Dockerfile").read_text(
        encoding="utf-8"
    )

    assert 'ENTRYPOINT ["python", "-m", "invarlock"]' in gguf
    assert 'ENTRYPOINT ["python", "-m", "invarlock"]' in multimodal
    assert (
        'ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", '
        '"/opt/invarlock/cli-venv/bin/python", "-m", "invarlock"]' in tensorrt
    )
    assert 'CMD ["python", "-m", "invarlock"]' not in tensorrt


def test_inspect_judge_keeps_the_sdk_optional_and_joins_installed_smokes() -> None:
    project = _project(ADDINS["inspect_judge"])
    assert project["optional-dependencies"] == {
        "inspect": ["inspect-ai==0.3.263", "openai==3.13.0", "httpx==0.28.1"]
    }
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "-p invarlock_addins.inspect_judge" in makefile
    for text in (
        makefile,
        (REPO_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8"),
    ):
        assert "import invarlock_addins.inspect_judge as judge" in text
        assert "version('invarlock-inspect-judge') == judge.__version__" in text
        assert "'inspect_ai' not in sys.modules" in text


def test_optional_judge_sdk_has_hashed_release_gate_without_source_shadowing() -> None:
    gate = (REPO_ROOT / "scripts/inspect_judge_sdk_gate.sh").read_text()
    assert '"${JUDGE_BIN}" -m pip install --no-index' in gate
    assert '"${judge_wheels[0]}[inspect]"' in gate
    assert '"${JUDGE_BIN}" -m pip check' in gate
    assert "unset PYTHONPATH" in gate
    assert "INVARLOCK_REQUIRE_INSPECT_SDK=1" in gate
    assert 'cd "${JUDGE_ENV}"' in gate
    for version in ("312", "313"):
        lock = (
            REPO_ROOT / f"requirements/workflows/inspect-judge-tests-py{version}.txt"
        ).read_text()
        for dependency in ("inspect-ai==0.3.263", "openai==3.13.0", "httpx==0.28.1"):
            assert dependency + " \\" in lock
        assert "--hash=sha256:" in lock
    refresh = (
        REPO_ROOT / "scripts/security/refresh_pinned_requirements.sh"
    ).read_text()
    assert "inspect-judge-tests.in" in refresh
    assert "inspect-judge-tests-py${judge_python/./}.txt" in refresh
    ci = (REPO_ROOT / ".github/workflows/ci.yml").read_text()
    release = (REPO_ROOT / ".github/workflows/release.yml").read_text()
    assert "make addins-install-smoke inspect-judge-sdk-test" in ci
    assert "bash scripts/inspect_judge_sdk_gate.sh" in release


def test_runtime_image_build_contexts_include_declared_addin_licenses() -> None:
    dockerignore = (REPO_ROOT / ".dockerignore").read_text()
    for name in ("gguf", "multimodal", "tensorrt_llm"):
        dockerfile = (ADDINS[name] / "runtime/Dockerfile").read_text()
        assert f"addins/{name}/LICENSE /addin/" in dockerfile
        assert f"!addins/{name}/LICENSE" in dockerignore
