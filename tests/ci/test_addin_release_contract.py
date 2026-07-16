from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ADDINS = {
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


def test_provider_addins_require_the_matching_core_release_line() -> None:
    core_version = str(_project(REPO_ROOT)["version"])
    major, minor, _patch = (int(part) for part in core_version.split("."))
    expected = f"invarlock>={core_version},<{major}.{minor + 1}"

    for name in ("gguf", "multimodal", "tensorrt_llm"):
        dependencies = _project(ADDINS[name])["dependencies"]
        assert isinstance(dependencies, list)
        assert dependencies == [expected]

    multimodal_project = _project(ADDINS["multimodal"])
    multimodal_dependencies = multimodal_project["optional-dependencies"]
    assert isinstance(multimodal_dependencies, dict)
    multimodal_runtime = multimodal_dependencies["runtime"]
    assert isinstance(multimodal_runtime, list)
    assert any(str(item).startswith("pillow>=") for item in multimodal_runtime)
    assert any(str(item).startswith("protobuf>=") for item in multimodal_runtime)
    assert any(str(item).startswith("sentencepiece>=") for item in multimodal_runtime)
    assert any(str(item).startswith("tiktoken>=") for item in multimodal_runtime)
    assert any(str(item).startswith("torch>=") for item in multimodal_runtime)
    assert any(str(item).startswith("transformers>=") for item in multimodal_runtime)


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
