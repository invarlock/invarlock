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

    for name in ("gguf", "tensorrt_llm"):
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
    multimodal_dependencies = multimodal_project["optional-dependencies"]
    assert isinstance(multimodal_dependencies, dict)
    multimodal_runtime = multimodal_dependencies["runtime"]
    assert isinstance(multimodal_runtime, list)
    assert not any(str(item).startswith("pillow>=") for item in multimodal_runtime)
    assert any(str(item).startswith("protobuf>=") for item in multimodal_runtime)
    assert any(str(item).startswith("sentencepiece>=") for item in multimodal_runtime)
    assert any(str(item).startswith("tiktoken>=") for item in multimodal_runtime)
    assert any(str(item).startswith("torch>=") for item in multimodal_runtime)
    assert any(str(item).startswith("torchvision>=") for item in multimodal_runtime)
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
