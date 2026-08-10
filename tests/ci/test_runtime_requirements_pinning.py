from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_REQUIREMENTS = ROOT / "requirements" / "workflows"
REFRESH_SCRIPT = ROOT / "scripts" / "security" / "refresh_pinned_requirements.sh"


def _locked_package_version(lock: dict[str, Any], package_name: str) -> str:
    packages = lock.get("package")
    assert isinstance(packages, list)
    matches = [
        package.get("version")
        for package in packages
        if isinstance(package, dict) and package.get("name") == package_name
    ]
    assert len(matches) == 1
    version = matches[0]
    assert isinstance(version, str)
    return version


def _hashed_requirement_version(path: Path, package_name: str) -> str:
    prefix = f"{package_name}=="
    matches = [
        line.removeprefix(prefix).removesuffix(" \\")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith(prefix)
    ]
    assert len(matches) == 1
    return matches[0]


def test_refresh_pinned_requirements_generates_canonical_runtime_locks() -> None:
    text = REFRESH_SCRIPT.read_text(encoding="utf-8")

    assert (
        '"${WORKFLOW_DIR}/runtime-image.in" \\\n'
        '    "${WORKFLOW_DIR}/runtime-image-py312.txt"'
    ) in text
    assert (
        '"${WORKFLOW_DIR}/runtime-image.in" \\\n'
        '    "${WORKFLOW_DIR}/runtime-image-py312-aarch64.txt"'
    ) in text
    assert (
        '"${WORKFLOW_DIR}/runtime-image-cu126.in" \\\n'
        '    "${WORKFLOW_DIR}/runtime-image-py312-cu126.txt"'
    ) in text
    assert text.count("--torch-backend cpu") == 5
    assert text.count("--torch-backend cu126") == 2
    assert (
        '"${WORKFLOW_DIR}/multimodal-runtime.in" \\\n'
        '    "${WORKFLOW_DIR}/multimodal-runtime-py312.txt"'
    ) in text
    assert (
        '"${WORKFLOW_DIR}/lm-evaluation-harness.in" \\\n    "${harness_full_lock}"'
    ) in text
    assert "build_cache_free_lm_eval_wheel.py" in text
    assert "filter-lock" in text
    assert '--output "${WORKFLOW_DIR}/lm-evaluation-harness-py312.txt"' in text
    assert "--constraints requirements/workflows/runtime-image.in" in text
    assert "--no-deps" in text


def test_type_checker_version_is_identical_in_local_and_workflow_locks() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional = project["project"]["optional-dependencies"]
    declared = {
        requirement.removeprefix("mypy==")
        for group in ("dev", "ci")
        for requirement in optional[group]
        if requirement.startswith("mypy==")
    }
    assert len(declared) == 1

    uv_version = _locked_package_version(
        tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8")), "mypy"
    )
    workflow_versions = {
        _hashed_requirement_version(
            WORKFLOW_REQUIREMENTS / f"ci-hf-py{python_tag}.txt", "mypy"
        )
        for python_tag in ("312", "313")
    }
    assert {uv_version, *workflow_versions} == declared


@pytest.mark.parametrize(
    ("lock_name", "extra_names"),
    (
        ("core-py312.txt", ()),
        ("hf-py313.txt", ("hf",)),
        ("ci-hf-py312.txt", ("hf", "ci")),
        ("ci-hf-py313.txt", ("hf", "ci")),
        ("docs-ci-py313.txt", ("docs-ci",)),
        ("precommit-ci-py313.txt", ("precommit-ci",)),
        ("release-security-py313.txt", ("release-ci", "security-ci")),
        ("security-ci-py313.txt", ("security-ci",)),
    ),
)
def test_workflow_locks_satisfy_every_declared_direct_requirement(
    lock_name: str,
    extra_names: tuple[str, ...],
) -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    requirements = list(project["dependencies"])
    optional = project["optional-dependencies"]
    for extra_name in extra_names:
        requirements.extend(optional[extra_name])

    lock_path = WORKFLOW_REQUIREMENTS / lock_name
    for raw_requirement in requirements:
        requirement = Requirement(raw_requirement)
        locked_version = Version(
            _hashed_requirement_version(lock_path, requirement.name.lower())
        )
        assert locked_version in requirement.specifier, (
            f"{lock_name} pins {requirement.name}=={locked_version}, which does not "
            f"satisfy the declared requirement {requirement}"
        )


def test_refresh_surface_excludes_retired_runtime_profiles() -> None:
    text = REFRESH_SCRIPT.read_text(encoding="utf-8")
    retired = (
        "advanced-py313",
        "assurance-ci",
        "evidence-packs",
        "runtime-image-quant",
        "training-profile",
    )

    for marker in retired:
        assert marker not in text


def test_docs_lock_excludes_the_model_runtime_test_stack() -> None:
    text = REFRESH_SCRIPT.read_text(encoding="utf-8")
    docs_compile = text.split(
        'compile_pyproject "${WORKFLOW_DIR}/docs-ci-py313.txt"', 1
    )[1].split("\n\n", 1)[0]

    assert "--extra docs-ci" in docs_compile
    assert "--extra ci" not in docs_compile

    lock = (WORKFLOW_REQUIREMENTS / "docs-ci-py313.txt").read_text(encoding="utf-8")
    assert "linkchecker==" in lock
    for package in ("peft==", "torch==", "torchao==", "transformers=="):
        assert package not in lock


def test_runtime_image_locks_are_cpu_only() -> None:
    runtime_locks = (
        "runtime-image-py312.txt",
        "runtime-image-py312-aarch64.txt",
    )

    for filename in runtime_locks:
        text = (WORKFLOW_REQUIREMENTS / filename).read_text(encoding="utf-8")
        assert "torch==2.13.0+cpu" in text
        assert "+cu" not in text
        assert "cu13" not in text


def test_cuda_runtime_image_lock_is_separate_and_backend_pinned() -> None:
    cuda_input = WORKFLOW_REQUIREMENTS / "runtime-image-cu126.in"
    assert cuda_input.is_file()
    input_text = cuda_input.read_text(encoding="utf-8")
    assert "torch==2.13.0" in input_text
    assert "torch==2.11.0" not in input_text

    text = (WORKFLOW_REQUIREMENTS / "runtime-image-py312-cu126.txt").read_text(
        encoding="utf-8"
    )

    assert "torch==2.13.0+cu126" in text
    assert "nvidia-cuda-runtime-cu12==" in text
    assert "--hash=sha256:" in text
    assert "bitsandbytes==" not in text
    assert "gptqmodel==" not in text


def test_runtime_wheel_build_lock_is_retained() -> None:
    lock = WORKFLOW_REQUIREMENTS / "runtime-wheel-build-py312.txt"

    assert lock.is_file()
    assert "--hash=sha256:" in lock.read_text(encoding="utf-8")


def test_multimodal_runtime_lock_pins_cuda_matched_torchvision() -> None:
    text = (WORKFLOW_REQUIREMENTS / "multimodal-runtime-py312.txt").read_text(
        encoding="utf-8"
    )

    assert "torchvision==0.28.0+cu126" in text
    assert "pillow==12.3.0" in text
    assert "torch==" not in text
    assert "--hash=sha256:" in text


def test_lm_evaluation_harness_lock_is_complete_and_cpu_aligned() -> None:
    text = (WORKFLOW_REQUIREMENTS / "lm-evaluation-harness-py312.txt").read_text(
        encoding="utf-8"
    )

    assert "lm-eval==" not in text
    assert "sqlitedict==" not in text
    assert "torch==2.13.0+cpu" in text
    assert "transformers==5.14.1" in text
    assert "+cu" not in text
    assert "--hash=sha256:" in text

    upstream = (
        WORKFLOW_REQUIREMENTS / "lm-evaluation-harness-upstream-wheel.txt"
    ).read_text(encoding="utf-8")
    assert "lm-eval==0.4.12" in upstream
    assert (
        "02971ff68284dd14cfa7fce9310a58452c4162e8d413ba96aa7988a0ff9352ef" in upstream
    )


def test_declared_typer_floor_matches_the_maintained_runtime_version() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    dependencies = project["dependencies"]
    assert isinstance(dependencies, list)
    typer_requirement = next(
        dependency for dependency in dependencies if dependency.startswith("typer")
    )
    runtime_input = (WORKFLOW_REQUIREMENTS / "runtime-image.in").read_text(
        encoding="utf-8"
    )
    locked_typer = next(
        line.removeprefix("typer==")
        for line in runtime_input.splitlines()
        if line.startswith("typer==")
    )

    assert typer_requirement == f"typer>={locked_typer}"


def test_refresh_pinned_requirements_help_is_side_effect_free() -> None:
    result = subprocess.run(
        ["bash", str(REFRESH_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--check" in result.stdout
    assert "all or workflows" in result.stdout
    assert "uv pip compile" not in result.stderr
