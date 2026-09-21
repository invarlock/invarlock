from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT


def _project(path: Path = REPO_ROOT) -> dict[str, object]:
    payload = tomllib.loads((path / "pyproject.toml").read_text(encoding="utf-8"))
    project = payload["project"]
    assert isinstance(project, dict)
    return project


def test_single_distribution_owns_all_first_party_provider_entry_points() -> None:
    project = _project()
    assert project["name"] == "invarlock"
    entry_points = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["entry-points"]["invarlock.runtime_providers"]
    assert set(entry_points) == {
        "hf_transformers",
        "hf_vision_text",
        "llama_cpp",
        "tensorrt_llm",
    }


def test_first_party_distribution_versions_match_core() -> None:
    project = _project()
    source = ast.parse((ROOT / "src/invarlock/__init__.py").read_text(encoding="utf-8"))
    module_version = next(
        ast.literal_eval(node.value)
        for node in source.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        )
    )
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    locked = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    root_package = next(
        package
        for package in locked["package"]
        if package["name"] == "invarlock" and package.get("source") == {"editable": "."}
    )
    assert {
        project["version"],
        module_version,
        citation["version"],
        citation["preferred-citation"]["version"],
        root_package["version"],
    } == {project["version"]}
    assert project["name"] == "invarlock"
    assert (ROOT / "packaging/README.md").is_file()


def test_judge_extra_requires_only_supported_provider_sdks_without_enlarging_core() -> (
    None
):
    core = _project(REPO_ROOT)
    assert core["optional-dependencies"]["judge"] == [
        "inspect-ai==0.3.263",
        "openai==3.13.0",
        "anthropic==1.6.0",
        "google-genai==2.24.0",
        "httpx==0.28.1",
        "httpx2==2.12.0",
    ]
    assert not any(
        str(dependency).startswith(
            ("inspect-ai", "openai", "anthropic", "google-genai", "httpx")
        )
        for dependency in core["dependencies"]
    )


def test_single_distribution_declares_the_repository_license() -> None:
    license_text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert _project()["license"] == "Apache-2.0"
    assert "Apache License" in license_text
    assert "Version 2.0" in license_text
    assert "license-files" not in _project()


def test_local_distribution_gate_validates_all_first_party_source_parity() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    dist_check = makefile.split("dist-check:", 1)[1].split(
        ".PHONY: inspect-judge-sdk-test", 1
    )[0]

    assert "first_party_distribution_validation.py" in dist_check
    assert "--core-dist-dir dist" in dist_check
    assert "dist/addins" not in dist_check


def test_optional_runtime_dependencies_remain_outside_core() -> None:
    project = _project()
    assert project["dependencies"]
    assert all(
        not str(item).startswith(
            ("inspect-ai", "openai", "anthropic", "google-genai", "httpx")
        )
        for item in project["dependencies"]
    )
    assert "hf" not in project["optional-dependencies"]


def test_multimodal_smoke_exercises_the_current_runtime_surface() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "runtime-test" in makefile
    assert "tests/runtime_providers" in makefile


def test_provider_images_expose_the_invarlock_front_door() -> None:
    for name in ("gguf", "hf-vision-text", "tensorrt-llm"):
        text = (REPO_ROOT / f"runtime/Dockerfile.{name}").read_text(encoding="utf-8")
        assert "COPY src /project/src" in text
        assert "invarlock-*.whl" in text


def test_core_collector_keeps_sdk_optional_in_installed_smokes() -> None:
    project = _project(REPO_ROOT)
    assert project["optional-dependencies"]["judge"] == [
        "inspect-ai==0.3.263",
        "openai==3.13.0",
        "anthropic==1.6.0",
        "google-genai==2.24.0",
        "httpx==0.28.1",
        "httpx2==2.12.0",
    ]
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "-p invarlock.judge_measurements" in makefile
    assert "scripts/release/core_wheel_consumers.py" in makefile
    consumers = (REPO_ROOT / "scripts/release/core_wheel_consumers.py").read_text()
    assert "import invarlock.judge_measurements as judge" in consumers
    assert (
        "callable(judge.import_export) and callable(judge.prepare_collection)"
        in consumers
    )
    assert "Path(judge.__file__).resolve().is_relative_to(site)" in consumers


def test_judge_collection_has_one_canonical_runner_module() -> None:
    package = REPO_ROOT / "src/invarlock/judge_measurements"

    assert (package / "runner.py").is_file()
    assert not (package / "collector_runner.py").exists()


def test_optional_judge_sdk_has_hashed_release_gate_without_source_shadowing() -> None:
    gate = (REPO_ROOT / "scripts/inspect_judge_sdk_gate.sh").read_text()
    assert '"${JUDGE_BIN}" -m pip install --no-index' in gate
    assert '"${core_wheels[0]}[judge]"' in gate
    assert "judge_wheels" not in gate
    assert '"${JUDGE_BIN}" -m pip check' in gate
    assert "unset PYTHONPATH" in gate
    assert "INVARLOCK_REQUIRE_INSPECT_SDK=1" in gate
    assert 'cd "${JUDGE_ENV}"' in gate
    for version in ("312", "313"):
        lock = (
            REPO_ROOT / f"requirements/workflows/inspect-judge-tests-py{version}.txt"
        ).read_text()
        for dependency in (
            "inspect-ai==0.3.263",
            "openai==3.13.0",
            "anthropic==1.6.0",
            "google-genai==2.24.0",
            "httpx==0.28.1",
            "httpx2==2.12.0",
        ):
            assert dependency + " \\" in lock
        assert "--hash=sha256:" in lock
    refresh = (
        REPO_ROOT / "scripts/security/refresh_pinned_requirements.sh"
    ).read_text()
    assert "inspect-judge-tests.in" in refresh
    assert "inspect-judge-tests-py${judge_python/./}.txt" in refresh
    ci = (REPO_ROOT / ".github/workflows/ci.yml").read_text()
    release = (REPO_ROOT / ".github/workflows/release.yml").read_text()
    assert "make install-smoke inspect-judge-sdk-test" in ci
    assert "bash scripts/inspect_judge_sdk_gate.sh" in release


def test_evaluator_parity_gate_replays_every_retained_campaign() -> None:
    gate = (REPO_ROOT / "scripts/evaluator_parity_gate.sh").read_text()
    assert "tests/integration/test_evaluator_parity.py" in gate
    assert "test_installed_sdk_free_recipient_signed_journey" in gate
    assert "INVARLOCK_REPLAY_RETAINED_CAMPAIGNS:-1" in gate
    assert 'if [[ "${PARITY_RETAINED}" = "0" ]]' in gate
    assert "mistral-7b-sentinel" in gate
    assert "priority-workflows" in gate
    for root in ("SENTINEL", "PRIORITY"):
        assert f'"${{{root}}}/replay.py"' in gate
        assert f'"${{{root}}}/judge_replay.py"' in gate


def test_runtime_image_build_contexts_include_declared_inputs() -> None:
    dockerignore = (REPO_ROOT / ".dockerignore").read_text()
    assert "!runtime/llama-completion-user-output.patch" in dockerignore
    assert "!runtime/Dockerfile.gguf" in dockerignore
    assert (
        "runtime/llama-completion-user-output.patch"
        in (REPO_ROOT / "runtime/Dockerfile.gguf").read_text()
    )
