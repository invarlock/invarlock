from __future__ import annotations

import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_FILES = (
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / "addins/diagnostics/pyproject.toml",
    REPO_ROOT / "addins/gguf/pyproject.toml",
    REPO_ROOT / "addins/multimodal/pyproject.toml",
    REPO_ROOT / "addins/tensorrt_llm/pyproject.toml",
)
RELEASE_INPUT = REPO_ROOT / "requirements/workflows/release-install.in"
RELEASE_LOCKS = (
    REPO_ROOT / "requirements/workflows/release-install-py312.txt",
    REPO_ROOT / "requirements/workflows/release-install-py313.txt",
)
PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)")


def _project(path: Path) -> dict[str, object]:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    project = payload["project"]
    assert isinstance(project, dict)
    return project


def _fingerprint(requirement: Requirement) -> tuple[tuple[str, ...], str, str | None]:
    return (
        tuple(sorted(requirement.extras)),
        str(requirement.specifier),
        str(requirement.marker) if requirement.marker is not None else None,
    )


def _external_base_dependencies() -> dict[str, tuple[tuple[str, ...], str, str | None]]:
    projects = [_project(path) for path in PROJECT_FILES]
    first_party_names = {
        canonicalize_name(str(project["name"])) for project in projects
    }
    dependencies: dict[str, tuple[tuple[str, ...], str, str | None]] = {}

    for project in projects:
        declared = project.get("dependencies", [])
        assert isinstance(declared, list)
        for value in declared:
            requirement = Requirement(str(value))
            name = canonicalize_name(requirement.name)
            if name in first_party_names:
                continue
            fingerprint = _fingerprint(requirement)
            previous = dependencies.setdefault(name, fingerprint)
            assert previous == fingerprint, (
                f"conflicting first-party base requirements for {name}: "
                f"{previous!r} != {fingerprint!r}"
            )
    return dependencies


def _release_input_dependencies() -> dict[str, tuple[tuple[str, ...], str, str | None]]:
    dependencies: dict[str, tuple[tuple[str, ...], str, str | None]] = {}
    for raw_line in RELEASE_INPUT.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        name = canonicalize_name(requirement.name)
        assert name not in dependencies, f"duplicate release dependency: {name}"
        dependencies[name] = _fingerprint(requirement)
    return dependencies


def _release_lock_pins(path: Path) -> dict[str, Version]:
    pins: dict[str, Version] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = PIN.match(line)
        if match is None:
            continue
        name = canonicalize_name(match.group(1))
        assert name not in pins, f"duplicate release lock pin: {name}"
        pins[name] = Version(match.group(2))
    return pins


def test_release_input_equals_every_first_party_external_base_dependency() -> None:
    expected = _external_base_dependencies()
    actual = _release_input_dependencies()

    assert actual == expected
    assert "pillow" in actual


def test_release_lock_contains_compatible_pins_for_the_dependency_closure() -> None:
    expected = _external_base_dependencies()

    for path in RELEASE_LOCKS:
        pins = _release_lock_pins(path)
        for name, (_, specifier, marker) in expected.items():
            assert marker is None, (
                f"release input marker requires explicit lock handling: {name}"
            )
            assert name in pins, (
                f"{path.name} is missing first-party dependency: {name}"
            )
            assert pins[name] in Requirement(f"{name}{specifier}").specifier

        lock = path.read_text(encoding="utf-8")
        assert "--hash=sha256:" in lock
        assert "pillow==" in lock


def test_release_lock_refresh_uses_the_audited_dependency_input() -> None:
    refresh = (REPO_ROOT / "scripts/security/refresh_pinned_requirements.sh").read_text(
        encoding="utf-8"
    )

    release_function = refresh.split("compile_release_install()", 1)[1].split(
        "run_workflow_locks()", 1
    )[0]
    assert "requirements/workflows/release-install.in" in release_function
    assert "addins/diagnostics/pyproject.toml" not in release_function
