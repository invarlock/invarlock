"""Known security floors apply to every maintained lock and its regeneration."""

import re
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]
FLOORS = {"filelock": "4.2.1", "setuptools": "84.1.0"}


@pytest.mark.parametrize("name,minimum", FLOORS.items())
def test_maintained_dependency_locks_meet_security_floor(name, minimum):
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    versions = {
        "uv.lock": next(p["version"] for p in lock["package"] if p["name"] == name)
    }
    for path in (ROOT / "requirements/workflows").glob("*.txt"):
        match = re.search(rf"(?m)^{name}==([^\s]+)", path.read_text())
        if match:
            versions[str(path.relative_to(ROOT))] = match[1]
    assert len(versions) > 1
    stale = {
        path: version
        for path, version in versions.items()
        if Version(version) < Version(minimum)
    }
    assert not stale, stale


def test_resolution_constraints_preserve_security_floors():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    expected = {f"{name}>={version}" for name, version in FLOORS.items()}
    for group in ("security", "precommit", "hf"):
        assert "filelock>=4.2.1" in project["dependency-groups"][group]
    # uv applies project constraints to hash-checked installs too, where a
    # non-exact constraint would reject an otherwise complete hashed lock.
    assert "constraint-dependencies" not in project["tool"]["uv"]
    constraints = (ROOT / "requirements/security-constraints.in").read_text()
    assert expected <= {line.strip() for line in constraints.splitlines()}
    for item in project["build-system"]["requires"]:
        requirement = Requirement(item)
        if requirement.name == "setuptools":
            assert Version("84.1.0") in requirement.specifier
            assert Version("84.0.0") not in requirement.specifier
