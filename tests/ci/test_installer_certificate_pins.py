"""Installer CA bundles must meet their own security floors, apart from certifi."""

import re
import tomllib
from pathlib import Path

import yaml
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]


def test_every_workflow_uv_selection_meets_certificate_security_floor():
    selections = []
    for path in sorted((ROOT / ".github/workflows").glob("*.yml")):
        workflow = yaml.safe_load(path.read_text())
        for job in workflow.get("jobs", {}).values():
            for step in job.get("steps", []):
                if step.get("uses", "").startswith("astral-sh/setup-uv@"):
                    version = str(step.get("with", {}).get("version", ""))
                    assert re.fullmatch(r"\d+\.\d+\.\d+", version), path
                    assert Version(version) >= Version("0.13.4"), (path, version)
                    selections.append((path, version))
    assert selections
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["tool"]["uv"]["required-version"] == ">=0.13.4"


def test_pip_locks_and_bootstrap_share_patched_artifacts():
    bootstrap = (ROOT / "requirements/workflows/pip-bootstrap.txt").read_text()
    expected = set(re.findall(r"--hash=sha256:([a-f0-9]{64})", bootstrap))
    assert (
        "fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70" in expected
    )
    found = []
    for path in (ROOT / "requirements/workflows").glob("*.txt"):
        match = re.search(
            r"(?m)^pip==([^\s]+)(?:.*\n)(?:[ \t]+.*\n)*", path.read_text()
        )
        if match:
            assert Version(match[1]) >= Version("26.2.2"), path
            assert (
                set(re.findall(r"--hash=sha256:([a-f0-9]{64})", match[0])) == expected
            ), path
            found.append(path.name)
    assert set(found) == {
        "pip-bootstrap.txt",
        "security-ci-py313.txt",
        "release-security-py313.txt",
        "k2-campaign-py312.txt",
    }
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    pip = next(p for p in lock["package"] if p["name"] == "pip")
    assert Version(pip["version"]) >= Version("26.2.2")
    assert {
        pip["sdist"]["hash"].split(":")[1],
        *(w["hash"].split(":")[1] for w in pip["wheels"]),
    } == expected
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "pip>=26.2.2" in project["dependency-groups"]["security"]
