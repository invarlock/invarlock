"""Keep SDK qualification isolated, hash-pinned, required, and measured."""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_langfuse_sdk_gate_uses_two_installed_environments_and_hashed_closures():
    gate = (ROOT / "scripts/langfuse_sdk_gate.sh").read_text()
    assert "for environment in sdk recipient" in gate
    assert gate.count("pip install --require-hashes") == 3
    assert 'pip install --no-index "${core_wheels[0]}"' in gate
    assert "pip check" in gate
    assert "unset PYTHONPATH" in gate
    assert 'find_spec("langfuse") is None' in gate
    assert "INVARLOCK_REQUIRE_LANGFUSE_SDK=1" in gate
    assert (
        'INVARLOCK_LANGFUSE_WHEEL_PYTHON="${LANGFUSE_ENV}/recipient/bin/python"' in gate
    )
    assert "tests/examples/test_langfuse_export.py" in gate
    assert "tests/integration/test_langfuse_sdk.py" in gate
    assert "langfuse-sdk-test: dist-check" in (ROOT / "Makefile").read_text()
    for tag in ("312", "313"):
        lock = (
            ROOT / f"requirements/workflows/langfuse-sdk-tests-py{tag}.txt"
        ).read_text()
        assert "langfuse==4.14.1 \\" in lock
        assert "pytest==9.1.1 \\" in lock
        assert "--hash=sha256:" in lock
    refresh = (ROOT / "scripts/security/refresh_pinned_requirements.sh").read_text()
    assert "langfuse-sdk-tests.in" in refresh
    assert "langfuse-sdk-tests-py${langfuse_python/./}.txt" in refresh
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    published = project["dependencies"] + [
        entry
        for group in project.get("optional-dependencies", {}).values()
        for entry in group
    ]
    assert not any("langfuse" in entry.lower() for entry in published)


def test_langfuse_sdk_is_required_only_for_examples_coverage_and_installed_gates():
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    for job in ("verify-fast", "minimum-python-packages", "verify-full"):
        assert any(
            "make install-smoke inspect-judge-sdk-test langfuse-sdk-test"
            == step.get("run")
            for step in jobs[job]["steps"]
        )
    steps = jobs["coverage-tests"]["steps"]
    install = [
        step
        for step in steps
        if "pip install" in step.get("run", "")
        and "langfuse-sdk-tests-py313.txt" in step["run"]
    ]
    assert len(install) == 1
    assert install[0]["if"] == "${{ startsWith(matrix.shard, 'examples') }}"
    assert install[0]["run"].strip().endswith("python -m pip check")
    collect = next(
        step for step in steps if "make coverage-collect-" in step.get("run", "")
    )
    assert steps.index(install[0]) < steps.index(collect)
    assert (
        collect["env"]["INVARLOCK_REQUIRE_LANGFUSE_SDK"]
        == "${{ startsWith(matrix.shard, 'examples') && '1' || '0' }}"
    )
    full = jobs["verify-full"]["steps"]
    verify = next(step for step in full if step.get("run") == "make verify")
    assert verify["env"]["INVARLOCK_REQUIRE_LANGFUSE_SDK"] == "1"
