from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")


def _target(name: str, next_name: str) -> str:
    return MAKEFILE.split(f"{name}:", 1)[1].split(f"{next_name}:", 1)[0]


def test_coverage_uses_pytest_cov_without_custom_policy_code() -> None:
    block = _target("coverage", "coverage-enforce")
    assert "--cov=src/invarlock" in block
    assert "--cov-branch" in block
    assert "--cov-fail-under=90" in block
    assert "check_coverage_thresholds.py" not in MAKEFILE
    assert "scripts/evidence_packs" not in MAKEFILE


def test_coverage_configuration_is_the_core_surface() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    run = pyproject["tool"]["coverage"]["run"]
    report = pyproject["tool"]["coverage"]["report"]
    assert run["source"] == ["src/invarlock"]
    assert report["fail_under"] == 90
    assert "include" not in report


def test_fast_and_parallel_lanes_share_the_same_selector() -> None:
    assert "test-parallel: PYTEST_WORKERS = auto" in MAKEFILE
    assert "$(MAKE) test-fast PYTEST_WORKERS=$(PYTEST_WORKERS)" in MAKEFILE
    assert "not integration and not slow and not manual and not gpu" in MAKEFILE
    assert "coverage-enforce-parallel: PYTEST_WORKERS = auto" in MAKEFILE


def test_primary_verification_and_coverage_targets_default_to_parallel() -> None:
    assert "verify: PYTEST_WORKERS = auto" in MAKEFILE
    assert "verify-fast: PYTEST_WORKERS = auto" in MAKEFILE
    assert "coverage-enforce: PYTEST_WORKERS = auto" in MAKEFILE
    assert "$(MAKE) test PYTEST_WORKERS=$(PYTEST_WORKERS)" in MAKEFILE
    assert "$(MAKE) coverage PYTEST_WORKERS=$(PYTEST_WORKERS)" in MAKEFILE
    assert "$(PYTEST) $(PYTEST_WORKER_ARGS) -q" in MAKEFILE


def test_verify_fast_never_requires_a_container() -> None:
    block = _target("verify-fast", "contracts-check")
    assert "docker" not in block.lower()
    assert "podman" not in block.lower()
    assert "container-front-door-smoke" not in block


def test_makefile_exposes_only_the_supported_cli_smoke() -> None:
    block = _target("cli-smoke-core", "hf-provider-smoke")
    assert "invarlock evaluate --help" in block
    assert "invarlock verify --help" in block
    assert "invarlock report --help" in block
    assert "invarlock advanced" not in MAKEFILE
    assert "cli-smoke-advanced" not in MAKEFILE
