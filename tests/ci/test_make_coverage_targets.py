from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")
QUALIFICATION_COVERAGE_CONFIG = (
    ROOT / "scripts" / "qualification.coveragerc"
).read_text(encoding="utf-8")


def _target(name: str, next_name: str) -> str:
    return MAKEFILE.split(f"{name}:", 1)[1].split(f"{next_name}:", 1)[0]


def test_coverage_uses_pytest_cov_with_an_individual_file_ratchet() -> None:
    block = _target("coverage", "coverage-addins")
    assert "--cov=src/invarlock" in block
    assert "--cov-branch" in block
    assert "--cov-fail-under=90" in block
    assert "git ls-files 'src/invarlock/**/*.py' 'src/invarlock/*.py'" in block
    assert "grep -v '/__init__.py$$'" in block
    assert '--include="$$source" --fail-under=80' in block
    assert "check_coverage_thresholds.py" not in MAKEFILE
    assert "scripts/evidence_packs" not in MAKEFILE


def test_addin_coverage_has_a_separate_parallel_ratchet() -> None:
    block = _target("coverage-addins", "coverage-qualification")
    for package in ("diagnostics", "gguf", "multimodal", "tensorrt_llm"):
        assert f"--cov=addins/{package}/src/invarlock_addins/{package}" in block
        assert f"--include='addins/{package}/src/*'" in block
    assert "--cov-branch" in block
    assert "--cov-fail-under=80" in block
    assert block.count("--fail-under=80") == 5
    assert "git ls-files 'addins/*/src/**/*.py'" in block
    assert "grep -v '/__init__.py$$'" in block
    assert '--include="$$source" --fail-under=80' in block
    assert "ADDIN_COVERAGE_MIN" not in MAKEFILE
    assert "coverage-addins: coverage-linux-check" in MAKEFILE
    assert 'test "$$(uname -s)" = Linux' in MAKEFILE


def test_qualification_scripts_have_an_individual_branch_coverage_ratchet() -> None:
    block = _target("coverage-qualification", "coverage-release")
    for script in (
        "authenticated_runtime_build.py",
        "qualification_candidate_wheels.py",
        "qualification_precheck.py",
        "qualification_receipt_check.py",
        "qualification_source.py",
        "runtime_qualification.py",
        "tensorrt_llm_canary_preflight.py",
    ):
        assert f"--include='scripts/{script}' --fail-under=80" in block
        assert f"scripts/{script}" in QUALIFICATION_COVERAGE_CONFIG
    assert "addins/tensorrt_llm/tests/test_tensorrt_llm_canary_preflight.py" in block
    assert "PYTHONPATH=src:addins/tensorrt_llm/src" in block
    assert "--cov-config=scripts/qualification.coveragerc" in block
    assert "--cov-branch" in block
    assert "$(PYTEST_WORKER_ARGS)" in block


def test_release_helpers_have_an_individual_branch_coverage_ratchet() -> None:
    block = _target("coverage-release", "coverage-examples")
    for script in (
        "first_party_distribution_validation.py",
        "release_distribution_validation.py",
        "release_preflight.py",
        "testpypi_promotion.py",
        "verify_hosted_distributions.py",
    ):
        assert f"--include='scripts/release/{script}' --fail-under=80" in block
    assert "--cov-config=scripts/release.coveragerc" in block
    assert "--cov-branch" in block
    assert "$(PYTEST_WORKER_ARGS)" in block


def test_example_launchers_have_an_individual_branch_coverage_ratchet() -> None:
    block = _target("coverage-examples", "coverage-enforce")
    assert "tests/examples" in block
    assert "--cov=examples.integrations.launch" in block
    assert "--cov=examples.integrations.run" in block
    assert "--cov-branch" in block
    assert "--cov-fail-under=80" in block
    assert "$(PYTEST_WORKER_ARGS)" in block


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
    assert "coverage-enforce: coverage-linux-check" in MAKEFILE
    assert "$(MAKE) test PYTEST_WORKERS=$(PYTEST_WORKERS)" in MAKEFILE
    assert "$(MAKE) coverage PYTEST_WORKERS=$(PYTEST_WORKERS)" in MAKEFILE
    assert "$(MAKE) coverage-addins PYTEST_WORKERS=$(PYTEST_WORKERS)" in MAKEFILE
    assert "$(MAKE) coverage-qualification PYTEST_WORKERS=$(PYTEST_WORKERS)" in MAKEFILE
    assert "$(MAKE) coverage-release PYTEST_WORKERS=$(PYTEST_WORKERS)" in MAKEFILE
    assert "$(MAKE) coverage-examples PYTEST_WORKERS=$(PYTEST_WORKERS)" in MAKEFILE
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
