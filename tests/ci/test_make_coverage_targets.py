from __future__ import annotations

import configparser
import re
import subprocess
import tomllib
from collections import Counter
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
    assert '--include="$$source" --fail-under=90' in block
    assert "check_coverage_thresholds.py" not in MAKEFILE
    assert "scripts/evidence_packs" not in MAKEFILE
    assert "--fail-under=80" not in MAKEFILE
    assert "COVERAGE_FILE=$(COVERAGE_CORE_FILE)" in block


def test_addin_coverage_has_a_separate_parallel_ratchet() -> None:
    block = _target("coverage-addins", "coverage-qualification")
    config = (ROOT / "scripts" / "addins.coveragerc").read_text(encoding="utf-8")
    for package in ("diagnostics", "gguf", "multimodal", "tensorrt_llm"):
        assert f"--include='addins/{package}/src/*'" in block
    assert "--cov --cov-config=scripts/addins.coveragerc" in block
    assert "source =\n    addins" in config
    assert "addins/*/tests/*" in config
    assert "--cov-branch" in block
    assert "--cov-fail-under=90" in block
    assert block.count("--fail-under=90") == 5
    assert "git ls-files 'addins/*/src/**/*.py'" in block
    assert "grep -v '/__init__.py$$'" in block
    assert '--include="$$source" --fail-under=90' in block
    assert "ADDIN_COVERAGE_MIN" not in MAKEFILE
    assert "coverage-addins: coverage-linux-check" in MAKEFILE
    assert 'test "$$(uname -s)" = Linux' in MAKEFILE
    assert "COVERAGE_FILE=$(COVERAGE_ADDINS_FILE)" in block


def test_qualification_scripts_have_an_individual_branch_coverage_ratchet() -> None:
    block = _target("coverage-qualification", "coverage-release")
    for script in (
        "authenticated_runtime_build.py",
        "qualification_candidate_wheels.py",
        "qualification_precheck.py",
        "qualification_receipt_check.py",
        "qualification_render_preflight.py",
        "qualification_source.py",
        "runtime_qualification.py",
        "tensorrt_llm_canary_preflight.py",
    ):
        assert f"--include='scripts/{script}' --fail-under=90" in block
        assert f"scripts/{script}" in QUALIFICATION_COVERAGE_CONFIG
    assert "addins/tensorrt_llm/tests/test_tensorrt_llm_canary_preflight.py" in block
    assert "PYTHONPATH=src:addins/tensorrt_llm/src" in block
    assert "--cov-config=scripts/qualification.coveragerc" in block
    assert "--cov-branch" in block
    assert "--cov-fail-under=90" in block
    assert "$(PYTEST_WORKER_ARGS)" in block
    assert "COVERAGE_FILE=$(COVERAGE_QUALIFICATION_FILE)" in block


def test_release_helpers_have_an_individual_branch_coverage_ratchet() -> None:
    block = _target("coverage-release", "coverage-examples")
    for script in (
        "first_party_distribution_validation.py",
        "release_distribution_validation.py",
        "release_preflight.py",
        "tagged_release_candidate.py",
        "verify_hosted_distributions.py",
    ):
        assert f"--include='scripts/release/{script}' --fail-under=90" in block
    assert "--cov-config=scripts/release.coveragerc" in block
    assert "--cov-branch" in block
    assert "--cov-fail-under=90" in block
    assert "$(PYTEST_WORKER_ARGS)" in block
    assert "COVERAGE_FILE=$(COVERAGE_RELEASE_FILE)" in block


def test_example_launchers_have_an_individual_branch_coverage_ratchet() -> None:
    block = _target("coverage-examples", "coverage-maintenance")
    assert "tests/examples" in block
    assert "--cov=examples" in block
    assert "--cov-branch" in block
    assert "--cov-fail-under=90" in block
    assert "find examples -type f -name '*.py'" in block
    assert '--include="$$source" --fail-under=90' in block
    assert "$(PYTEST_WORKER_ARGS)" in block
    assert "COVERAGE_FILE=$(COVERAGE_EXAMPLES_FILE)" in block


def test_maintenance_scripts_participate_in_repo_branch_coverage() -> None:
    block = _target("coverage-maintenance", "coverage-enforce")
    config = (ROOT / "scripts" / "maintenance.coveragerc").read_text(encoding="utf-8")
    for test_file in (
        "test_coverage_branch_rate.py",
        "test_public_evidence_audit.py",
        "test_check_repo_cruft.py",
        "test_sync_packaged_contracts.py",
        "test_sync_packaged_public_evidence.py",
        "test_prepare_qualification_suites.py",
        "test_cve_audit.py",
        "test_filter_scorecard_sarif.py",
        "test_run_pip_audit.py",
    ):
        assert test_file in block
    assert "--cov-config=scripts/maintenance.coveragerc" in block
    assert "--cov-fail-under=90" in block
    assert "COVERAGE_FILE=$(COVERAGE_MAINTENANCE_FILE)" in block
    assert "git ls-files 'scripts/checks/*.py' 'scripts/security/*.py'" in block
    assert '--include="$$source" --fail-under=90' in block
    assert "scripts/checks/*.py" in config
    assert "scripts/prepare_qualification_suites.py" in config
    assert "scripts/security/*.py" in config


def test_every_maintained_script_is_assigned_to_one_coverage_surface() -> None:
    tracked = {
        line
        for line in subprocess.check_output(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "scripts/*.py",
                "scripts/**/*.py",
            ],
            cwd=ROOT,
            text=True,
        ).splitlines()
        if not line.endswith("/__init__.py") and (ROOT / line).is_file()
    }
    assigned: Counter[str] = Counter()
    for relative in (
        "scripts/qualification.coveragerc",
        "scripts/release.coveragerc",
        "scripts/maintenance.coveragerc",
    ):
        config = configparser.ConfigParser()
        config.read(ROOT / relative)
        for pattern in filter(None, config["run"]["include"].splitlines()):
            assigned.update(
                path.relative_to(ROOT).as_posix()
                for path in ROOT.glob(pattern.strip())
                if path.is_file() and not path.name.startswith("__init__")
            )

    assert assigned == Counter(tracked)


def test_every_ratchet_example_module_is_collected_for_coverage() -> None:
    block = _target("coverage-examples", "coverage-enforce")
    selectors = set(re.findall(r"--cov=([^ \\\n]+)", block))
    maintained = sorted((ROOT / "examples").rglob("*.py"))

    missing: list[str] = []
    for source in maintained:
        if source.name == "__init__.py":
            continue
        relative = source.relative_to(ROOT).as_posix()
        module = relative.removesuffix(".py").replace("/", ".")
        covered = any(
            module == selector
            or module.startswith(f"{selector}.")
            or ("/" in selector and relative.startswith(f"{selector.rstrip('/')}/"))
            for selector in selectors
        )
        if not covered:
            missing.append(relative)

    assert missing == [], f"example modules omitted from coverage collection: {missing}"


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
    assert "coverage-enforce-parallel: PYTEST_WORKERS = 2" in MAKEFILE


def test_primary_verification_and_coverage_targets_default_to_parallel() -> None:
    assert "VERIFY_TARGET_JOBS ?= 3" in MAKEFILE
    assert "verify: PYTEST_WORKERS = 2" in MAKEFILE
    assert "verify-fast: PYTEST_WORKERS = 2" in MAKEFILE
    assert "COVERAGE_TARGET_JOBS ?= 3" in MAKEFILE
    assert "coverage-enforce: PYTEST_WORKERS = 2" in MAKEFILE
    assert "coverage-enforce: coverage-linux-check" in MAKEFILE
    assert "$(MAKE) -j $(COVERAGE_TARGET_JOBS)" in MAKEFILE
    assert "coverage coverage-addins coverage-qualification" in MAKEFILE
    assert "coverage-release coverage-examples coverage-maintenance" in MAKEFILE
    assert "PYTEST_WORKERS=$(PYTEST_WORKERS)" in MAKEFILE
    assert "coverage-enforce-parallel: PYTEST_WORKERS = 2" in MAKEFILE
    assert "$(PYTEST) $(PYTEST_WORKER_ARGS) -q" in MAKEFILE
    assert "scripts/checks/check_coverage_branch_rate.py" in MAKEFILE
    assert "reports/cov.xml reports/addins-cov.xml" in MAKEFILE
    assert (
        "reports/examples-cov.xml reports/maintenance-cov.xml --minimum 90" in MAKEFILE
    )
    assert "--class-exemptions reports/examples-cov.xml examples" in MAKEFILE
    assert "examples/coverage-exemptions.txt" in MAKEFILE


def test_primary_verification_runs_independent_suites_with_bounded_parallelism() -> (
    None
):
    complete = _target("verify", "verify-fast")
    fast = _target("verify-fast", "contracts-check")

    for block, test_target in ((complete, "test"), (fast, "test-fast")):
        assert "$(MAKE) repo-cruft-check" in block
        assert "$(MAKE) -j $(VERIFY_TARGET_JOBS)" in block
        assert "public-evidence-audit contracts-check" in block
        assert f"{test_target} addins-test" in block
        assert "cli-smoke-core lint" in block
        assert "PYTEST_WORKERS=$(PYTEST_WORKERS)" in block
        assert block.index("$(MAKE) repo-cruft-check") < block.index(
            "$(MAKE) -j $(VERIFY_TARGET_JOBS)"
        )
        assert block.index("$(MAKE) -j $(VERIFY_TARGET_JOBS)") < block.index(
            "$(MAKE) examples-check"
        )

    assert "docs-check-build" in complete
    assert "docs-check-build" not in fast


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
