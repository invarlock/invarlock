from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.coverage.check_coverage_thresholds import (
    LIVE_HARDWARE_CLOSURE_REQUIREMENTS,
    REPOSITORY_FLOOR,
    TIER_FLOORS,
    _classification_matches,
    maintained_coverage_files,
)
from tests.scripts._support_check_coverage_thresholds import (
    _run_checker,
    _write_cov_xml,
    _write_line_only_cov_xml,
)

COMPACT = "src/invarlock/reporting/report_contract.py"
BEHAVIORAL = "scripts/evidence_packs/python/editing/streaming_pruning.py"
LIVE = "src/invarlock/guards/exact_svd.py"


def test_every_maintained_module_matches_exactly_one_risk_tier() -> None:
    root = Path(__file__).resolve().parents[2]
    maintained = maintained_coverage_files(root)

    assert maintained
    assert all(len(_classification_matches(path, root)) == 1 for path in maintained)
    assert _classification_matches(COMPACT, root) == ("compact_contract",)
    assert _classification_matches(BEHAVIORAL, root) == ("behavioral",)
    assert _classification_matches(LIVE, root) == ("live_backend",)


def test_risk_tier_floors_are_repo_wide_and_non_green() -> None:
    assert REPOSITORY_FLOOR.line == 0.90
    assert REPOSITORY_FLOOR.branch == 0.80
    assert TIER_FLOORS == {
        "compact_contract": type(REPOSITORY_FLOOR)(line=1.00, branch=1.00),
        "behavioral": type(REPOSITORY_FLOOR)(line=0.95, branch=0.90),
        "live_backend": type(REPOSITORY_FLOOR)(line=0.85, branch=0.75),
    }


def test_checker_enforces_both_line_and_branch_floors(tmp_path: Path) -> None:
    xml = tmp_path / "cov.xml"
    out = tmp_path / "out.json"
    _write_cov_xml(
        xml,
        [
            (COMPACT, 1.0, 0.999),
            (BEHAVIORAL, 0.899, 0.95),
            (LIVE, 0.75, 0.85),
        ],
    )

    proc = _run_checker(xml, out)

    assert proc.returncode == 1
    assert f"{COMPACT}: line coverage 99.900% below required 100%" in proc.stderr
    assert f"{BEHAVIORAL}: branch coverage 89.900% below required 90%" in proc.stderr
    assert LIVE not in proc.stderr


def test_checker_accepts_exact_tier_floors_for_diagnostic_subset(
    tmp_path: Path,
) -> None:
    xml = tmp_path / "cov.xml"
    out = tmp_path / "out.json"
    _write_cov_xml(
        xml,
        [
            (COMPACT, 1.0, 1.0),
            (BEHAVIORAL, 0.90, 0.95),
            (LIVE, 0.75, 0.85),
        ],
    )

    proc = _run_checker(xml, out)

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    records = {record["path"]: record for record in payload["files"]}
    assert records[COMPACT]["tier"] == "compact_contract"
    assert records[BEHAVIORAL]["tier"] == "behavioral"
    assert records[LIVE]["tier"] == "live_backend"


def test_live_tier_records_real_required_hardware_test_receipts(
    tmp_path: Path,
) -> None:
    xml = tmp_path / "cov.xml"
    out = tmp_path / "out.json"
    _write_cov_xml(xml, [(LIVE, 0.75, 0.85)])

    proc = _run_checker(xml, out)

    assert proc.returncode == 0, proc.stderr
    requirement = LIVE_HARDWARE_CLOSURE_REQUIREMENTS[LIVE]
    assert requirement.accelerator == "cuda"
    assert len(requirement.required_test_ids) == 2
    record = json.loads(out.read_text(encoding="utf-8"))["files"][0]
    assert record["hardware_receipt_closure"] == {
        "accelerator": "cuda",
        "required_test_ids": list(requirement.required_test_ids),
    }


def test_duplicate_normalized_paths_fail_instead_of_selecting_best_rate(
    tmp_path: Path,
) -> None:
    xml = tmp_path / "cov.xml"
    out = tmp_path / "out.json"
    _write_cov_xml(xml, [(COMPACT, 1.0, 1.0), (COMPACT, 0.0, 0.0)])

    proc = _run_checker(xml, out)

    assert proc.returncode == 1
    assert f"{COMPACT}: duplicate normalized coverage entry" in proc.stderr
    assert json.loads(out.read_text())["duplicate_coverage_files"] == [COMPACT]


def test_missing_line_or_branch_data_fails_closed(tmp_path: Path) -> None:
    xml = tmp_path / "line-only.xml"
    out = tmp_path / "out.json"
    _write_line_only_cov_xml(xml, COMPACT, 1.0)

    proc = _run_checker(xml, out)

    assert proc.returncode == 1
    assert f"{COMPACT}: no valid branch coverage data present" in proc.stderr


def test_measured_module_outside_maintained_surface_fails(tmp_path: Path) -> None:
    xml = tmp_path / "cov.xml"
    out = tmp_path / "out.json"
    excluded = "src/invarlock/adapters/base.py"
    assert excluded not in maintained_coverage_files()
    _write_cov_xml(xml, [(excluded, 1.0, 1.0)])

    proc = _run_checker(xml, out)

    assert proc.returncode == 1
    assert "measured module is outside the maintained coverage surface" in proc.stderr


def test_missing_maintained_module_fails_by_default(tmp_path: Path) -> None:
    xml = tmp_path / "cov.xml"
    out = tmp_path / "out.json"
    _write_cov_xml(xml, [(COMPACT, 1.0, 1.0)])

    proc = _run_checker(xml, out, allow_missing_threshold_files=False)

    assert proc.returncode == 1
    assert (
        f"{BEHAVIORAL}: maintained production module has no coverage data"
        in proc.stderr
    )


def test_coverage_policy_cli_exposes_tier_metadata() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/coverage/check_coverage_thresholds.py",
            "tier-policy",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["repository_floor"] == {"line": 0.90, "branch": 0.80}
    assert LIVE in payload["live_hardware_closure"]


def test_coverage_module_and_include_cli_remain_makefile_frontdoors() -> None:
    modules = subprocess.check_output(
        [
            sys.executable,
            "scripts/coverage/check_coverage_thresholds.py",
            "coverage-modules",
        ],
        text=True,
    ).strip()
    includes = subprocess.check_output(
        [
            sys.executable,
            "scripts/coverage/check_coverage_thresholds.py",
            "coverage-include",
        ],
        text=True,
    ).strip()

    assert modules == "--cov"
    assert "src/invarlock/reporting/*" in includes.split(",")
    assert LIVE.rsplit("/", 1)[0] + "/*" in includes.split(",")
