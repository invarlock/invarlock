from __future__ import annotations

import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "contracts" / "broad_exception_review_buckets.json"


def _load_contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _scan_broad_exception_sites() -> dict[str, list[int]]:
    contract = _load_contract()
    actual: dict[str, list[int]] = {}
    for raw_root in contract["scope_roots"]:
        root = REPO_ROOT / str(raw_root)
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            lines = sorted(
                node.lineno
                for node in ast.walk(tree)
                if isinstance(node, ast.ExceptHandler)
                and isinstance(node.type, ast.Name)
                and node.type.id == "Exception"
            )
            if lines:
                actual[path.relative_to(REPO_ROOT).as_posix()] = lines
    return actual


def _contract_sites() -> dict[str, list[int]]:
    contract = _load_contract()
    expected: dict[str, list[int]] = {}
    for entry in contract["entries"]:
        rel_path = str(entry["path"]).replace("\\", "/")
        expected[rel_path] = sorted(int(line) for line in entry["lines"])
    return expected


def test_broad_exception_review_bucket_contract_matches_source() -> None:
    assert _scan_broad_exception_sites() == _contract_sites()


def test_reporting_broad_exception_count_stays_zero() -> None:
    actual = _scan_broad_exception_sites()
    reporting_paths = [
        path for path in actual if path.startswith("src/invarlock/reporting/")
    ]
    assert not reporting_paths, "\n".join(reporting_paths)


def test_should_remove_bucket_keeps_trust_critical_surfaces_explicit() -> None:
    contract = _load_contract()
    must_include = {
        "src/invarlock/core/determinism_policy.py",
        "src/invarlock/core/runner_guards.py",
        "src/invarlock/core/runner_pairing.py",
        "src/invarlock/guards/invariants.py",
        "src/invarlock/guards/spectral_detection.py",
        "src/invarlock/guards/variance_prepare.py",
    }
    actual = {
        str(entry["path"]).replace("\\", "/")
        for entry in contract["entries"]
        if entry["bucket"] == "should_remove"
    }
    assert must_include.issubset(actual)
