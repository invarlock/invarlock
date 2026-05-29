from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _write_verdict(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_verdict_tables_core_and_without_pm_counts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts/evidence_packs/python/verdict/tables.py"

    verdict_path = tmp_path / "final_verdict.json"
    out_json = tmp_path / "tables.json"
    out_md = tmp_path / "tables.md"

    # Minimal verdict payload with 3 records:
    # - r0: PM-only
    # - r1: invariants-only (without PM)
    # - r2: rmt + PM (with PM)
    payload = {
        "verdict": "PASS",
        "counts": {"models_total": 1},
        "core_guard_order": [
            "invariants",
            "spectral",
            "rmt",
            "variance",
            "primary_metric",
        ],
        "records": [
            {
                "category": "error_injection",
                "name": "pm_only",
                "passed": False,
                "guard_flags": {
                    "invariants": False,
                    "spectral": False,
                    "rmt": False,
                    "variance": False,
                    "primary_metric": True,
                },
                "spectral_caps_applied": 0,
            },
            {
                "category": "error_injection",
                "name": "inv_only",
                "passed": False,
                "guard_flags": {
                    "invariants": True,
                    "spectral": False,
                    "rmt": False,
                    "variance": False,
                    "primary_metric": False,
                },
                "spectral_caps_applied": 0,
            },
            {
                "category": "error_injection",
                "name": "rmt_with_pm",
                "passed": False,
                "guard_flags": {
                    "invariants": False,
                    "spectral": False,
                    "rmt": True,
                    "variance": False,
                    "primary_metric": True,
                },
                "spectral_caps_applied": 0,
            },
        ],
    }
    _write_verdict(verdict_path, payload)

    subprocess.run(
        [
            "python3",
            str(script),
            "--verdict",
            str(verdict_path),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        check=True,
        cwd=repo_root,
    )

    tables = json.loads(out_json.read_text(encoding="utf-8"))
    core = {row["guard"]: row for row in tables["core_guards"]}

    assert core["primary_metric"]["flagged"] == 2
    assert core["invariants"]["flagged"] == 1
    assert core["invariants"]["unique"] == 1
    assert core["invariants"]["flagged_without_pm"] == 1
    assert core["invariants"]["flagged_with_pm"] == 0

    assert core["rmt"]["flagged"] == 1
    assert core["rmt"]["unique"] == 0
    assert core["rmt"]["flagged_without_pm"] == 0
    assert core["rmt"]["flagged_with_pm"] == 1

    assert tables["non_pm_without_pm"]["any_non_pm_without_pm"] == 1
    assert tables["non_pm_without_pm"]["multi_non_pm_without_pm"] == 0
