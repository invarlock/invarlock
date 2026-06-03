from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.security.filter_scorecard_sarif import filter_sarif


def test_filter_sarif_removes_excluded_results_and_rules() -> None:
    payload = {
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "Scorecard",
                        "rules": [
                            {"id": "CIIBestPracticesID"},
                            {"id": "BranchProtectionID"},
                        ],
                    },
                    "extensions": [
                        {"name": "extra", "rules": [{"id": "CIIBestPracticesID"}]}
                    ],
                },
                "results": [
                    {"ruleId": "CIIBestPracticesID", "message": {"text": "drop me"}},
                    {"ruleId": "BranchProtectionID", "message": {"text": "keep me"}},
                ],
            }
        ],
    }

    filtered = filter_sarif(payload, {"CIIBestPracticesID"})

    assert filtered["runs"][0]["results"] == [
        {"ruleId": "BranchProtectionID", "message": {"text": "keep me"}}
    ]
    assert filtered["runs"][0]["tool"]["driver"]["rules"] == [
        {"id": "BranchProtectionID"}
    ]
    assert filtered["runs"][0]["tool"]["extensions"][0]["rules"] == []


def test_filter_sarif_preserves_malformed_rule_entries() -> None:
    payload = {
        "runs": [
            {
                "tool": {
                    "driver": {
                        "rules": [
                            {"id": "CIIBestPracticesID"},
                            "malformed-rule",
                        ],
                    },
                    "extensions": [
                        {
                            "rules": [
                                {"id": "CIIBestPracticesID"},
                                None,
                            ]
                        }
                    ],
                },
                "results": [
                    {"ruleId": "CIIBestPracticesID"},
                    "malformed-result",
                ],
            },
            "malformed-run",
        ],
    }

    filtered = filter_sarif(payload, {"CIIBestPracticesID"})

    assert filtered["runs"][0]["results"] == ["malformed-result"]
    assert filtered["runs"][0]["tool"]["driver"]["rules"] == ["malformed-rule"]
    assert filtered["runs"][0]["tool"]["extensions"][0]["rules"] == [None]
    assert filtered["runs"][1] == "malformed-run"


def test_cli_writes_filtered_output(tmp_path: Path) -> None:
    input_path = tmp_path / "input.sarif"
    output_path = tmp_path / "output.sarif"
    input_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "tool": {"driver": {"rules": [{"id": "CIIBestPracticesID"}]}},
                        "results": [{"ruleId": "CIIBestPracticesID"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/security/filter_scorecard_sarif.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--exclude-rule",
            "CIIBestPracticesID",
        ],
        check=True,
        cwd=Path.cwd(),
    )

    filtered = json.loads(output_path.read_text(encoding="utf-8"))
    assert filtered["runs"][0]["results"] == []
    assert filtered["runs"][0]["tool"]["driver"]["rules"] == []
