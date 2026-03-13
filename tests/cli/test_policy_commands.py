from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

from typer.testing import CliRunner

os.environ["INVARLOCK_LIGHT_IMPORT"] = "1"

from invarlock.cli.app import app


def test_policy_build_and_verify_cli(tmp_path: Path) -> None:
    resolved_policy = tmp_path / "resolved_policy.json"
    resolved_policy.write_text(
        json.dumps({"metrics": {"pm_ratio": {"ratio_limit_base": 1.1}}}),
        encoding="utf-8",
    )
    overrides = tmp_path / "overrides.json"
    overrides.write_text(
        json.dumps([{"path": "metrics.pm_ratio.ratio_limit_base", "value": 1.1}]),
        encoding="utf-8",
    )
    compatibility = tmp_path / "compatibility.json"
    compatibility.write_text(
        json.dumps({"support_tiers": ["published_basis"]}),
        encoding="utf-8",
    )
    out = tmp_path / "policy-pack.json"

    runner = CliRunner()
    build = runner.invoke(
        app,
        [
            "policy",
            "build",
            "--resolved-policy",
            str(resolved_policy),
            "--overrides",
            str(overrides),
            "--compatibility",
            str(compatibility),
            "--out",
            str(out),
            "--owner",
            "oss",
        ],
    )
    assert build.exit_code == 0, build.output
    assert out.is_file()

    verify = runner.invoke(app, ["policy", "verify", str(out), "--json"])
    assert verify.exit_code == 0, verify.output
    payload = json.loads(verify.stdout.strip().splitlines()[-1])
    assert payload["format_version"] == "policy-pack-verify-v1"
    assert payload["ok"] is True


def test_policy_load_structured_input_supports_json_yaml_and_none(
    tmp_path: Path,
) -> None:
    policy_cmd = importlib.import_module("invarlock.cli.commands.policy")
    json_path = tmp_path / "payload.json"
    json_path.write_text(json.dumps({"kind": "json"}), encoding="utf-8")
    yaml_path = tmp_path / "payload.yaml"
    yaml_path.write_text("kind: yaml\n", encoding="utf-8")

    assert policy_cmd._load_structured_input(None) is None
    assert policy_cmd._load_structured_input(str(json_path)) == {"kind": "json"}
    assert policy_cmd._load_structured_input(str(yaml_path)) == {"kind": "yaml"}


def test_policy_build_rejects_non_mapping_resolved_policy(tmp_path: Path) -> None:
    resolved_policy = tmp_path / "resolved_policy.json"
    resolved_policy.write_text(json.dumps(["not", "a", "mapping"]), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "policy",
            "build",
            "--resolved-policy",
            str(resolved_policy),
            "--out",
            str(tmp_path / "policy-pack.json"),
        ],
    )

    assert result.exit_code == 2
    assert "--resolved-policy must decode to an object" in result.output


def test_policy_build_records_optional_approval_metadata(tmp_path: Path) -> None:
    resolved_policy = tmp_path / "resolved_policy.json"
    resolved_policy.write_text(
        json.dumps({"metrics": {"pm_ratio": {"ratio_limit_base": 1.05}}}),
        encoding="utf-8",
    )
    overrides = tmp_path / "overrides.yaml"
    overrides.write_text(
        "\n".join(
            [
                "- path: metrics.pm_ratio.ratio_limit_base",
                "  value: 1.05",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "policy-pack.json"

    runner = CliRunner()
    build = runner.invoke(
        app,
        [
            "policy",
            "build",
            "--resolved-policy",
            str(resolved_policy),
            "--overrides",
            str(overrides),
            "--out",
            str(out),
            "--owner",
            "release-bot",
            "--change-ticket",
            "PR-42",
            "--rationale",
            "ratchet threshold",
            "--effective-date",
            "2026-03-12",
            "--signature",
            "sig:abc123",
        ],
    )
    assert build.exit_code == 0, build.output
    assert "Wrote policy pack" in build.output

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["approval"] == {
        "owner": "release-bot",
        "change_ticket": "PR-42",
        "rationale": "ratchet threshold",
        "effective_date": "2026-03-12",
        "signature": "sig:abc123",
    }


def test_policy_verify_reports_human_and_json_failures(tmp_path: Path) -> None:
    resolved_policy = tmp_path / "resolved_policy.json"
    resolved_policy.write_text(
        json.dumps({"metrics": {"pm_ratio": {"ratio_limit_base": 1.1}}}),
        encoding="utf-8",
    )
    out = tmp_path / "policy-pack.json"

    runner = CliRunner()
    build = runner.invoke(
        app,
        [
            "policy",
            "build",
            "--resolved-policy",
            str(resolved_policy),
            "--out",
            str(out),
        ],
    )
    assert build.exit_code == 0, build.output

    ok_result = runner.invoke(app, ["policy", "verify", str(out)])
    assert ok_result.exit_code == 0, ok_result.output
    assert "Policy pack verified" in ok_result.output

    tampered = json.loads(out.read_text(encoding="utf-8"))
    tampered["policy_digest"] = "0000000000000000"
    out.write_text(json.dumps(tampered), encoding="utf-8")

    human_fail = runner.invoke(app, ["policy", "verify", str(out)])
    assert human_fail.exit_code == 2
    assert "policy digest mismatch" in human_fail.output

    json_fail = runner.invoke(app, ["policy", "verify", str(out), "--json"])
    assert json_fail.exit_code == 2
    payload = json.loads(json_fail.stdout.strip().splitlines()[-1])
    assert payload["ok"] is False
    assert payload["resolution"]["exit_code"] == 2
    assert any("policy digest mismatch" in error for error in payload["errors"])
