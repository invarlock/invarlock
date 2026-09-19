"""Isolated installed-wheel recipients for all Langfuse scorer workflows."""

from __future__ import annotations

import json
import os
import subprocess

import pytest


@pytest.mark.parametrize(
    "metric,decision,status",
    [
        ("exact_match", "pass", 0),
        ("normalized_nll", "regression", 7),
        ("judge", "pass", 0),
    ],
)
def test_installed_wheel_recipient_evaluates_verifies_and_reports(
    tmp_path, metric, decision, status
):
    python = os.environ.get("INVARLOCK_LANGFUSE_WHEEL_PYTHON")
    if not python:
        pytest.skip(
            "set INVARLOCK_LANGFUSE_WHEEL_PYTHON to the candidate wheel interpreter"
        )
    from tests.judge_measurements.test_langfuse_journeys import (
        comparison_request,
        synthetic_judge_request,
    )

    request, key, trust, _ = (
        synthetic_judge_request(tmp_path)
        if metric == "judge"
        else comparison_request(tmp_path, metric)
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("INVARLOCK_SIGNING_KEY", None)

    def command(*arguments, expected=0):
        result = subprocess.run(
            [python, "-I", "-m", "invarlock", *map(str, arguments)],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        assert result.returncode == expected, result.stdout + result.stderr
        return result.stdout

    provenance = subprocess.run(
        [
            python,
            "-I",
            "-c",
            "import invarlock, json; print(json.dumps(invarlock.__file__))",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "site-packages" in json.loads(provenance.stdout)
    evaluated = json.loads(
        command(
            "evaluate",
            request,
            "--signing-key",
            key,
            "--fail-on-policy",
            "--json",
            expected=status,
        )
    )
    assert evaluated["decision"] == decision and evaluated["authentication"] == "signed"
    verify_args = (
        []
        if metric == "judge"
        else ["--receipt", tmp_path / "verification.receipt.json"]
    )
    verified = json.loads(
        command(
            "verify",
            tmp_path / "evidence",
            "--trust-profile",
            trust,
            *verify_args,
            "--json",
            expected=status,
        )
    )
    if metric == "judge":
        assert (
            verified["authenticated"] and verified["replayed"] and verified["accepted"]
        )
    else:
        assert verified["integrity_ok"] and verified["replay_status"] == "completed"
        assert verified["decision"] == decision
    assert json.loads(command("report", tmp_path / "evidence", "--json"))
    command("report", tmp_path / "evidence", "--html", tmp_path / "report.html")
    assert "langfuse" in (tmp_path / "report.html").read_text()
