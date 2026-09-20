"""Native-shape replay across all evaluator profiles and scorer workflows.

The dictionaries exercise native export contracts; no installed evaluator SDK
execution is implied. Retained EM/NLL values are real; judge trials are synthetic.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/integrations/evaluator-parity/run.py"
SPEC = importlib.util.spec_from_file_location("evaluator_parity", SCRIPT)
assert SPEC and SPEC.loader
PARITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARITY)
EVALUATORS = tuple(PARITY.profiles())


@pytest.mark.parametrize("input_format", PARITY.INPUT_FORMATS)
@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize("scorer", PARITY.SCORERS)
def test_native_export_roundtrip_preserves_each_retained_case(
    tmp_path, evaluator, scorer, input_format
):
    from invarlock.evaluation_records.adapters import load_run

    sources, runs, _, _ = PARITY.export_pair(
        tmp_path, evaluator, scorer, input_format=input_format
    )
    assert len(runs[0]["records"]) == (2 if scorer == "judge" else 400)
    for source, expected in zip(sources, runs, strict=True):
        if input_format == "native-json" and evaluator == "langfuse":
            payload = PARITY.read(tmp_path / source["path"])
            assert "item_results" in payload and "format" not in payload
        assert (
            load_run(
                tmp_path / source["path"],
                **{k: v for k, v in source.items() if k != "path"},
            )
            == expected
        )
        import hashlib

        assert (
            expected["source_digest"]
            == "sha256:"
            + hashlib.sha256((tmp_path / source["path"]).read_bytes()).hexdigest()
        )
        assert all(
            ("langfuse" if evaluator == "langfuse" else "upstream_record")
            in row["context"]
            for row in expected["records"]
        )


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_independent_schedule_rejects_missing_duplicate_and_existing_export(
    tmp_path, evaluator
):
    from invarlock.engine import export_evaluator_result

    rows = PARITY.retained("judge")[0][0]["records"]
    options = {
        "expected_ids": [r["id"] for r in rows],
        "source_version": PARITY.profiles()[evaluator],
        "run_id": "synthetic-refusals",
        "artifact_digest": "sha256:" + "a" * 64,
    }
    destination = tmp_path / "export.json"
    for invalid in ([], rows[:1], [rows[0], rows[0]]):
        with pytest.raises(ValueError):
            export_evaluator_result(
                evaluator,
                PARITY.SHAPES.payload(
                    evaluator,
                    invalid,
                    options["source_version"],
                    run_id=options["run_id"],
                ),
                destination,
                **options,
            )
        assert not destination.exists()
    destination.write_bytes(b"preserve original")
    with pytest.raises((ValueError, FileExistsError)):
        export_evaluator_result(
            evaluator,
            PARITY.SHAPES.payload(
                evaluator, rows, options["source_version"], run_id=options["run_id"]
            ),
            destination,
            **options,
        )
    assert destination.read_bytes() == b"preserve original"


@pytest.mark.parametrize("input_format", PARITY.INPUT_FORMATS)
@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize("scorer", PARITY.SCORERS)
def test_installed_sdk_free_recipient_signed_journey(
    tmp_path, evaluator, scorer, input_format
):
    python = os.environ.get("INVARLOCK_EVALUATOR_PARITY_PYTHON")
    if not python:
        pytest.skip(
            "set INVARLOCK_EVALUATOR_PARITY_PYTHON to the isolated candidate wheel interpreter"
        )
    import json
    import subprocess

    output = tmp_path / "journey"
    completed = subprocess.run(
        [
            python,
            "-I",
            str(SCRIPT),
            "--evaluator",
            evaluator,
            "--scorer",
            scorer,
            "--input-format",
            input_format,
            "--recipient-python",
            python,
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
        cwd=tmp_path,
    )
    result = json.loads(completed.stdout)
    assert len(result["recipient"]["sdk_modules_absent"]) == 18
    assert result["input_format"] == input_format
    assert result["tamper_rejected"]
    assert result["decision"] == (
        "regression" if scorer == "normalized_nll" else "pass"
    )


@pytest.mark.parametrize("input_format", PARITY.INPUT_FORMATS)
@pytest.mark.parametrize("scorer", PARITY.SCORERS)
def test_source_transaction_keeps_policy_result_and_rejects_tampering(
    tmp_path, scorer, input_format, monkeypatch, capsys
):
    """Source regression check; installed-recipient proof is a separate test."""
    import subprocess
    import sys

    # This check exercises the launcher and real CLI transaction with source
    # imports. Installed tests separately enforce the recipient identity guard.
    real_run = subprocess.run

    def source_process(command, **kwargs):
        if command[1:4] == ["-I", "-m", "invarlock"]:
            # Select this checkout even when the test interpreter has an older
            # wheel installed. Production recipient calls stay isolated.
            bootstrap = (
                "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); "
                "sys.argv[0]='invarlock'; runpy.run_module('invarlock',run_name='__main__')"
            )
            command = [
                command[0],
                "-I",
                "-c",
                bootstrap,
                str(ROOT / "src"),
                *command[4:],
            ]
        return real_run(command, **kwargs)

    monkeypatch.setattr(PARITY.subprocess, "run", source_process)
    monkeypatch.setattr(
        PARITY, "installed_identity", lambda _: {"scope": "source test"}
    )
    output = tmp_path / "journey"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--evaluator",
            "pydantic-evals",
            "--scorer",
            scorer,
            "--input-format",
            input_format,
            "--recipient-python",
            sys.executable,
            "--output",
            str(output),
        ],
    )
    PARITY.main()
    result = PARITY.read(output / "result.json")
    assert '"scope": "source test"' in capsys.readouterr().out
    assert result["input_format"] == input_format
    assert result["tamper_rejected"]
    assert result["decision"] == (
        "regression" if scorer == "normalized_nll" else "pass"
    )
    assert PARITY.read(output / "verification.json")
    assert PARITY.read(output / "tamper-refusal.json")


def test_recipient_guard_refuses_unprovisioned_interpreter(tmp_path):
    import subprocess
    import sys

    recipient = tmp_path / "recipient"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(recipient)],
        check=True,
        timeout=30,
    )
    with pytest.raises(subprocess.CalledProcessError):
        PARITY.installed_identity(recipient / "bin/python")


def test_native_shape_refuses_unknown_profile():
    rows = PARITY.retained("judge")[0][0]["records"]
    with pytest.raises(ValueError, match="unsupported evaluator profile"):
        PARITY.SHAPES.payload("unknown-evaluator", rows, "1.0")
