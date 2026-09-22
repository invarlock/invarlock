"""Native-shape replay across all evaluator profiles and scorer workflows.

The dictionaries exercise native export contracts; no installed evaluator SDK
execution is implied. Retained EM/NLL values are real; judge trials are synthetic.
"""

from __future__ import annotations

import importlib.util
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/integrations/evaluator-parity/run.py"
SPEC = importlib.util.spec_from_file_location("evaluator_parity", SCRIPT)
assert SPEC and SPEC.loader
PARITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARITY)
EVALUATORS = tuple(PARITY.profiles())


def test_qualification_invariants_survive_optimized_python():
    code = (
        "import runpy; "
        f"module=runpy.run_path({str(SCRIPT)!r}); "
        "module['require'](False, 'optimized qualification refusal')"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "optimized qualification refusal" in result.stderr


def test_qualification_key_is_new_and_owner_only(tmp_path):
    path = tmp_path / "signer.pem"
    PARITY._key(path)
    original = path.read_bytes()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        PARITY._key(path)
    assert path.read_bytes() == original


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


@pytest.mark.parametrize("supplied_capture", [False, True])
@pytest.mark.parametrize("input_format", PARITY.INPUT_FORMATS)
@pytest.mark.parametrize("scorer", PARITY.SCORERS)
def test_source_transaction_keeps_policy_result_and_rejects_tampering(
    tmp_path, scorer, input_format, supplied_capture, monkeypatch, capsys
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
    if supplied_capture:
        captures = _native_captures(tmp_path / "captures", "pydantic-evals", scorer)
        sys.argv.extend(["--native-captures", str(captures)])
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
    assert PARITY.read(output / "preflight.json")["ok"]
    assert "--preflight" in result["commands"][0]["command"]


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


def _native_captures(directory, evaluator="pydantic-evals", scorer="judge"):
    """Native-shape test fixtures only; this helper does not execute an SDK."""
    import hashlib
    import json

    directory.mkdir()
    version = PARITY.profiles()[evaluator]
    manifest = {
        "format": "invarlock/evaluator-native-captures-v1",
        "evaluator": evaluator,
        "source_version": version,
        "scorer": scorer,
        "files": {},
        "provenance": {"capture": "Native-shape test fixture; no SDK execution"},
    }
    for side, original in zip(
        ("baseline", "subject"), PARITY.retained(scorer)[0], strict=True
    ):
        payload = PARITY.SHAPES.payload(
            evaluator, original["records"], version, run_id=f"parity-{scorer}-{side}"
        )
        if evaluator == "langfuse":
            payload = payload["result"]
        raw = (json.dumps(payload, separators=(",", ":")) + "\n\n").encode()
        (directory / f"{side}.json").write_bytes(raw)
        manifest["files"][side] = {
            "sha256": "sha256:" + hashlib.sha256(raw).hexdigest()
        }
    PARITY.write(directory / "origin.json", manifest)
    return directory


@pytest.mark.parametrize("input_format", PARITY.INPUT_FORMATS)
@pytest.mark.parametrize("evaluator", ["pydantic-evals", "langfuse"])
@pytest.mark.parametrize("scorer", PARITY.SCORERS)
def test_supplied_capture_preserves_raw_bytes_and_original_records(
    tmp_path, monkeypatch, evaluator, scorer, input_format
):
    captures = _native_captures(tmp_path / "captures", evaluator, scorer)
    output = tmp_path / "output"
    output.mkdir()

    def no_shape_builder(*args, **kwargs):
        raise AssertionError("supplied captures must not rebuild native payloads")

    monkeypatch.setattr(PARITY.SHAPES, "payload", no_shape_builder)
    sources, runs, _, origin = PARITY.export_pair(
        output, evaluator, scorer, input_format=input_format, native_captures=captures
    )
    assert (
        origin["native_captures"]["provenance"]
        == PARITY.read(captures / "origin.json")["provenance"]
    )
    assert (output / "capture-origin.json").read_bytes() == (
        captures / "origin.json"
    ).read_bytes()
    for side, source, run in zip(("baseline", "subject"), sources, runs, strict=True):
        raw = (captures / f"{side}.json").read_bytes()
        assert (output / f"capture-{side}.json").read_bytes() == raw
        if input_format == "native-json":
            assert (output / source["path"]).read_bytes() == raw
        assert len(run["records"]) == (2 if scorer == "judge" else 400)


@pytest.mark.parametrize(
    "failure",
    [
        "not-object",
        "unknown-field",
        "format",
        "evaluator",
        "version",
        "scorer",
        "provenance",
        "missing-side",
        "path-binding",
        "digest",
        "oversize-manifest",
    ],
)
def test_supplied_capture_refuses_invalid_manifest(tmp_path, failure):
    captures = _native_captures(tmp_path / "captures")
    manifest = PARITY.read(captures / "origin.json")
    if failure == "not-object":
        manifest = []
    elif failure == "unknown-field":
        manifest["extra"] = True
    elif failure in {"format", "evaluator", "scorer"}:
        manifest[failure] = "unexpected"
    elif failure == "version":
        manifest["source_version"] = "wrong-version"
    elif failure == "provenance":
        manifest["provenance"] = {}
    elif failure == "missing-side":
        del manifest["files"]["subject"]
    elif failure == "path-binding":
        manifest["files"]["baseline"]["path"] = "../outside.json"
    elif failure == "digest":
        manifest["files"]["subject"]["sha256"] = "sha256:" + "0" * 64
    PARITY.write(captures / "origin.json", manifest)
    if failure == "oversize-manifest":
        (captures / "origin.json").write_bytes(b" " * 65537)
    with pytest.raises((ValueError, OSError)):
        PARITY.read_native_captures(
            captures, "pydantic-evals", "judge", PARITY.profiles()["pydantic-evals"]
        )


@pytest.mark.parametrize(
    "failure", ["missing", "root-symlink", "file-symlink", "non-file", "raw-tamper"]
)
def test_supplied_capture_refuses_unsafe_or_changed_files(tmp_path, failure):
    captures = _native_captures(tmp_path / "captures")
    if failure == "missing":
        (captures / "subject.json").unlink()
    elif failure == "root-symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(captures, target_is_directory=True)
        captures = alias
    elif failure == "file-symlink":
        (captures / "subject.json").unlink()
        (captures / "subject.json").symlink_to(captures / "baseline.json")
    elif failure == "non-file":
        (captures / "subject.json").unlink()
        (captures / "subject.json").mkdir()
    else:
        with (captures / "subject.json").open("ab") as handle:
            handle.write(b" ")
    with pytest.raises((ValueError, OSError)):
        PARITY.read_native_captures(
            captures, "pydantic-evals", "judge", PARITY.profiles()["pydantic-evals"]
        )


@pytest.mark.parametrize("input_format", PARITY.INPUT_FORMATS)
@pytest.mark.parametrize("failure", ["missing-id", "changed-output"])
def test_supplied_capture_rejects_rehashed_semantic_changes_before_signing(
    tmp_path, input_format, failure
):
    import hashlib

    captures = _native_captures(tmp_path / "captures")
    payload = PARITY.read(captures / "subject.json")
    if failure == "missing-id":
        payload["cases"].pop()
    else:
        payload["cases"][0]["output"] = "changed output"
    PARITY.write(captures / "subject.json", payload)
    manifest = PARITY.read(captures / "origin.json")
    manifest["files"]["subject"]["sha256"] = (
        "sha256:" + hashlib.sha256((captures / "subject.json").read_bytes()).hexdigest()
    )
    PARITY.write(captures / "origin.json", manifest)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises((AssertionError, ValueError)):
        PARITY.prepare(
            output,
            "pydantic-evals",
            "judge",
            input_format=input_format,
            native_captures=captures,
        )
    assert not (output / "signer.pem").exists()
    assert not (output / "evidence").exists()


@pytest.mark.parametrize("input_format", PARITY.INPUT_FORMATS)
@pytest.mark.parametrize("scorer", PARITY.SCORERS)
def test_installed_supplied_native_captures(tmp_path, input_format, scorer):
    import json
    import subprocess

    python = os.environ.get("INVARLOCK_EVALUATOR_PARITY_PYTHON")
    if not python:
        pytest.skip(
            "set INVARLOCK_EVALUATOR_PARITY_PYTHON to the isolated candidate wheel interpreter"
        )
    captures = _native_captures(tmp_path / "captures", "langfuse", scorer)
    output = tmp_path / "journey"
    result = subprocess.run(
        [
            python,
            "-I",
            str(SCRIPT),
            "--evaluator",
            "langfuse",
            "--scorer",
            scorer,
            "--input-format",
            input_format,
            "--native-captures",
            str(captures),
            "--recipient-python",
            python,
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    summary = json.loads(result.stdout)
    assert len(summary["recipient"]["sdk_modules_absent"]) == 18
    assert summary["tamper_rejected"]
    assert summary["decision"] == (
        "regression" if scorer == "normalized_nll" else "pass"
    )
    assert (output / "capture-origin.json").read_bytes() == (
        captures / "origin.json"
    ).read_bytes()


@pytest.mark.parametrize("input_format", PARITY.INPUT_FORMATS)
@pytest.mark.parametrize("valid_projection", [False, True])
def test_supplied_capture_uses_explicit_logger_projection(
    tmp_path, input_format, valid_projection
):
    import hashlib

    evaluator = "lm-evaluation-harness"
    captures = _native_captures(tmp_path / "captures", evaluator, "judge")
    manifest = PARITY.read(captures / "origin.json")
    projection = {
        "kind": "json-pointer",
        "pointer": "/context/arguments/gen_args_0/arg_0"
        if valid_projection
        else "/context/arguments/missing",
    }
    manifest["provenance"]["input_projection"] = projection
    for side in ("baseline", "subject"):
        payload = PARITY.read(captures / f"{side}.json")
        for row in payload:
            row["arguments"] = {"gen_args_0": {"arg_0": row["arguments"][0][0]}}
        PARITY.write(captures / f"{side}.json", payload)
        manifest["files"][side]["sha256"] = (
            "sha256:"
            + hashlib.sha256((captures / f"{side}.json").read_bytes()).hexdigest()
        )
    PARITY.write(captures / "origin.json", manifest)
    output = tmp_path / "output"
    output.mkdir()
    if not valid_projection:
        with pytest.raises(ValueError):
            PARITY.export_pair(
                output,
                evaluator,
                "judge",
                input_format=input_format,
                native_captures=captures,
            )
        return
    sources, runs, _, _ = PARITY.export_pair(
        output, evaluator, "judge", input_format=input_format, native_captures=captures
    )
    assert all(source["input_projection"] == projection for source in sources)
    assert runs[0]["records"][0]["input"] == "Capital of France?"
