#!/usr/bin/env python3
"""Run the wheel-user signed-evidence quickstart outside the source tree."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, BinaryIO

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

_DIGEST = re.compile(r"sha256:[a-f0-9]{64}\Z")
_MAX_ANCHOR_BYTES = 64 * 1024
_MAX_COMMAND_OUTPUT = 1024 * 1024
_COMMAND_TIMEOUT_SECONDS = 30


class QuickstartError(RuntimeError):
    """Raised when the quickstart cannot produce a trustworthy result."""


def _strict_object(path: Path, *, label: str) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise QuickstartError(f"{label} contains a duplicate field")
            result[key] = value
        return result

    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise QuickstartError(f"{label} must be a real regular file")
        payload = b""
        while len(payload) <= _MAX_ANCHOR_BYTES:
            chunk = os.read(descriptor, _MAX_ANCHOR_BYTES + 1 - len(payload))
            if not chunk:
                break
            payload += chunk
        if len(payload) > _MAX_ANCHOR_BYTES:
            raise QuickstartError(f"{label} exceeds the size limit")
        value = json.loads(payload, object_pairs_hook=pairs)
    except QuickstartError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QuickstartError(f"{label} is not readable strict JSON") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(value, dict):
        raise QuickstartError(f"{label} must be a JSON object")
    return value


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise QuickstartError(f"{label} must be a lowercase sha256 digest")
    return value


def _anchors(path: Path) -> dict[str, str]:
    value = _strict_object(path, label="technical anchors")
    artifacts = value.get("artifact_digests")
    runtimes = value.get("runtime_digests")
    if not isinstance(artifacts, dict) or set(artifacts) != {"baseline", "subject"}:
        raise QuickstartError("technical anchors must bind both artifacts")
    if not isinstance(runtimes, dict) or set(runtimes) != {"baseline", "subject"}:
        raise QuickstartError("technical anchors must bind both runtimes")
    return {
        "baseline_artifact": _digest(
            artifacts["baseline"], label="baseline artifact anchor"
        ),
        "subject_artifact": _digest(
            artifacts["subject"], label="subject artifact anchor"
        ),
        "schedule": _digest(value.get("schedule_digest"), label="schedule anchor"),
        "baseline_runtime": _digest(
            runtimes["baseline"], label="baseline runtime anchor"
        ),
        "subject_runtime": _digest(runtimes["subject"], label="subject runtime anchor"),
        "signer": _digest(
            value.get("evidence_signer_fingerprint"),
            label="evidence signer anchor",
        ),
    }


def _real_fixture(path: Path) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    if lexical.is_symlink() or not lexical.is_dir():
        raise QuickstartError("fixture must be a real directory")
    resolved = lexical.resolve()
    if resolved != lexical:
        raise QuickstartError("fixture path must not traverse symbolic links")
    required = (
        resolved / "evidence",
        resolved / "evaluated-policy.json",
        resolved / "technical-anchors.json",
    )
    if any(not item.exists() or item.is_symlink() for item in required):
        raise QuickstartError("fixture is incomplete or contains symbolic links")
    if any(item.is_symlink() for item in resolved.rglob("*")):
        raise QuickstartError("fixture must not contain symbolic links")
    return resolved


def _new_output(path: Path) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    try:
        parent = lexical.parent.resolve(strict=True)
    except OSError as exc:
        raise QuickstartError("output parent must be an existing directory") from exc
    if parent != lexical.parent or not parent.is_dir():
        raise QuickstartError("output parent must be a real directory")
    destination = parent / lexical.name
    if destination.exists() or destination.is_symlink():
        raise QuickstartError("output directory must be new")
    try:
        destination.mkdir(mode=0o700)
    except OSError as exc:
        raise QuickstartError("could not create output directory") from exc
    return destination


def _write_verifier_key(path: Path) -> None:
    key = ed25519.Ed25519PrivateKey.generate()
    payload = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o600)
    except OSError as exc:
        path.unlink(missing_ok=True)
        raise QuickstartError("could not create the temporary verifier key") from exc


def _read_command_output(handle: BinaryIO, *, limit: int) -> bytes:
    handle.seek(0)
    return handle.read(limit + 1)


def _run_cli(arguments: list[str], *, cwd: Path) -> dict[str, Any]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update({"PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1"})
    try:
        with (
            tempfile.TemporaryFile(dir=cwd) as stdout_file,
            tempfile.TemporaryFile(dir=cwd) as stderr_file,
        ):
            completed = subprocess.run(
                [sys.executable, "-m", "invarlock", *arguments],
                cwd=cwd,
                env=environment,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                timeout=_COMMAND_TIMEOUT_SECONDS,
            )
            stdout_bytes = _read_command_output(stdout_file, limit=_MAX_COMMAND_OUTPUT)
            stderr_bytes = _read_command_output(
                stderr_file,
                limit=max(0, _MAX_COMMAND_OUTPUT - len(stdout_bytes)),
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise QuickstartError("InvarLock command did not complete safely") from exc
    if len(stdout_bytes) + len(stderr_bytes) > _MAX_COMMAND_OUTPUT:
        raise QuickstartError("InvarLock command output exceeded the size limit")
    try:
        stdout = stdout_bytes.decode("utf-8")
        stderr = stderr_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QuickstartError("InvarLock command did not return UTF-8 output") from exc
    if completed.returncode != 0:
        detail = (stderr or stdout).strip()
        raise QuickstartError(f"InvarLock command rejected the fixture: {detail}")
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise QuickstartError("InvarLock command did not return JSON") from exc
    if not isinstance(result, dict):
        raise QuickstartError("InvarLock command returned an invalid result")
    return result


def run_quickstart(*, fixture: Path, output: Path) -> dict[str, Path]:
    """Verify the fixture, issue a receipt, and render a human report."""

    fixture_root = _real_fixture(fixture)
    anchors = _anchors(fixture_root / "technical-anchors.json")
    destination = _new_output(output)
    key = destination / ".verifier.private.pem"
    receipt = destination / "verification.receipt.json"
    verification = destination / "verification.result.json"
    report = destination / "evidence.html"
    try:
        _write_verifier_key(key)
        verified = _run_cli(
            [
                "verify",
                str(fixture_root / "evidence"),
                "--policy",
                str(fixture_root / "evaluated-policy.json"),
                "--expected-baseline-artifact",
                anchors["baseline_artifact"],
                "--expected-subject-artifact",
                anchors["subject_artifact"],
                "--expected-schedule",
                anchors["schedule"],
                "--expected-baseline-runtime",
                anchors["baseline_runtime"],
                "--expected-subject-runtime",
                anchors["subject_runtime"],
                "--expected-signer",
                anchors["signer"],
                "--receipt",
                str(receipt),
                "--verifier-signing-key",
                str(key),
                "--verifier-identity",
                "quickstart-verifier",
                "--json",
            ],
            cwd=destination,
        )
        if verified.get("ok") is not True or verified.get("policy_verdict") != "pass":
            raise QuickstartError("signed evidence did not produce a passing verdict")
        with verification.open("x", encoding="utf-8") as handle:
            json.dump(verified, handle, separators=(",", ":"), sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        rendered = _run_cli(
            [
                "report",
                str(fixture_root / "evidence"),
                "--html",
                str(report),
                "--explain",
                "--json",
            ],
            cwd=destination,
        )
        if (
            rendered.get("ok") is not True
            or not receipt.is_file()
            or not report.is_file()
        ):
            raise QuickstartError("signed handoff outputs are incomplete")
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    finally:
        key.unlink(missing_ok=True)
    return {"receipt": receipt, "verification": verification, "report": report}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path("golden"))
    parser.add_argument(
        "--output", type=Path, default=Path("invarlock-quickstart-output")
    )
    arguments = parser.parse_args(argv)
    try:
        outputs = run_quickstart(fixture=arguments.fixture, output=arguments.output)
    except QuickstartError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 2
    print("PASS signed evidence verified")
    print("Decision: pass")
    print(f"Signed verifier receipt: {outputs['receipt']}")
    print(f"Verification result: {outputs['verification']}")
    print(f"HTML report: {outputs['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
