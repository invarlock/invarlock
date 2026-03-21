from __future__ import annotations

import hashlib
import json
from collections.abc import Generator
from pathlib import Path

import pytest

from invarlock.runtime_security import (
    RUNTIME_MANIFEST_FILENAME,
    RUNTIME_MANIFEST_VERSION,
    RUNTIME_VERIFIER_CONTRACT_VERSION,
)

_VALID_TEST_IMAGE_DIGEST = "sha256:" + ("a" * 64)


@pytest.fixture(autouse=True)
def _reset_plugin_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """
    Normalize plugin-discovery env for CLI tests.

    Third-party plugin discovery is disabled by default. Most CLI plugin tests
    exercise built-ins only, so explicitly keep third-party discovery off unless
    an individual test opts in.
    """
    monkeypatch.setenv("INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS", "0")
    yield


def _should_auto_attest_verify_test(node: pytest.FixtureRequest) -> bool:
    path = Path(str(node.node.path))
    return (
        path.name.startswith("test_verify")
        and path.name != "test_unattested_verify_gate.py"
    )


def _write_test_runtime_manifest(report_path: Path) -> None:
    payload = {
        "manifest_version": RUNTIME_MANIFEST_VERSION,
        "generated_at_utc": "2026-03-21T00:00:00+00:00",
        "verifier_contract_version": RUNTIME_VERIFIER_CONTRACT_VERSION,
        "report": {
            "path": str(report_path.resolve()),
            "filename": report_path.name,
            "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        },
        "config": {
            "path": None,
            "sha256": None,
            "source": "missing",
        },
        "execution_mode": "container",
        "runtime": {
            "image_ref": "ghcr.io/invarlock/invarlock-runtime:test",
            "image_digest": _VALID_TEST_IMAGE_DIGEST,
            "container_execution": True,
            "allow_network": False,
            "allow_remote_code": False,
            "allow_third_party_plugins": False,
        },
    }
    (report_path.parent / RUNTIME_MANIFEST_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _auto_attest_verify_reports(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[None, None, None]:
    if not _should_auto_attest_verify_test(request):
        yield
        return

    verifier_path = (
        tmp_path_factory.mktemp("runtime-verifier") / "invarlock-runtime-verify"
    )
    verifier_path.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import jsonschema
from pathlib import Path
import sys

from invarlock.public_contracts import load_runtime_manifest_schema


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report_path = Path(args.report)
    manifest_path = Path(args.manifest)
    errors: list[str] = []
    if not report_path.exists():
        errors.append(f"missing report: {report_path}")
    if not manifest_path.exists():
        errors.append(f"missing manifest: {manifest_path}")
    manifest = {}
    if not errors:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid manifest: {exc}")
    if not errors:
        try:
            jsonschema.validate(
                instance=manifest, schema=load_runtime_manifest_schema()
            )
        except jsonschema.ValidationError as exc:
            errors.append(f"schema invalid: {exc.message}")
    if not errors:
        expected = hashlib.sha256(report_path.read_bytes()).hexdigest()
        actual = ((manifest.get("report") or {}).get("sha256"))
        if actual != expected:
            errors.append("report digest mismatch")
        if manifest.get("execution_mode") != "container":
            errors.append("execution mode must be container")
        runtime = manifest.get("runtime") or {}
        if not runtime.get("image_digest"):
            errors.append("missing runtime image digest")

    payload = {"ok": not errors, "errors": errors}
    print(json.dumps(payload))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )
    verifier_path.chmod(0o755)
    monkeypatch.setenv("INVARLOCK_RUNTIME_VERIFIER", str(verifier_path))
    from invarlock.cli.commands import verify as verify_mod

    original_verify_runtime_attestation = verify_mod.verify_runtime_attestation

    def _verify_runtime_attestation(
        report_path: str | Path,
        *,
        allow_unattested: bool = False,
    ):
        if not allow_unattested:
            _write_test_runtime_manifest(Path(report_path))
        return original_verify_runtime_attestation(
            report_path,
            allow_unattested=allow_unattested,
        )

    monkeypatch.setattr(
        verify_mod,
        "verify_runtime_attestation",
        _verify_runtime_attestation,
        raising=True,
    )
    yield
