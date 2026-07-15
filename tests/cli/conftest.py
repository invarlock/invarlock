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
    RuntimeSecurityPolicy,
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


def _should_auto_seed_runtime_provenance(node: pytest.FixtureRequest) -> bool:
    path = Path(str(node.node.path))
    return (
        path.name.startswith("test_verify")
        or path.name == "test_evidence_pack_commands.py"
    ) and path.name != "test_verify_runtime_provenance.py"


def _should_preserve_container_default_cli_test(node: pytest.FixtureRequest) -> bool:
    path = Path(str(node.node.path))
    return path.name in {
        "test_container_delegation.py",
        "test_container_default_contract.py",
        "test_container_default_model_mounts.py",
    }


@pytest.fixture(autouse=True)
def _default_cli_host_execution(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> Generator[None, None, None]:
    if _should_preserve_container_default_cli_test(request):
        yield
        return

    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE_DIGEST", _VALID_TEST_IMAGE_DIGEST)
    monkeypatch.setattr(
        "invarlock.cli.config_execution.host_execution_allowed",
        lambda: True,
    )
    monkeypatch.setattr(
        "invarlock.cli.config_execution.resolve_shell_runtime_security_policy",
        lambda **_: RuntimeSecurityPolicy(allow_host_execution=True),
    )
    monkeypatch.setattr(
        "invarlock.cli.security_helpers.resolve_shell_runtime_security_policy",
        lambda **_: RuntimeSecurityPolicy(allow_host_execution=True),
    )
    yield


def _write_test_runtime_manifest(
    report_path: Path,
    *,
    report_bytes: bytes | None = None,
) -> None:
    payload = {
        "manifest_version": RUNTIME_MANIFEST_VERSION,
        "generated_at_utc": "2026-03-21T00:00:00+00:00",
        "verifier_contract_version": RUNTIME_VERIFIER_CONTRACT_VERSION,
        "report": {
            "path": str(report_path.resolve()),
            "filename": report_path.name,
            "sha256": hashlib.sha256(
                report_path.read_bytes() if report_bytes is None else report_bytes
            ).hexdigest(),
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
def _auto_seed_verify_runtime_provenance(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> Generator[None, None, None]:
    if not _should_auto_seed_runtime_provenance(request):
        yield
        return

    from invarlock.reporting import verify_contract as verify_mod

    original_verify_runtime_provenance = verify_mod.verify_runtime_provenance

    def _verify_runtime_provenance(
        report_path: str | Path,
        *,
        allow_unverified: bool = False,
        expected_image_digest: str | None = None,
        report_bytes: bytes | None = None,
        require_strict_runtime: bool = False,
    ):
        if not allow_unverified:
            _write_test_runtime_manifest(
                Path(report_path),
                report_bytes=report_bytes,
            )
        return original_verify_runtime_provenance(
            report_path,
            allow_unverified=allow_unverified,
            expected_image_digest=expected_image_digest,
            report_bytes=report_bytes,
            require_strict_runtime=require_strict_runtime,
        )

    monkeypatch.setattr(
        verify_mod,
        "verify_runtime_provenance",
        _verify_runtime_provenance,
        raising=True,
    )
    yield
