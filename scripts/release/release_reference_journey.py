#!/usr/bin/env python3
"""Replay one retained evidence pack through an installed release candidate."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

REFERENCE_FORMAT = "invarlock/release-reference-journey-v1"
SUMMARY_FORMAT = "invarlock/release-reference-result-v1"
TRUST_FORMAT = "invarlock/trust-inputs-v1"
VERIFY_FORMAT = "invarlock/evidence-pack-verify-v1"
REPORT_FORMAT = "invarlock/evidence-report-v1"
RECEIPT_FORMAT = "invarlock/evidence-verification-receipt-v2"
RECEIPT_SIGNATURE_FORMAT = "invarlock/evidence-verification-receipt-signature-v1"
REFERENCE_CONFIG = Path("scripts/release/reference_evidence/qwen38-27b-anchors.json")
MAX_JSON_BYTES = 1024 * 1024
MAX_REPORT_BYTES = 16 * 1024 * 1024
_DIGEST = re.compile(r"sha256:[a-f0-9]{64}\Z")


class ReleaseReferenceJourneyError(RuntimeError):
    """Raised when the retained-evidence consumer journey fails closed."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReleaseReferenceJourneyError("reference JSON is not canonical") from exc


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    if not raw or len(raw) > MAX_JSON_BYTES:
        raise ReleaseReferenceJourneyError(f"{label} has an invalid byte length")

    def reject_duplicate_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"duplicate key: {key}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid constant: {value}")

    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReleaseReferenceJourneyError(f"{label} is not strict JSON") from exc
    if not isinstance(parsed, dict):
        raise ReleaseReferenceJourneyError(f"{label} must be a JSON object")
    return parsed


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _relative_parts(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ReleaseReferenceJourneyError(f"{label} must be a safe relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ReleaseReferenceJourneyError(f"{label} must be a safe relative path")
    return path.parts


def _resolve_checkout_path(
    repo_root: Path,
    value: object,
    *,
    label: str,
    directory: bool,
) -> Path:
    current = repo_root
    for part in _relative_parts(value, label=label):
        current = current / part
        try:
            observed = current.lstat()
        except OSError as exc:
            raise ReleaseReferenceJourneyError(f"{label} is unavailable") from exc
        if stat.S_ISLNK(observed.st_mode):
            raise ReleaseReferenceJourneyError(f"{label} must not contain symlinks")
    if directory:
        if not current.is_dir():
            raise ReleaseReferenceJourneyError(f"{label} must be a directory")
    elif not current.is_file():
        raise ReleaseReferenceJourneyError(f"{label} must be a regular file")
    return current


def _read_regular_file(path: Path, *, label: str, maximum: int) -> bytes:
    try:
        observed = path.lstat()
        if not stat.S_ISREG(observed.st_mode) or observed.st_size > maximum:
            raise ReleaseReferenceJourneyError(f"{label} is not a bounded regular file")
        raw = path.read_bytes()
    except OSError as exc:
        raise ReleaseReferenceJourneyError(f"{label} is unavailable") from exc
    if len(raw) != observed.st_size:
        raise ReleaseReferenceJourneyError(f"{label} changed while being read")
    return raw


def _require_digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ReleaseReferenceJourneyError(f"{label} must be a sha256 digest")
    return value


def _load_reference(repo_root: Path) -> dict[str, Any]:
    config_path = _resolve_checkout_path(
        repo_root,
        REFERENCE_CONFIG.as_posix(),
        label="release reference configuration",
        directory=False,
    )
    config = _strict_json_object(
        _read_regular_file(
            config_path,
            label="release reference configuration",
            maximum=MAX_JSON_BYTES,
        ),
        label="release reference configuration",
    )
    if set(config) != {"anchors", "evidence", "expected", "format", "policy"}:
        raise ReleaseReferenceJourneyError("release reference fields are invalid")
    if config.get("format") != REFERENCE_FORMAT:
        raise ReleaseReferenceJourneyError("release reference format is invalid")

    anchors = config.get("anchors")
    expected_anchor_fields = {
        "baseline_artifact_digest",
        "baseline_runtime_digest",
        "evidence_signer_fingerprint",
        "request_digest",
        "schedule_digest",
        "subject_artifact_digest",
        "subject_runtime_digest",
    }
    if not isinstance(anchors, dict) or set(anchors) != expected_anchor_fields:
        raise ReleaseReferenceJourneyError("release reference anchors are invalid")
    for name, value in anchors.items():
        _require_digest(value, label=f"release reference anchor {name}")

    evidence = config.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != {
        "comparison_id",
        "pack_manifest_digest",
        "path",
        "reference_id",
    }:
        raise ReleaseReferenceJourneyError("release reference evidence is invalid")
    _relative_parts(evidence.get("path"), label="release reference evidence path")
    _require_digest(
        evidence.get("pack_manifest_digest"),
        label="release reference manifest digest",
    )
    for name in ("comparison_id", "reference_id"):
        value = evidence.get(name)
        if not isinstance(value, str) or not value or value != value.strip():
            raise ReleaseReferenceJourneyError(f"release reference {name} is invalid")

    expected = config.get("expected")
    if not isinstance(expected, dict) or set(expected) != {
        "policy_verdict",
        "verification_scope",
    }:
        raise ReleaseReferenceJourneyError("release reference expectations are invalid")
    if expected.get("policy_verdict") != "pass":
        raise ReleaseReferenceJourneyError("release reference verdict is invalid")
    if expected.get("verification_scope") != "paired_comparison":
        raise ReleaseReferenceJourneyError("release reference scope is invalid")

    policy = config.get("policy")
    if not isinstance(policy, dict) or set(policy) != {"digest", "path"}:
        raise ReleaseReferenceJourneyError("release reference policy is invalid")
    _relative_parts(policy.get("path"), label="release reference policy path")
    _require_digest(policy.get("digest"), label="release reference policy digest")
    return config


def _prepare_workspace(repo_root: Path, workspace: Path) -> Path:
    lexical = Path(os.path.abspath(os.fspath(workspace)))
    if lexical.exists() or lexical.is_symlink():
        raise ReleaseReferenceJourneyError("release reference workspace must be new")
    try:
        parent = lexical.parent.resolve(strict=True)
    except OSError as exc:
        raise ReleaseReferenceJourneyError(
            "release reference workspace parent is unavailable"
        ) from exc
    resolved = parent / lexical.name
    if _is_within(resolved, repo_root):
        raise ReleaseReferenceJourneyError(
            "release reference workspace must be outside the checkout"
        )
    try:
        resolved.mkdir(mode=0o700)
    except OSError as exc:
        raise ReleaseReferenceJourneyError(
            "release reference workspace could not be created"
        ) from exc
    return resolved


def _sanitized_environment(
    repo_root: Path, *, allow_checkout_source: bool
) -> dict[str, str]:
    environment = dict(os.environ)
    for key in tuple(environment):
        if key in {"PYTHONHOME", "PYTHONPATH"} or key.startswith("INVARLOCK_"):
            environment.pop(key, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONSAFEPATH"] = "1"
    if allow_checkout_source:
        environment["PYTHONPATH"] = str(repo_root / "src")
    return environment


def _run_cli(
    command: Sequence[str],
    arguments: Sequence[str],
    *,
    cwd: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [*command, *arguments],
            check=False,
            capture_output=True,
            text=True,
            cwd=cwd,
            env=environment,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeError) as exc:
        raise ReleaseReferenceJourneyError(
            "release reference candidate command failed"
        ) from exc
    if completed.returncode != 0:
        raise ReleaseReferenceJourneyError(
            "release reference candidate command returned nonzero"
        )
    return _strict_json_object(
        completed.stdout.encode("utf-8"),
        label="release reference candidate output",
    )


def _expected_result_anchors(
    anchors: dict[str, Any], policy_digest: str
) -> dict[str, Any]:
    return {
        "artifact_digests": {
            "baseline": anchors["baseline_artifact_digest"],
            "subject": anchors["subject_artifact_digest"],
        },
        "policy_digest": policy_digest,
        "request_digest": anchors["request_digest"],
        "runtime_digests": {
            "baseline": anchors["baseline_runtime_digest"],
            "subject": anchors["subject_runtime_digest"],
        },
        "schedule_digest": anchors["schedule_digest"],
        "signer_fingerprint": anchors["evidence_signer_fingerprint"],
    }


def _validate_verification_result(
    result: dict[str, Any], config: dict[str, Any]
) -> None:
    evidence = config["evidence"]
    expected = config["expected"]
    policy = config["policy"]
    anchors = config["anchors"]
    required = {
        "assurance_status": "verified",
        "authenticity": "pinned",
        "comparison_id": evidence["comparison_id"],
        "format_version": VERIFY_FORMAT,
        "integrity_ok": True,
        "ok": True,
        "pack_format": "invarlock/evidence-pack-v1",
        "pack_manifest_digest": evidence["pack_manifest_digest"],
        "policy_verdict": expected["policy_verdict"],
        "reports_verified": True,
        "request_digest": anchors["request_digest"],
        "verification_scope": expected["verification_scope"],
        "verifier_identity": "release-reference-verifier",
    }
    if any(result.get(name) != value for name, value in required.items()):
        raise ReleaseReferenceJourneyError(
            "release reference verification result is inconsistent"
        )
    if result.get("errors") != [] or result.get("warnings") != []:
        raise ReleaseReferenceJourneyError(
            "release reference verification emitted diagnostics"
        )
    if result.get("anchors") != _expected_result_anchors(anchors, policy["digest"]):
        raise ReleaseReferenceJourneyError(
            "release reference verification anchors changed"
        )
    for name in ("trust_profile_digest", "verifier_fingerprint"):
        _require_digest(result.get(name), label=f"verification result {name}")


def _public_key_fingerprint(public_key: ed25519.Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _validate_receipt(
    receipt_path: Path,
    verification: dict[str, Any],
    config: dict[str, Any],
) -> None:
    raw = _read_regular_file(
        receipt_path,
        label="release reference verification receipt",
        maximum=MAX_JSON_BYTES,
    )
    receipt = _strict_json_object(raw, label="release reference verification receipt")
    if raw != _canonical_json_bytes(receipt) or set(receipt) != {
        "signature",
        "statement",
    }:
        raise ReleaseReferenceJourneyError(
            "release reference verification receipt is not canonical"
        )
    statement = receipt.get("statement")
    signature = receipt.get("signature")
    if not isinstance(statement, dict) or not isinstance(signature, dict):
        raise ReleaseReferenceJourneyError(
            "release reference verification receipt is invalid"
        )
    anchors = config["anchors"]
    expected_statement_anchors = {
        **_expected_result_anchors(anchors, config["policy"]["digest"]),
        "pack_signer_fingerprint": anchors["evidence_signer_fingerprint"],
    }
    expected_statement_anchors.pop("signer_fingerprint")
    if (
        statement.get("format") != RECEIPT_FORMAT
        or statement.get("pack_manifest_digest")
        != config["evidence"]["pack_manifest_digest"]
        or statement.get("anchors") != expected_statement_anchors
        or statement.get("verdict")
        != {
            "integrity_ok": True,
            "ok": True,
            "policy_verdict": config["expected"]["policy_verdict"],
            "verification_status": 0,
        }
    ):
        raise ReleaseReferenceJourneyError(
            "release reference verification receipt statement changed"
        )
    verifier = statement.get("verifier")
    if not isinstance(verifier, dict) or verifier != {
        "identity": "release-reference-verifier",
        "signing_key_fingerprint": verification["verifier_fingerprint"],
        "trust_profile_digest": verification["trust_profile_digest"],
    }:
        raise ReleaseReferenceJourneyError(
            "release reference verification receipt verifier changed"
        )
    if set(signature) != {"algorithm", "format", "public_key", "value"} or (
        signature.get("algorithm") != "ed25519"
        or signature.get("format") != RECEIPT_SIGNATURE_FORMAT
    ):
        raise ReleaseReferenceJourneyError(
            "release reference verification receipt signature is invalid"
        )
    public_key_block = signature.get("public_key")
    if (
        not isinstance(public_key_block, dict)
        or set(public_key_block)
        != {
            "encoding",
            "value",
        }
        or public_key_block.get("encoding") != "pem"
    ):
        raise ReleaseReferenceJourneyError(
            "release reference verification receipt public key is invalid"
        )
    try:
        public_key = serialization.load_pem_public_key(
            str(public_key_block["value"]).encode("ascii")
        )
        encoded_signature = signature.get("value")
        if not isinstance(public_key, ed25519.Ed25519PublicKey) or not isinstance(
            encoded_signature, str
        ):
            raise ValueError("invalid Ed25519 receipt material")
        public_key.verify(
            base64.b64decode(encoded_signature, validate=True),
            _canonical_json_bytes(statement),
        )
    except (InvalidSignature, TypeError, ValueError) as exc:
        raise ReleaseReferenceJourneyError(
            "release reference verification receipt signature did not verify"
        ) from exc
    if _public_key_fingerprint(public_key) != verification["verifier_fingerprint"]:
        raise ReleaseReferenceJourneyError(
            "release reference verification receipt fingerprint changed"
        )


def _validate_report_result(
    result: dict[str, Any], *, html_path: Path, manifest_digest: str
) -> None:
    if result != {
        "format_version": REPORT_FORMAT,
        "html": str(html_path),
        "ok": True,
        "pack_manifest_digest": manifest_digest,
    }:
        raise ReleaseReferenceJourneyError(
            "release reference report result is inconsistent"
        )


def run_release_reference_journey(
    *,
    repo_root: Path,
    command: Sequence[str],
    workspace: Path,
    allow_checkout_source: bool,
) -> dict[str, Any]:
    """Verify and render the retained reference through one selected CLI."""

    try:
        root = repo_root.resolve(strict=True)
    except OSError as exc:
        raise ReleaseReferenceJourneyError("release checkout is unavailable") from exc
    if (
        not root.is_dir()
        or not command
        or any(not isinstance(part, str) or not part for part in command)
    ):
        raise ReleaseReferenceJourneyError("release reference invocation is invalid")
    candidate = Path(command[0])
    try:
        candidate_stat = candidate.stat()
    except OSError as exc:
        raise ReleaseReferenceJourneyError(
            "release reference candidate executable is unavailable"
        ) from exc
    if not stat.S_ISREG(candidate_stat.st_mode) or candidate_stat.st_mode & 0o111 == 0:
        raise ReleaseReferenceJourneyError(
            "release reference candidate executable is invalid"
        )

    config = _load_reference(root)
    evidence_source = _resolve_checkout_path(
        root,
        config["evidence"]["path"],
        label="release reference evidence",
        directory=True,
    )
    manifest = _read_regular_file(
        evidence_source / "manifest.json",
        label="release reference evidence manifest",
        maximum=MAX_JSON_BYTES,
    )
    if (
        f"sha256:{hashlib.sha256(manifest).hexdigest()}"
        != config["evidence"]["pack_manifest_digest"]
    ):
        raise ReleaseReferenceJourneyError(
            "release reference evidence manifest does not match its pin"
        )
    policy_source = _resolve_checkout_path(
        root,
        config["policy"]["path"],
        label="release reference policy",
        directory=False,
    )
    policy_bytes = _read_regular_file(
        policy_source,
        label="release reference policy",
        maximum=MAX_JSON_BYTES,
    )
    if (
        f"sha256:{hashlib.sha256(policy_bytes).hexdigest()}"
        != config["policy"]["digest"]
    ):
        raise ReleaseReferenceJourneyError(
            "release reference policy does not match its pin"
        )

    output_root = _prepare_workspace(root, workspace)
    incoming = output_root / "incoming"
    trust = output_root / "trust"
    incoming.mkdir(mode=0o700)
    trust.mkdir(mode=0o700)
    evidence = incoming / "evidence"
    shutil.copytree(evidence_source, evidence, symlinks=True)
    policy_path = trust / "acceptance.json"
    policy_path.write_bytes(policy_bytes)

    private_key = ed25519.Ed25519PrivateKey.generate()
    key_path = trust / "verifier.pem"
    key_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    key_path.chmod(0o600)
    profile = {
        "allow_installed_scorers": False,
        "anchors": config["anchors"],
        "format": TRUST_FORMAT,
        "policy": {"path": policy_path.name},
        "verifier": {
            "identity": "release-reference-verifier",
            "signing_key_path": key_path.name,
        },
    }
    profile_path = trust / "profile.json"
    profile_path.write_bytes(_canonical_json_bytes(profile))

    environment = _sanitized_environment(
        root, allow_checkout_source=allow_checkout_source
    )
    receipt_path = output_root / "verification.receipt.json"
    try:
        verification = _run_cli(
            command,
            (
                "verify",
                str(evidence),
                "--trust-profile",
                str(profile_path),
                "--receipt",
                str(receipt_path),
                "--json",
            ),
            cwd=output_root,
            environment=environment,
        )
        _validate_verification_result(verification, config)
        _validate_receipt(receipt_path, verification, config)
    finally:
        try:
            key_path.unlink()
        except OSError as exc:
            raise ReleaseReferenceJourneyError(
                "release reference verifier key could not be removed"
            ) from exc

    report_paths = (output_root / "report-a.html", output_root / "report-b.html")
    for report_path in report_paths:
        report = _run_cli(
            command,
            (
                "report",
                str(evidence),
                "--html",
                str(report_path),
                "--explain",
                "--json",
            ),
            cwd=output_root,
            environment=environment,
        )
        _validate_report_result(
            report,
            html_path=report_path,
            manifest_digest=config["evidence"]["pack_manifest_digest"],
        )
    first_report = _read_regular_file(
        report_paths[0],
        label="release reference HTML report",
        maximum=MAX_REPORT_BYTES,
    )
    second_report = _read_regular_file(
        report_paths[1],
        label="release reference repeated HTML report",
        maximum=MAX_REPORT_BYTES,
    )
    if not first_report or first_report != second_report:
        raise ReleaseReferenceJourneyError(
            "release reference report rendering is not deterministic"
        )
    return {
        "comparison_id": config["evidence"]["comparison_id"],
        "format": SUMMARY_FORMAT,
        "ok": True,
        "pack_manifest_digest": config["evidence"]["pack_manifest_digest"],
        "policy_verdict": config["expected"]["policy_verdict"],
        "reference_id": config["evidence"]["reference_id"],
        "report_sha256": f"sha256:{hashlib.sha256(first_report).hexdigest()}",
        "verification_scope": config["expected"]["verification_scope"],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--invarlock-cli", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        repo_root = args.repo_root.resolve(strict=True)
        if args.invarlock_cli is None:
            command = (sys.executable, "-m", "invarlock.cli.app")
            allow_checkout_source = True
        else:
            command = (str(args.invarlock_cli.resolve(strict=True)),)
            allow_checkout_source = False
        if args.workspace is None:
            with tempfile.TemporaryDirectory(
                prefix="invarlock-release-reference-"
            ) as directory:
                summary = run_release_reference_journey(
                    repo_root=repo_root,
                    command=command,
                    workspace=Path(directory).resolve() / "journey",
                    allow_checkout_source=allow_checkout_source,
                )
        else:
            summary = run_release_reference_journey(
                repo_root=repo_root,
                command=command,
                workspace=args.workspace,
                allow_checkout_source=allow_checkout_source,
            )
    except (OSError, ReleaseReferenceJourneyError) as exc:
        print(f"ERROR: release reference journey rejected: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("Release reference journey passed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
