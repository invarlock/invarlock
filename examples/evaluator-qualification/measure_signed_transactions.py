#!/usr/bin/env python3
"""Measure compact retained signed-transaction verification and rendering."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock.evidence_receipt import verify_signed_verification_receipt
from invarlock.evidence_reporting import render_evidence
from invarlock.evidence_verification import EvidenceVerificationError, verify_evidence

FORMAT = "invarlock/signed-transaction-costs-v1"
TRANSACTION_IDS = (
    "deployment-approval-inspect-ai",
    "gemma4-lm-evaluation-harness",
    "qwen35-inspect-ai",
    "qwen35-lm-evaluation-harness",
)
PROFILE_IDS = {"lm-evaluation-harness", "inspect-ai"}


class MeasurementError(RuntimeError):
    """Raised when a retained transaction cannot be measured faithfully."""


def _object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MeasurementError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise MeasurementError(f"{label} must be a JSON object")
    return value


def _tree_stats(path: Path) -> tuple[int, int]:
    if not path.is_dir() or path.is_symlink():
        raise MeasurementError("measurement root must be a real directory")
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    if any(candidate.is_symlink() for candidate in path.rglob("*")):
        raise MeasurementError("measurement root must not contain symbolic links")
    return len(files), sum(candidate.stat().st_size for candidate in files)


def _record_count(evidence: Path) -> int:
    report = _object(
        evidence / "reports/evaluation.report.json", label="comparison report"
    )
    count = report.get("record_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise MeasurementError("comparison report record count is invalid")
    return count


def _verification_anchors(transaction: dict[str, Any]) -> dict[str, Any]:
    verification = transaction.get("verification")
    if not isinstance(verification, dict):
        raise MeasurementError("transaction verification anchors are invalid")
    return verification


def _verify_retained_receipt(
    transaction_root: Path,
    verification: dict[str, Any],
    *,
    expected_policy_verdict: str,
) -> None:
    result = verify_signed_verification_receipt(
        transaction_root / "verification.receipt.json",
        transaction_root / "evidence",
        policy_path=transaction_root / "policy.json",
        expected_artifact_digests=verification["artifact_digests"],
        expected_schedule_digest=verification["schedule_digest"],
        expected_runtime_digests=verification["runtime_digests"],
        expected_pack_signer_fingerprint=verification["evidence_signer_fingerprint"],
        expected_verifier_identity=verification["verifier_identity"],
        expected_verifier_fingerprint=verification["verifier_fingerprint"],
        expected_trust_profile_digest=verification["trust_profile_digest"],
    )
    verdict = (
        result.statement.get("verdict") if isinstance(result.statement, dict) else None
    )
    if (
        not result.ok
        or not isinstance(verdict, dict)
        or verdict.get("integrity_ok") is not True
        or verdict.get("policy_verdict") != expected_policy_verdict
        or verdict.get("ok") is not (expected_policy_verdict == "pass")
    ):
        raise MeasurementError(
            "retained receipt verification failed: " + "; ".join(result.errors)
        )


def _verify_and_issue_receipt(
    transaction_root: Path,
    verification: dict[str, Any],
    *,
    receipt: Path,
    verifier_key: bytes,
    expected_policy_verdict: str,
) -> None:
    try:
        verify_evidence(
            transaction_root / "evidence",
            policy_path=transaction_root / "policy.json",
            expected_baseline_artifact=verification["artifact_digests"]["baseline"],
            expected_subject_artifact=verification["artifact_digests"]["subject"],
            expected_schedule=verification["schedule_digest"],
            expected_baseline_runtime=verification["runtime_digests"]["baseline"],
            expected_subject_runtime=verification["runtime_digests"]["subject"],
            expected_signer=verification["evidence_signer_fingerprint"],
            receipt_path=receipt,
            verifier_signing_key_bytes=verifier_key,
            verifier_identity="operational-measurement-verifier",
            trust_profile_digest=verification["trust_profile_digest"],
        )
    except EvidenceVerificationError as exc:
        if expected_policy_verdict != "fail" or exc.exit_code != 7:
            raise
    else:
        if expected_policy_verdict != "pass":
            raise MeasurementError("retained policy outcome changed unexpectedly")


def _timings(operation: Callable[[int], None], *, runs: int) -> list[float]:
    operation(-1)
    values: list[float] = []
    for index in range(runs):
        started = time.perf_counter_ns()
        operation(index)
        values.append((time.perf_counter_ns() - started) / 1_000_000)
    return values


def measure_transaction(
    transaction_root: Path, *, runs: int, temporary_root: Path
) -> dict[str, object]:
    """Measure one retained transaction after a successful warmup."""

    transaction = _object(
        transaction_root / "transaction.json", label="transaction record"
    )
    profile_id = transaction.get("profile_id")
    if profile_id not in PROFILE_IDS:
        raise MeasurementError("transaction profile is not retained")
    evidence = transaction_root / "evidence"
    verification = _verification_anchors(transaction)
    expected_policy_verdict = verification.get("policy_verdict")
    if expected_policy_verdict not in {"pass", "fail"}:
        raise MeasurementError("transaction policy verdict is invalid")
    assert isinstance(expected_policy_verdict, str)
    _verify_retained_receipt(
        transaction_root,
        verification,
        expected_policy_verdict=expected_policy_verdict,
    )
    evidence_files, evidence_bytes = _tree_stats(evidence)
    package_files, package_bytes = _tree_stats(transaction_root)
    verifier_key = ed25519.Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    with tempfile.TemporaryDirectory(
        prefix=f".invarlock-{profile_id}-", dir=temporary_root
    ) as raw_directory:
        rendered = Path(raw_directory)
        verification_ms = _timings(
            lambda index: _verify_and_issue_receipt(
                transaction_root,
                verification,
                receipt=rendered / f"receipt-{index}.json",
                verifier_key=verifier_key,
                expected_policy_verdict=expected_policy_verdict,
            ),
            runs=runs,
        )
        report_ms = _timings(
            lambda index: render_evidence(
                evidence,
                html_path=rendered / f"report-{index}.html",
                explain=True,
            ),
            runs=runs,
        )
    return {
        "evidence_bytes": evidence_bytes,
        "evidence_files": evidence_files,
        "package_bytes": package_bytes,
        "package_files": package_files,
        "policy_verdict": expected_policy_verdict,
        "profile_id": profile_id,
        "record_count": _record_count(evidence),
        "report_render_median_ms": round(statistics.median(report_ms), 3),
        "transaction_id": transaction_root.name,
        "verification_and_receipt_median_ms": round(
            statistics.median(verification_ms), 3
        ),
    }


def measure_all(*, root: Path, runs: int) -> dict[str, object]:
    """Measure the current-model retained transactions."""

    if not 1 <= runs <= 100:
        raise MeasurementError("runs must be between 1 and 100")
    transaction_parent = root / "examples/evaluator-qualification/signed-transactions"
    results = [
        measure_transaction(
            transaction_parent / transaction_id,
            runs=runs,
            temporary_root=root,
        )
        for transaction_id in TRANSACTION_IDS
    ]
    return {
        "environment": {
            "machine": platform.machine(),
            "platform": platform.system(),
            "python": platform.python_version(),
        },
        "format": FORMAT,
        "runs": runs,
        "transactions": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--runs", type=int, default=7)
    arguments = parser.parse_args(argv)
    try:
        result = measure_all(root=arguments.root.resolve(), runs=arguments.runs)
    except (MeasurementError, KeyError, OSError, ValueError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
