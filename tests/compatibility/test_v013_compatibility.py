from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import stat
import sys
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock.evidence_pack_verification import verify_comparison_evidence
from invarlock.evidence_receipt import verify_signed_verification_receipt

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/compatibility/v0.13.0"


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def _generator():
    path = FIXTURE_ROOT / "generate_corpus.py"
    spec = importlib.util.spec_from_file_location("v013_compatibility_generator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _case() -> dict[str, Any]:
    corpus = _object(FIXTURE_ROOT / "corpus.json")
    assert corpus["format"] == "invarlock/compatibility-corpus-v1"
    assert corpus["contract_release"] == "0.13.0"
    case = corpus["cases"][0]
    assert isinstance(case, dict)
    return case


def _paths(case: dict[str, Any]) -> tuple[Path, Path, Path]:
    return (
        REPO_ROOT / case["evidence"],
        REPO_ROOT / case["receipt"],
        REPO_ROOT / case["policy"],
    )


def _verify_receipt(
    receipt: Path,
    evidence: Path,
    policy: Path,
    case: dict[str, Any],
    *,
    verifier_fingerprint: str | None = None,
):
    anchors = case["anchors"]
    return verify_signed_verification_receipt(
        receipt,
        evidence,
        policy_path=policy,
        expected_artifact_digests=anchors["artifact_digests"],
        expected_schedule_digest=anchors["schedule_digest"],
        expected_runtime_digests=anchors["runtime_digests"],
        expected_pack_signer_fingerprint=anchors["evidence_signer_fingerprint"],
        expected_verifier_identity=anchors["verifier_identity"],
        expected_verifier_fingerprint=(
            verifier_fingerprint or anchors["verifier_fingerprint"]
        ),
    )


def _mutate(value: dict[str, Any], path: list[str], replacement: object) -> None:
    parent: dict[str, Any] = value
    for component in path[:-1]:
        child = parent[component]
        assert isinstance(child, dict)
        parent = child
    parent[path[-1]] = replacement


def test_v013_corpus_inventory_is_immutable() -> None:
    case = _case()
    evidence, receipt, _policy = _paths(case)
    package = FIXTURE_ROOT / "package"
    assert evidence.is_relative_to(package)
    assert receipt.is_relative_to(package)
    assert _policy.is_relative_to(package)
    files = {
        "manifest": evidence / "manifest.json",
        "receipt": receipt,
        "report": evidence / "reports/evaluation.report.json",
        "schedule": evidence / "schedule/runtime-behavioral-schedule.json",
    }

    assert {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in files.items()
    } == case["sha256"]


def test_v013_corpus_inventory_matches_its_canonical_generator() -> None:
    assert (FIXTURE_ROOT / "corpus.json").read_bytes() == _generator().generate()


def test_v013_pack_replays_under_its_original_semantics() -> None:
    case = _case()
    evidence, _receipt, policy = _paths(case)
    anchors = case["anchors"]

    result = verify_comparison_evidence(
        evidence,
        policy_path=policy,
        expected_artifact_digests=anchors["artifact_digests"],
        expected_schedule_digest=anchors["schedule_digest"],
        expected_runtime_digests=anchors["runtime_digests"],
        expected_signer_fingerprint=anchors["evidence_signer_fingerprint"],
    )
    report = _object(evidence / "reports/evaluation.report.json")

    assert result.status == 0
    assert result.payload["ok"] is True
    assert result.payload["integrity_ok"] is True
    assert result.payload["reports_verified"] is True
    assert result.payload["pack_format"] == case["expected"]["pack_format"]
    assert result.payload["policy_verdict"] == case["expected"]["technical_verdict"]
    assert report["format"] == case["expected"]["report_format"]
    assert report["metric"] == case["expected"]["metric"]
    assert report["record_count"] == case["expected"]["record_count"]


def test_v013_receipt_remains_parseable_and_is_not_relabelled() -> None:
    case = _case()
    evidence, receipt, policy = _paths(case)

    verified = _verify_receipt(receipt, evidence, policy, case)

    assert verified.ok is True
    assert verified.statement is not None
    assert verified.statement["format"] == case["expected"]["receipt_format"]
    assert verified.statement["verdict"]["policy_verdict"] == "pass"


@pytest.mark.parametrize(
    "mutation_name",
    ["tampered-technical-verdict", "silent-contract-relabel"],
)
def test_v013_receipt_mutations_fail_closed(tmp_path: Path, mutation_name: str) -> None:
    case = _case()
    evidence, receipt, policy = _paths(case)
    mutations = _object(FIXTURE_ROOT / "mutations.json")["cases"]
    mutation = next(item for item in mutations if item["name"] == mutation_name)
    changed = _object(receipt)
    _mutate(changed, mutation["path"], mutation["value"])
    mutated_receipt = tmp_path / "mutated.receipt.json"
    mutated_receipt.write_text(json.dumps(changed), encoding="utf-8")

    verified = _verify_receipt(mutated_receipt, evidence, policy, case)

    assert verified.ok is False
    assert mutation["expected_error"] in " ".join(verified.errors)


def test_v013_pack_tampering_fails_before_replay(tmp_path: Path) -> None:
    case = _case()
    evidence, _receipt, policy = _paths(case)
    copied = tmp_path / "evidence"
    shutil.copytree(evidence, copied)
    report_path = copied / "reports/evaluation.report.json"
    report = _object(report_path)
    report["verdict"] = "fail"
    report_path.chmod(stat.S_IMODE(report_path.stat().st_mode) | stat.S_IWUSR)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    anchors = case["anchors"]

    result = verify_comparison_evidence(
        copied,
        policy_path=policy,
        expected_artifact_digests=anchors["artifact_digests"],
        expected_schedule_digest=anchors["schedule_digest"],
        expected_runtime_digests=anchors["runtime_digests"],
        expected_signer_fingerprint=anchors["evidence_signer_fingerprint"],
    )

    assert result.status != 0
    assert result.payload["ok"] is False
    assert "checksum" in " ".join(result.payload["errors"]).lower()


def test_v013_semantically_inconsistent_but_signed_receipt_is_rejected(
    tmp_path: Path,
) -> None:
    case = _case()
    evidence, receipt, policy = _paths(case)
    changed = _object(receipt)
    statement = changed["statement"]
    statement["verdict"]["policy_verdict"] = "fail"
    key = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    public = key.public_key()
    public_raw = public.public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    fingerprint = "sha256:" + hashlib.sha256(public_raw).hexdigest()
    statement["verifier"]["signing_key_fingerprint"] = fingerprint
    changed["signature"]["public_key"]["value"] = public.public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("ascii")
    canonical_statement = (
        json.dumps(
            statement,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    import base64

    changed["signature"]["value"] = base64.b64encode(
        key.sign(canonical_statement)
    ).decode("ascii")
    inconsistent = tmp_path / "inconsistent.receipt.json"
    inconsistent.write_text(json.dumps(changed), encoding="utf-8")

    verified = _verify_receipt(
        inconsistent,
        evidence,
        policy,
        case,
        verifier_fingerprint=fingerprint,
    )

    assert verified.ok is False
    assert "successful verdict is inconsistent" in " ".join(verified.errors)
