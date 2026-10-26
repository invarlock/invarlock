"""Actual installed PyO3 transaction plus explicitly authorized Python scorer."""

import pytest

from invarlock.evaluation_transaction import evaluate_request_file
from invarlock.evidence_verification import EvidenceVerificationError, verify_evidence
from invarlock.native_backend import transact
from tests.cli.test_import_journey import (
    _input_anchors,
    _key,
    _materialize_request,
    _text_scorer_registry_and_binding,
)


def test_authorized_scorer_bridge_and_public_sdk(tmp_path):
    root = tmp_path.resolve()
    registry, binding = _text_scorer_registry_and_binding()
    material = _materialize_request(
        root, scorer_binding=binding, scorer_registry=registry
    )
    key, signer = _key(root / "evidence.pem")
    verifier, _ = _key(root / "verifier.pem")
    publication = evaluate_request_file(
        material["request"], signing_key_path=key, scorer_registry=registry
    )
    evidence = publication.evidence_path
    anchors = _input_anchors(evidence)
    options = {
        "policy_path": material["policy"],
        "expected_baseline_artifact": anchors["baseline"],
        "expected_subject_artifact": anchors["subject"],
        "expected_schedule": anchors["dataset"],
        "expected_baseline_runtime": material["runtime_digests"]["baseline"],
        "expected_subject_runtime": material["runtime_digests"]["subject"],
        "expected_signer": signer,
        "verifier_signing_key_path": verifier,
        "verifier_identity": "bridge-test",
        "scorer_registry": registry,
    }
    # Same receipt basename in separate directories makes the public payload equal.
    for name in ("python", "rust"):
        (root / name).mkdir()
    reference = verify_evidence(
        evidence, **options, receipt_path=root / "python/receipt.json"
    )
    actual = verify_evidence(
        evidence, **options, receipt_path=root / "rust/receipt.json", backend="rust"
    )
    assert actual.payload == reference.payload
    assert (root / "rust/receipt.json").read_bytes() == (
        root / "python/receipt.json"
    ).read_bytes()
    with pytest.raises(EvidenceVerificationError) as absent:
        verify_evidence(
            evidence,
            **{**options, "scorer_registry": None},
            receipt_path=root / "absent.json",
            backend="rust",
        )
    assert "explicitly authorized scorer registry" in str(absent.value)

    # Reentrant callback runs a native transaction while the outer GIL is released.
    class Reentrant:
        def replay(self, request):
            assert transact("canonicalize", {"hello": "world"})["payload"] == {
                "hello": "world"
            }
            return registry.replay(request)

    again = verify_evidence(
        evidence,
        **{**options, "scorer_registry": Reentrant()},
        receipt_path=root / "again.json",
        backend="rust",
    )
    assert again.payload["ok"]


def test_captured_immutable_inputs(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "experiments/rust/python")
    )
    from invarlock_rust_experiment.fixtures import make_fixture

    from native.tests.test_evidence_differential import config_for

    root = tmp_path.resolve()
    make_fixture(root / "fixture", 12, "pass")
    options = config_for(root / "fixture", root / "receipt.json")
    evidence = options.pop("evidence_path")
    for name in ("policy_path", "verifier_signing_key_path", "receipt_path"):
        options[name] = Path(options[name])
    policy = Path(options["policy_path"]).read_bytes()
    key = Path(options["verifier_signing_key_path"]).read_bytes()
    Path(options["policy_path"]).write_bytes(b"changed after snapshot")
    Path(options["verifier_signing_key_path"]).write_bytes(b"changed after snapshot")
    result = verify_evidence(
        Path(evidence),
        **options,
        policy_bytes=policy,
        verifier_signing_key_bytes=key,
        backend="rust",
    )
    assert result.payload["ok"]
