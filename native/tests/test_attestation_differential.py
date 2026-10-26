"""Whole DSSE write/verify and recipient artifact identity cross-runtime parity."""

from __future__ import annotations

import base64
import copy
import dataclasses
import json
import os
import subprocess
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from invarlock.acceptance_attestation import (
    _artifact_path_digest,
    verify_acceptance_attestation,
    write_acceptance_attestation,
)
from tests.attestations.test_acceptance_attestation import (
    EVIDENCE,
    ISSUED_AT,
    RECEIPT,
    SUBJECT_DIGEST,
    _canonical,
    _envelope,
    _key,
    _payload,
    _policy,
    _resign,
)


class AttestationDifferential(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        executable = os.environ.get("INVARLOCK_ATTESTATION_BIN")
        if not executable:
            raise unittest.SkipTest("INVARLOCK_ATTESTATION_BIN is unset")
        cls.process = subprocess.Popen(
            [executable], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )

    @classmethod
    def tearDownClass(cls):
        cls.process.stdin.close()
        cls.process.wait(timeout=10)
        cls.process.stdout.close()

    def native(self, **request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        self.assertTrue(line, "native process terminated")
        return json.loads(line)

    def check_verify(self, envelope, public, fingerprint, policy, **kwargs):
        args = {
            "trusted_public_keys": {fingerprint: public},
            "recipient_policy": policy,
            "expected_subject_digest": SUBJECT_DIGEST,
            "now": ISSUED_AT + timedelta(minutes=5),
        }
        args.update(kwargs)
        expected = dataclasses.asdict(verify_acceptance_attestation(envelope, **args))
        expected["errors"] = list(expected["errors"])
        request = {
            "envelope_path": str(envelope),
            "trusted_public_keys": {
                k: {"path": str(p)} for k, p in args["trusted_public_keys"].items()
            },
            "recipient_policy": args["recipient_policy"],
            "expected_subject_digest": args.get("expected_subject_digest"),
            "now": args["now"].isoformat(),
        }
        if isinstance(request["recipient_policy"], Path):
            request["recipient_policy"] = {"path": str(request["recipient_policy"])}
        if args.get("subject_artifact_path") is not None:
            request["subject_artifact_path"] = str(args["subject_artifact_path"])
        actual = self.native(**request)
        self.assertIn("ok", actual, actual)
        self.assertEqual(actual["ok"], expected)

    def test_canonical_writer_and_no_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            private, public, fingerprint = _key(root, "signer")
            kw = {
                "signing_key_path": private,
                "signer_identity": "envelope-signer.example/release-assurance",
                "policy_identity": "evaluation.example/policies/release-regression-v3",
                "issued_at": ISSUED_AT,
                "evaluation_completed_at": ISSUED_AT - timedelta(minutes=5),
            }
            expected = write_acceptance_attestation(
                RECEIPT, EVIDENCE, root / "python.json", **kw
            )
            request = {
                "operation": "write",
                "receipt_path": str(RECEIPT),
                "evidence_path": str(EVIDENCE),
                "output_path": str(root / "rust.json"),
                "signing_key_path": str(private),
                "signer_identity": kw["signer_identity"],
                "policy_identity": kw["policy_identity"],
                "issued_at": kw["issued_at"].isoformat(),
                "evaluation_completed_at": kw["evaluation_completed_at"].isoformat(),
            }
            actual = self.native(**request)
            self.assertIn("ok", actual, actual)
            self.assertEqual(actual["ok"]["statement"], expected.statement)
            self.assertEqual(actual["ok"]["signer_fingerprint"], fingerprint)
            self.assertEqual(
                (root / "rust.json").read_bytes(), (root / "python.json").read_bytes()
            )
            self.assertEqual((root / "rust.json").stat().st_mode & 0o777, 0o444)
            self.assertEqual(
                self.native(**request),
                {"error": "attestation destination already exists: rust.json"},
            )
            self.check_verify(
                root / "rust.json", public, fingerprint, _policy(fingerprint)
            )

    def _mutate_statement(self, mutation, statement):
        p = statement["predicate"]
        if mutation == "statement_fields":
            statement["extra"] = True
        if mutation == "predicate_type":
            statement["predicateType"] = "https://wrong.example/predicate"
        if mutation == "subject":
            statement["subject"][0]["digest"]["sha256"] = "a" * 64
        if mutation == "receipt_digest":
            p["receipt"]["digest"] = "sha256:" + "a" * 64
        if mutation == "receipt_raw":
            p["receipt"]["raw_base64"] = "!"
        if mutation == "receipt_content":
            p["receipt"]["content"]["statement"]["verifier"]["identity"] = "different"
        if mutation == "receipt_signature":
            p["receipt"]["content"]["signature"]["value"] = base64.b64encode(
                b"a" * 64
            ).decode()
        if mutation == "receipt_contract":
            p["contracts"]["receipt"] = "invarlock/evidence-verification-receipt-v1"
        if mutation == "technical":
            p["technical_verdict"]["policy_verdict"] = "fail"
        if mutation == "receipt_time":
            p["timestamps"]["receipt_issued_at"] = ISSUED_AT.isoformat()
        if mutation == "receipt_raw_invalid_and_digest":
            p["receipt"]["raw_base64"] = base64.b64encode(b"not json").decode()
            p["receipt"]["digest"] = "sha256:" + "a" * 64
        if mutation == "artifact_payload":
            p["subject"]["artifact_identity_payload"] = base64.b64encode(b"{}").decode()
        if mutation == "artifact_digest":
            p["subject"]["artifact_digest"] = "sha256:" + "a" * 64
        if mutation == "source_digest":
            p["evaluation_source"]["identity_digest"] = "sha256:" + "a" * 64
        if mutation == "schedule":
            p["evaluation_source"]["schedule_digest"] = "sha256:" + "a" * 64
        if mutation == "policy_digest":
            p["policy"]["digest"] = "sha256:" + "a" * 64
        if mutation == "envelope_signer":
            p["signers"]["envelope"]["fingerprint"] = "sha256:" + "a" * 64
        if mutation == "relationship":
            p["signers"]["relationship"] = "same_signer"

    def test_envelope_receipt_policy_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            envelope, public, fingerprint = _envelope(root)
            original = envelope.read_bytes()
            original_statement = _payload(envelope)
            private = root / "envelope-signer.private.pem"
            for mutation in [
                None,
                "unknown_key",
                "payload",
                "signature",
                "statement_fields",
                "predicate_type",
                "subject",
                "receipt_digest",
                "receipt_raw",
                "receipt_raw_invalid_and_digest",
                "receipt_content",
                "receipt_signature",
                "receipt_contract",
                "technical",
                "receipt_time",
                "artifact_payload",
                "artifact_digest",
                "source_digest",
                "schedule",
                "policy_digest",
                "envelope_signer",
                "relationship",
                "revoked",
                "receipt_revoked",
                "version",
                "countersign",
                "stale",
                "future",
                "evidence_age",
                "trust_profile",
                "duplicate_trust",
                "wrong_subject",
                "no_subject",
                "naive_time",
            ]:
                envelope.chmod(0o600)
                envelope.write_bytes(original)
                statement = copy.deepcopy(original_statement)
                policy = _policy(fingerprint)
                kwargs = {}
                if mutation == "unknown_key":
                    kwargs["trusted_public_keys"] = {}
                if mutation in ["payload", "signature"]:
                    outer = json.loads(original)
                    if mutation == "payload":
                        outer["payload"] = "!"
                    else:
                        outer["signatures"][0]["sig"] = base64.b64encode(
                            b"a" * 64
                        ).decode()
                    envelope.write_bytes(_canonical(outer))
                self._mutate_statement(mutation, statement)
                if mutation == "revoked":
                    policy["trusted_signers"][0]["status"] = "revoked"
                if mutation == "receipt_revoked":
                    policy["trusted_receipt_verifiers"][0]["status"] = "revoked"
                if mutation == "version":
                    policy["allowed_contract_versions"] = ["0.13.0"]
                if mutation == "countersign":
                    policy["allow_countersigned_receipts"] = False
                if mutation == "stale":
                    kwargs["now"] = ISSUED_AT + timedelta(days=1)
                if mutation == "future":
                    kwargs["now"] = ISSUED_AT - timedelta(seconds=1)
                if mutation == "evidence_age":
                    policy["freshness"]["max_evidence_age_seconds"] = 3600
                if mutation == "trust_profile":
                    policy["expected_receipt_trust_profile_digest"] = (
                        "sha256:" + "a" * 64
                    )
                if mutation == "duplicate_trust":
                    policy["trusted_signers"].append(
                        dict(policy["trusted_signers"][0], status="revoked")
                    )
                if mutation == "wrong_subject":
                    kwargs["expected_subject_digest"] = "sha256:" + "a" * 64
                if mutation == "no_subject":
                    kwargs["expected_subject_digest"] = None
                if mutation == "naive_time":
                    kwargs["now"] = ISSUED_AT.replace(tzinfo=None)
                if statement != original_statement:
                    _resign(envelope, private, statement)
                with self.subTest(mutation=mutation):
                    self.check_verify(envelope, public, fingerprint, policy, **kwargs)

    def test_file_checkpoint_and_engine_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            file = root / "model.gguf"
            file.write_bytes(b"GGUF\0" * 10000)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text('{"model_type":"test"}')
            (checkpoint / "model.bin").write_bytes(b"weights")
            (checkpoint / "a").mkdir()
            (checkpoint / "a" / "nested.bin").write_bytes(b"nested")
            (checkpoint / "a.bin").write_bytes(b"sibling")
            (checkpoint / "logs").mkdir()
            (checkpoint / "logs" / "ignored").symlink_to(file)
            (checkpoint / "training_receipt.json").write_text("ignored")
            engine = root / "engine"
            engine.mkdir()
            (engine / "config.json").write_text(
                json.dumps(
                    {
                        "version": "1.0",
                        "pretrained_config": {
                            "architecture": "LlamaForCausalLM",
                            "dtype": "float16",
                            "mapping": {"world_size": 2},
                        },
                        "build_config": {"max_batch_size": 1},
                    }
                )
            )
            (engine / "rank0.engine").write_bytes(b"engine0")
            (engine / "rank1.engine").write_bytes(b"engine1")
            for p, kind in [
                (file, "file_sha256"),
                (checkpoint, "hf_snapshot_tree_sha256"),
                (engine, "tensorrt_llm_engine_tree_sha256"),
            ]:
                expected = _artifact_path_digest(p, digest_kind=kind)
                self.assertEqual(
                    self.native(operation="artifact", path=str(p), digest_kind=kind),
                    {"ok": expected},
                )


if __name__ == "__main__":
    unittest.main()
