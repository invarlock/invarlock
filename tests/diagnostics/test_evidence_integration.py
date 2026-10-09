"""The opt-in numerical method travels through the existing signed envelope."""

from __future__ import annotations

from invarlock.diagnostics import canonical_observation_bytes, rmt_observation
from invarlock.evidence_pack import EvidenceObservation, verify_comparison_evidence
from tests.evidence_packs.test_evidence_pack import _publish, _verification_anchors


def test_smaller_gram_is_authenticated_without_changing_acceptance(tmp_path):
    payload = canonical_observation_bytes(
        rmt_observation([[-1.0] * 8, [1.0] * 8], method="smaller_gram")
    )
    observation = EvidenceObservation(
        observation_id="rmt-summary", scope="subject", kind="rmt", payload=payload
    )
    pack, policy, fingerprint, runtimes, _key, arguments = _publish(
        tmp_path, observations=(observation,)
    )
    result = verify_comparison_evidence(
        pack,
        policy_path=policy,
        **_verification_anchors(arguments),
        expected_runtime_digests=runtimes,
        expected_signer_fingerprint=fingerprint,
    )
    assert result.status == 0
    assert result.payload["policy_verdict"] == "pass"
    assert result.payload["observations"][0]["kind"] == "rmt"
    path = pack / "observations/rmt-summary.json"
    raw = path.read_bytes()
    assert b"column_standardized_smaller_gram_eigh" in raw
    path.chmod(0o644)
    path.write_bytes(
        raw.replace(
            b"column_standardized_smaller_gram_eigh",
            b"column_standardized_covariance_eigh",
        )
    )
    rejected = verify_comparison_evidence(
        pack,
        policy_path=policy,
        **_verification_anchors(arguments),
        expected_runtime_digests=runtimes,
        expected_signer_fingerprint=fingerprint,
    )
    assert rejected.payload["integrity_ok"] is False
