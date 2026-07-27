package acceptance

import "list"

format: "invarlock/acceptance-policy-input-v1"
authentication: {
	envelope_signature:    true
	receipt_signature:     true
	projection_consistent: true
}
recipient: {
	allowed_contract_versions: [...string] & [_, ...]
	expected_predicate_type: "https://invarlock.dev/attestations/acceptance/v2"
	expected_subject: {
		name:   string
		sha256: =~"^[a-f0-9]{64}$"
	}
	max_attestation_age_seconds: int & >=0
	required_technical_verdict:  "pass" | "fail"
	trusted_receipt_verifiers: [...{
		fingerprint: =~"^sha256:[a-f0-9]{64}$"
		identity:    string
		status:      "active" | "revoked"
	}]
	trusted_signers: [...{
		fingerprint: =~"^sha256:[a-f0-9]{64}$"
		identity:    string
		status:      "active" | "revoked"
	}]
}
statement: {
	_type:         "https://in-toto.io/Statement/v1"
	predicateType: recipient.expected_predicate_type
	subject: [{
		name: recipient.expected_subject.name
		digest: sha256: recipient.expected_subject.sha256
	}]
	predicate: {
		format: "invarlock/acceptance-predicate-v2"
		contracts: invarlock_release: string
		subject: {
			name:            statement.subject[0].name
			artifact_digest: "sha256:\(statement.subject[0].digest.sha256)"
		}
		technical_verdict: {
			ok:                  true
			integrity_ok:        true
			verification_status: 0
			policy_verdict:      recipient.required_technical_verdict
		}
	}
}
verified: {
	attestation_issued_at_unix:   int
	now_unix:                     int
	envelope_signer_fingerprint:  =~"^sha256:[a-f0-9]{64}$"
	envelope_signer_identity:     string
	receipt_signature:            true
	receipt_verifier_fingerprint: =~"^sha256:[a-f0-9]{64}$"
	receipt_verifier_identity:    string
}

_contractMatches: [
	for version in recipient.allowed_contract_versions
	if version == statement.predicate.contracts.invarlock_release {
		version
	},
]
_contractMatchCount: len(_contractMatches) & >=1

_trustedSigner: list.Contains(recipient.trusted_signers, {
	fingerprint: verified.envelope_signer_fingerprint
	identity:    verified.envelope_signer_identity
	status:      "active"
})
_trustedSigner: true

_trustedReceiptVerifier: list.Contains(recipient.trusted_receipt_verifiers, {
	fingerprint: verified.receipt_verifier_fingerprint
	identity:    verified.receipt_verifier_identity
	status:      "active"
})
_trustedReceiptVerifier: true

_ageSeconds: verified.now_unix - verified.attestation_issued_at_unix
_ageSeconds: >=0 & <=recipient.max_attestation_age_seconds
