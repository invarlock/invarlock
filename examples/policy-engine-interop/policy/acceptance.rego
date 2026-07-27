package invarlock.acceptance

import rego.v1

default allow := false

valid_input if {
	input.format == "invarlock/acceptance-policy-input-v1"
}

authenticated if {
	input.authentication.envelope_signature == true
	input.authentication.receipt_signature == true
	input.authentication.projection_consistent == true
}

valid_statement if {
	input.statement._type == "https://in-toto.io/Statement/v1"
	input.statement.predicateType == input.recipient.expected_predicate_type
	input.statement.predicate.format == "invarlock/acceptance-predicate-v2"
}

valid_subject if {
	count(input.statement.subject) == 1
	subject := input.statement.subject[0]
	subject.name == input.recipient.expected_subject.name
	subject.digest.sha256 == input.recipient.expected_subject.sha256
	input.statement.predicate.subject.name == subject.name
	input.statement.predicate.subject.artifact_digest == sprintf("sha256:%s", [subject.digest.sha256])
}

supported_contract if {
	some version in input.recipient.allowed_contract_versions
	version == input.statement.predicate.contracts.invarlock_release
}

trusted_signer if {
	some signer in input.recipient.trusted_signers
	signer.status == "active"
	signer.identity == input.verified.envelope_signer_identity
	signer.fingerprint == input.verified.envelope_signer_fingerprint
}

trusted_receipt_verifier if {
	some verifier in input.recipient.trusted_receipt_verifiers
	verifier.status == "active"
	verifier.identity == input.verified.receipt_verifier_identity
	verifier.fingerprint == input.verified.receipt_verifier_fingerprint
}

fresh if {
	age := input.verified.now_unix - input.verified.attestation_issued_at_unix
	age >= 0
	age <= input.recipient.max_attestation_age_seconds
}

technical_verdict_allowed if {
	verdict := input.statement.predicate.technical_verdict
	verdict.ok == true
	verdict.integrity_ok == true
	verdict.verification_status == 0
	verdict.policy_verdict == input.recipient.required_technical_verdict
}

reasons contains "unsupported_input" if {
	not valid_input
}

reasons contains "authentication_failed" if {
	not authenticated
}

reasons contains "statement_contract_rejected" if {
	not valid_statement
}

reasons contains "subject_rejected" if {
	not valid_subject
}

reasons contains "unsupported_contract" if {
	not supported_contract
}

reasons contains "untrusted_signer" if {
	not trusted_signer
}

reasons contains "untrusted_receipt_verifier" if {
	not trusted_receipt_verifier
}

reasons contains "stale_evidence" if {
	not fresh
}

reasons contains "technical_verdict_rejected" if {
	not technical_verdict_allowed
}

allow if {
	count(reasons) == 0
}

decision := {
	"allow": allow,
	"reasons": sort([reason | reasons[reason]]),
}
