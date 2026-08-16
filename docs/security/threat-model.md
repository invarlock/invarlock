# Threat model

This threat model covers the `evaluate`, `verify`, and `report` transactions,
the portable acceptance-attestation handoff, and the public provider
contracts. It distinguishes attacks that the evidence verifier or acceptance
verifier can detect from claims that require controls outside InvarLock.

!!! warning "Security guidance"

    **In plain language:** InvarLock can detect tampering and inconsistent
    evidence, but it cannot prove that an honest runtime executed or that the
    chosen test and threshold are sufficient.

    **Objective:** Identify threats to one evidence transaction and distinguish
    verifier-enforced properties from risks that require deployment controls.

    **Assets or boundary:** The `evaluate`, `verify`, `report`, and portable
    acceptance data flows, provider-contract inputs, signing identities,
    recipient policy, independent anchors, and signed outputs; host and
    accelerator security remain external boundaries.

    **Use this page when:** Performing architecture review, assigning controls,
    interpreting a verifier failure, or deciding whether a deployment needs
    attestation or safeguards beyond InvarLock.

## Security objectives

| Objective | Required property |
| --- | --- |
| Bundle integrity | Every accepted file is inventoried, checksummed, path-safe, and transitively bound by the signed manifest. |
| Evidence signer authenticity | The manifest signature verifies under the independently expected evidence-signer fingerprint. |
| Input and runtime binding | Request, artifact, schedule, provider, observation, and runtime identities agree across all contract layers. |
| Replay correctness | Pairing, built-in scores or explicitly authorized scorer results, comparison arithmetic, paired interval, optional count/width and exact-match side-accuracy qualification, and policy verdict are reconstructed from lower-level facts. |
| Independent acceptance | Policy, artifact, schedule, runtime, evidence-signer, and GGUF normalized-request anchors are controlled outside the submitted evidence path. |
| Receipt accountability | The verdict and exact anchors are bound to a separately recorded verifier identity and key. |
| Recipient-controlled transport acceptance | The recipient independently authorizes both the DSSE envelope signer and technical receipt verifier, binds the exact subject, and applies current freshness and contract policy. |
| Fail-closed handling | Missing, malformed, unknown, partial, extra, or inconsistent material cannot become accepted evidence. |

Availability and confidentiality are not primary bundle-verification
objectives. Size limits and safe path handling reduce resource abuse, while
deployment controls must protect sensitive model, dataset, and key material.

## Assets

- baseline and subject artifact identities;
- schedule, inputs, expected outputs, and dataset coordinates;
- provider observations, receipts, runtime manifests, and support resources;
- acceptance policy and artifact, schedule, and runtime trust anchors;
- scorer binding, configuration, implementation, and independent authorization
  when an extension is selected;
- canonical evidence pack and signed verification receipt;
- the acceptance predicate, DSSE envelope, and exact embedded receipt bytes;
- recipient-controlled envelope-signer and receipt-verifier registries,
  freshness rules, contract-version rules, and expected subject binding;
- evidence signer and verifier private keys; and
- the authorization mapping from actor identities to public-key fingerprints.

## Adversaries and failure sources

The model considers accidental corruption, unsafe input, a malicious evidence
submitter, compromised evidence signer or verifier environments, compromised provider
add-ins, key theft, and a decision owner who obtains trust anchors from the submitted
evidence. It also considers nonmalicious selection bias and numerical drift.

The model does not assume that an evidence signature makes the evidence signer honest.
The verifier treats the bundle as untrusted until replay succeeds, but replay
can only reason about the facts in that bundle.

## Adversary capability classes

| Class | Capability | Expected outcome |
| --- | --- | --- |
| Transport attacker | Can add, remove, reorder, or alter bundle files but lacks authorized private keys | Detected by inventory, checksum, canonical-form, or signature verification |
| Unauthorized evidence signer | Can create internally consistent evidence and sign with an unrecognized key | Rejected by independently pinned evidence-signer fingerprint |
| Authorized malicious evidence signer | Controls provider facts and authorized evidence-signing key | Can fabricate mutually consistent evidence; requires independent execution control or attestation to address |
| Configuration attacker | Attempts to replace policy, artifact expectations, schedule, runtime expectations, provider or scorer support, or request paths | Independent anchors, explicit scorer authorization, and root-confined resolution detect covered substitutions |
| Selection attacker | Chooses favorable schedule, baseline, subject build, seed, retry, or stopping rule | Not detectable from one internally valid pack; prevented or exposed by advance commitment and attempt retention |
| Verifier attacker | Controls verifier host/key or anchor source | Can issue misleading receipts; requires independent verifier authorization, key custody, and audit |
| Receipt-verification error | Trusts embedded keys, HTML, or copied summaries without independent checks | Prevented only by independent receipt verification and authorization discipline |
| Authorized malicious envelope signer | Controls an authorized DSSE key and can wrap or rewrap arbitrary material | Cannot authorize an unknown technical receipt verifier, renew receipt-authenticated evidence freshness, or change the exact subject without recipient-policy rejection |
| Recipient-policy configuration attacker | Adds contradictory, duplicate, revoked, or attacker-controlled trust records | Duplicate identity/fingerprint pairs and zero-or-multiple signer matches fail closed; policy authorization and distribution remain deployment responsibilities |

## Threats, controls, and residual risk

| Threat | InvarLock control | Residual risk |
| --- | --- | --- |
| File changed, removed, inserted, or renamed after publication | Complete checksummed inventory, manifest binding, no-extra-files check, canonical paths, and evidence signature | An evidence signer can sign fabricated but internally consistent files. |
| Attacker substitutes its own signed bundle | Verifier requires an evidence-signer fingerprint obtained independently | Fingerprint authorization, revocation, and secure distribution are external. |
| Bundle weakens or replaces acceptance policy | Verifier reads policy bytes from a caller-owned path and requires their digest to match the bundle identity and report | The external policy may itself be poorly chosen or compromised. |
| Bundle substitutes the baseline or subject | Verifier requires caller-owned artifact-identity digests and cross-checks both sides through the request, provider evidence, runtime manifests, report, and receipt | Digest equality identifies declared authenticated material; it does not establish suitability or execution. |
| Bundle substitutes the evaluation records | Verifier requires a caller-owned canonical schedule digest and reconstructs the ordered schedule and record bindings | A schedule approved in advance can still be unrepresentative or insufficient. |
| Bundle claims another runtime | Verifier requires independent per-side image digests and cross-checks manifests, identities, and provider receipts | Digest equality does not prove that the image executed. |
| Baseline and subject use different records | Canonical schedule, unique record IDs, input digests, exact ordered pairing, and record-level replay | The shared schedule can still be unrepresentative or cherry-picked. |
| Provider supplies a false aggregate | InvarLock ignores provider aggregates for acceptance and derives per-record scores and the report | A malicious provider can fabricate the underlying record facts. |
| Bundle selects or substitutes scorer code | Request binding and independent policy pin scorer identity, descriptor, and configuration; verifier requires an explicitly authorized registry and repeats canonical replay | Authorized scorer code executes in the verifier trust boundary and can be malicious or flawed; code review and controlled distribution remain external. |
| Observation or sidecar is transplanted between artifacts or runs | Artifact, provider, schedule, runtime, observation, report, and receipt cross-bindings | A fully compromised evidence signer can fabricate all bindings together. |
| Error or record with a value that is not finite is hidden | Every scheduled record must be present, successful, finite where scored, and included in the canonical record-array digest | An evidence signer can select a different successful run unless run-selection policy is external. |
| Unsafe request path or YAML feature escapes the request root | Strict schema, bounded YAML, no aliases/includes/tags, relative no-follow path traversal, and output revalidation | The surrounding host and dependencies remain trusted computing base. |
| Host launch substitutes an image, engine argument, device, or input mount | The host CLI requires per-side digest agreement, uses argv-only Docker/Podman execution with `--pull=never`, allowlists environment, validates device selection, and mounts each side's job, artifact, and support material read-only with an isolated writable output | The container engine, host coordinator, kernel, and device remain trusted computing base. |
| Malicious model or provider code executes during evaluation | Built-in strict HF path uses local safetensors, disables remote code and network access, and authenticates checkpoint/tokenizer material | Native libraries, container runtime, kernel, driver, add-ins, and optional backends can contain vulnerabilities. InvarLock is not a sandbox. |
| Human report differs from machine evidence | Renderer authenticates the bundle and reads the canonical bound report | Screenshots, copied text, or externally modified HTML are not acceptance records. Verify the bundle and receipt. |
| Evidence signer or verifier private key is stolen | Ed25519 signatures expose stable fingerprints suitable for pinning and rotation | Key storage, compromise detection, revocation, and incident response are external. |
| Trusted envelope signer substitutes a self-signed technical receipt | Acceptance verification authenticates the embedded receipt and requires exactly one independently trusted receipt-verifier identity/fingerprint record | The recipient must maintain and securely distribute the receipt-verifier registry and its revocation state. |
| Old evidence is placed in a newly issued envelope | Envelope age and receipt-authenticated evidence age are evaluated separately; missing authoritative evidence time rejects when an evidence-age limit is configured | v0.13 receipts have no authenticated issuance time and cannot satisfy a recipient policy that requires bounded evidence age. |
| Recipient policy contains contradictory duplicate trust records | Both trust registries reject repeated identity/fingerprint pairs regardless of array order or status, and signer lookup requires exactly one match | The engine cannot decide which identities or keys the recipient should authorize. |
| A historical receipt is reformatted during wrapping | The predicate authenticates `receipt.raw_base64` and its digest while requiring parsed content to agree with those exact supplied bytes | Byte preservation does not make the historical receipt current or change its original contract semantics. |

## Trust-boundary data flow

The critical transitions are:

```text
host CLI -> separately pinned per-side OCI workers -> paired records -> host-signed pack
signed pack + independent anchors -> verifier replay -> signed receipt
signed receipt + bound evidence -> DSSE envelope -> recipient verifier + exact subject + current policy
signed pack -> authenticated renderer -> console/HTML
```

At the first transition, the host prepares the canonical schedule and launches
one worker for each side with network disabled, reduced privileges, a read-only
container root, read-only job and model material, and an isolated writable
output. The evidence-signing key remains host-side; workers never receive it.
The host validates both closed outputs before it creates and signs the pack.
Workers sharing a generic or identical CUDA selection run sequentially; workers
on distinct explicit CUDA indexes may run in parallel. Model and
runtime-integration code can still affect reported facts. At the second, the
verifier treats every pack byte as hostile and must keep policy, artifact,
schedule, runtime, and signer anchor sources outside the pack. At the third,
the recipient authenticates both signer roles from separate registries, checks
the exact subject, applies independent envelope and evidence freshness, and
evaluates its current policy. At the fourth, presentation is read-only and
carries no acceptance authority. The
[evidence signer/verifier diagram](../assets/evidence-signer-verifier-trust.svg) shows the
separate signing boundaries.

## Key attack scenarios

### Mutually consistent fabricated execution

A compromised evidence signer writes plausible artifact identities, runtime facts,
provider observations, reports, and a runtime digest without running the model,
then signs the pack. Internal replay can pass because every submitted binding
agrees.

**What helps:** independent evidence-signer authorization prevents an arbitrary
key from being accepted; external artifact, schedule, and runtime digests keep
the pack from choosing different declared inputs; an independent rerun or
measured-execution attestation can test the execution claim.

**What does not help:** checksums, an evidence signature, and digest agreement do
not prove execution or measurement truth.

### Biased schedule or run selection

An evidence signer selects favorable records, expected outputs, baseline, subject
build, seed, or run after observing results. Exact pairing and deterministic
replay still pass.

**Mitigation:** approve and retain the schedule digest, policy, artifacts, run
selection rule, and stopping rule before subject results are visible. Review
failed and superseded attempts according to the decision process. Use a held-
out or independently selected schedule where appropriate.

### Circular trust anchors

A decision owner copies the expected evidence-signer fingerprint, policy,
artifact identities, schedule digest, runtime digest, or GGUF request digest from the pack and
passes those values to `verify`. Verification can establish internal
consistency but not independent acceptance.

**Mitigation:** obtain all anchors from authorized configuration or another
independent channel. Compare the signed receipt with that source before acting.
For GGUF, approve the execution-free preflight request digest before evaluation;
the verifier uses it to reject substituted llama.cpp backend, artifact, or
execution settings even when the submitted bundle is internally consistent.

### Compromised verifier

A compromised verifier can sign a misleading receipt or use unauthorized
anchors. Receipt verifiers that trust only the embedded verifier key may accept it.

**Mitigation:** pin the expected verifier fingerprint and identity outside the
receipt, protect verifier keys separately from evidence-signing keys, and retain the
policy, artifact, schedule, and runtime sources used for the decision. Receipt
verification must use those external expectations.

### Key compromise and replayed history

An attacker obtains an authorized evidence signer or verifier private key and signs
new material. Cryptographic verification succeeds until receipt verifiers learn that
the key is no longer authorized.

**Mitigation:** maintain activation, expiry, and revocation records; timestamp
or log approval events in a trusted system; rotate keys with explicit
predecessor/successor records; and preserve the decision-time authorization
state. InvarLock does not provide trusted time or revocation lookup.

After compromise, identify every bundle or receipt signed during the exposure
window, suspend reliance, re-verify source evidence, and reissue receipts under
a newly selected verifier key where appropriate. Re-signing fabricated
evidence does not repair it; obtain new evidence from a trusted evaluation
environment.

### Trusted transport signer substitutes technical authority

An authorized envelope signer wraps a receipt issued by an attacker-controlled
technical verifier and supplies that verifier's embedded public key. The outer
DSSE signature is valid, but it does not authorize the inner technical
decision.

**Mitigation:** maintain separate recipient-controlled envelope-signer and
receipt-verifier registries. Acceptance verification authenticates the inner
receipt and requires exactly one active receipt-verifier record matching both
identity and fingerprint. Embedded key material alone is not a trust anchor.

### Rewrapping renews old evidence

An envelope signer places an old or undated receipt in a newly issued envelope
and attempts to satisfy a current freshness policy using only the envelope
time.

**Mitigation:** constrain envelope age and receipt-authenticated evidence age
independently. A missing receipt issuance time rejects whenever evidence age is
bounded. Wrapper-supplied evaluation context cannot substitute for an
authenticated receipt timestamp.

### Denial of service through hostile evidence

A submitter provides oversized, deeply nested, symlinked, sparse, or malformed
material to consume verifier resources or escape the pack boundary.

**Mitigation:** contract size/count limits, strict JSON/YAML decoding, regular-
file requirements, path normalization, symlink rejection, exact inventory,
and bounded reads. Run verification with operating-system resource limits when
processing adversarial input; parser checks do not replace process isolation.

## Security boundaries

### Enforced by the engine

- strict, versioned and bounded request/evidence/provider documents;
- root-confined and no-follow request paths;
- authenticated file inventory and canonical encodings;
- Ed25519 evidence signature with caller-pinned fingerprint;
- exact provider, artifact, schedule, runtime, and record cross-bindings;
- verifier-owned deterministic metric or explicitly authorized scorer replay
  and policy arithmetic; and
- an external Ed25519 receipt binding independent anchors and verdict;
- an authenticated DSSE acceptance envelope that independently verifies the
  embedded receipt, exact subject, both signer roles, and current recipient
  policy; and
- exact historical receipt-byte preservation plus separate envelope and
  receipt-authenticated evidence freshness.

### Required from the deployment

- protected and separately authorized evidence-signing and verifier keys;
- independently maintained, duplicate-free envelope-signer and
  receipt-verifier registries for portable acceptance;
- independently distributed policy, artifact, schedule, runtime,
  evidence-signer, and verifier anchors;
- immutable artifact and runtime acquisition;
- trusted host, container engine, dependencies, drivers, provider code, and any
  authorized scorer implementation;
- representative schedule and run-selection governance;
- external execution attestation or independent rerun when required; and
- retention, revocation, and incident-response procedures.

The acceptance envelope is standards-shaped in-toto/DSSE transport. The
maintained interoperability example separately authenticates the envelope and
embedded receipt before OPA/Rego or CUE evaluates recipient policy over the
authenticated projection. It does not imply that arbitrary policy-engine
configurations perform signature verification, full evidence replay, or safe
acceptance without equivalent validation.

## Verification-failure handling

Do not make a verifier error disappear by deleting files, relaxing signer
pins, substituting the embedded policy, or changing runtime expectations. Use
this triage order:

1. preserve the original evidence and command output;
2. classify format/integrity, authentication, binding, replay, or policy
   failure;
3. determine whether the source transaction is invalid or infrastructure
   transport failed;
4. fix the evaluation or transport cause; and
5. create new evidence when any signed byte or evaluation input changes.

A valid policy failure is evidence of nonacceptance, not an infrastructure
error. An invalid bundle is no accepted evidence, not proof that the subject
failed the metric.

## Explicit non-goals

InvarLock does not claim to:

- prove that a model or runtime actually executed;
- verify semantic truth of provider measurements;
- establish population performance, confidence, or statistical power;
- detect every malicious checkpoint, dependency, backend, or native library;
- validate model safety, alignment, fairness, privacy, robustness, or content;
- authorize deployment or replace domain-specific review;
- secure a host, container engine, kernel, accelerator, or multi-tenant system;
- issue or revoke identities and keys;
- demonstrate external CUE, Open Policy Agent, or other policy-engine
  authentication or evaluation of the acceptance envelope; or
- establish the baseline as correct or trustworthy.

See [Security practices](best-practices.md) for operating guidance and the
[acceptance checklist](../assurance/acceptance-checklist.md) for one evidence
decision. Portable recipient verification is specified in
[Acceptance attestations](../reference/acceptance-attestations.md).

## References

- National Institute of Standards and Technology,
  [SP 800-190: Application Container Security Guide](https://doi.org/10.6028/NIST.SP.800-190),
  2017.
- National Institute of Standards and Technology,
  [SP 800-218: Secure Software Development Framework 1.1](https://doi.org/10.6028/NIST.SP.800-218),
  2022.
- National Institute of Standards and Technology,
  [SP 800-57 Part 1 Revision 5: Recommendation for Key Management](https://doi.org/10.6028/NIST.SP.800-57pt1r5),
  2020.
- in-toto project,
  [Attestation Framework Specification 1.2](https://github.com/in-toto/attestation/blob/v1.2.0/spec/README.md),
  2026.
