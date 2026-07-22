# Security practices

These practices preserve the trust separation described in the
[Trust model](trust-model.md). They supplement rather than replace host,
container, key-management, and model-safety controls.

!!! warning "Security guidance"

    **In plain language:** Keep signing authority, fixed inputs, and
    acceptance decisions separate so one compromised role cannot silently
    manufacture a trusted result.

    **Objective:** Operate evidence production and verification while
    preserving key separation, immutable inputs, independent anchors, and
    fail-closed handling.

    **Assets or boundary:** Evidence signer and verifier keys, artifact and runtime
    identities, schedules, policies, evidence bundles, receipts, and the host
    controls outside InvarLock's evidence contract.

    **Use this page when:** Deploying or reviewing an InvarLock workflow,
    defining custody and retention controls, or responding to suspected key,
    runtime, or artifact compromise.

## Operating assumptions and residual risk

These practices assume that the deployment can keep private keys out of
untrusted provider code, maintain an authoritative fingerprint and anchor
source, restrict access to materialized inputs, and preserve immutable outputs.
If those capabilities are absent, signature and replay checks can still expose
inconsistency but cannot create the missing trust separation.

The controls below reduce accidental substitution, key misuse, mutable-input
drift, and unsafe artifact handling. They do not attest execution, prove that a
provider reported truthful measurements, establish that a schedule or
threshold is sufficient, or harden the host and accelerator. Use the
[Threat model](threat-model.md) to decide which residual risks require external
controls.

The stock OCI run path uses a single orchestrating process on the host plus two
model workers. The host loads the evidence-signing key, prepares the canonical
schedule, launches the separately pinned workers, validates their closed
outputs, and signs the finalized bundle. Each worker sees only its own
read-only artifact and support material, the read-only side job, and an
isolated writable output directory. The signing key remains host-side and is
never mounted into model workers.

The host coordinator is still a signing trust boundary: compromise of that
process can reach file-backed key material and can misstate worker execution.
InvarLock does not currently expose a KMS, HSM, or remote-signer interface.
Treat file-backed signing as the baseline implementation. A deployment needing
a stronger signing boundary can create complete records in isolated execution,
then use import mode in a controlled signing environment until an external
signer interface is available.

## Separate roles and keys

Use different Ed25519 keys for evidence production and independent
verification. Generate keys under a restrictive umask when a managed signing
service is unavailable:

```bash
umask 077
openssl genpkey -algorithm ED25519 -out evidence-signer.pem
openssl genpkey -algorithm ED25519 -out evidence-verifier.pem
```

- keep private keys outside request, evidence, logs, shell history, images, and
  source control;
- authorize evidence-signer and verifier fingerprints through an authorized channel;
- pin the expected evidence-signer fingerprint during every verification;
- pin the verifier identity and fingerprint when consuming receipts;
- document rotation and revocation; and
- treat any output signed after suspected compromise as untrusted until
  independently assessed.

An evidence pack records its signing-key fingerprint in `manifest.json`. A signed
receipt records the verifier fingerprint. Those embedded values identify the
signing keys used; they are not their own authorization source.

### Key-custody baseline

| Lifecycle stage | Minimum practice |
| --- | --- |
| Generation | Generate evidence signer and verifier keys in separate controlled contexts with cryptographically secure system randomness. |
| Storage | Keep private keys in a managed signer, hardware-backed store, or access-controlled file unavailable to untrusted model/provider code. |
| Authorization | Bind the raw-public-key fingerprint to role, owner, scope, activation, and expiry in authorized configuration. |
| Use | Restrict each key to its role; log signing requests without logging private material or sensitive evidence content. |
| Backup | Encrypt, access-control, inventory, and recovery-test any backup; omit backup if recovery policy prefers rotation. |
| Rotation | Authorize the successor before use, stop new use of the predecessor, and retain historical authorization needed to interpret old signatures. |
| Revocation | Publish the affected fingerprint and effective time through the same or stronger channel used for authorization. |
| Destruction | Remove active, cached, staged, and backup copies under a documented process. |

File mode `0600` is useful hygiene but not isolation from the same user,
administrator, debugger, memory reader, or compromised process. High-impact
signing should place the key behind a narrowly authorized signing interface.
That last control is deployment guidance: the current CLI accepts a local
PKCS#8 Ed25519 key path and does not natively call a managed signer.

### Fingerprint handling

InvarLock fingerprints the raw 32-byte Ed25519 public key with SHA-256 and
prefixes the lowercase hexadecimal digest with `sha256:`. Do not substitute a
SHA-256 digest of PEM text or SubjectPublicKeyInfo DER; those are different
bytes and produce a different value.

Bootstrap fingerprints through an authorized identity record or out-of-band
comparison. Reading the first observed fingerprint from an untrusted pack and
authorizing it automatically is trust on first use and does not resist an
initial substitution attack.

## Pin and materialize inputs

Before evaluation:

- materialize checkpoint, tokenizer, dataset source, policy, provider
  executable, and support files;
- for run-mode JSONL, review the input/expected-output field mapping, optional
  ID field, exact-prefix limit, source order, and SHA-256 digest before subject
  results are visible;
- use immutable upstream revisions and verify local content digests;
- place request-owned material under one access-controlled request root;
- configure optional provider support beneath caller-owned resource roots;
- address the OCI runtime image by SHA-256 digest; and
- prevent writes to fixed inputs during the transaction.

Do not accept a mutable tag, branch, `latest` reference, or moving dataset
revision as the identity for an acceptance decision.

Record both a portable locator and authenticated material digest where the
provider supports them. A locator helps a decision owner find material; only the
digest or typed immutable identity determines equality.

## Keep runtime configuration outside request authority

The request expresses comparison intent. Runtime image digest, device, and
optional provider support resources come from caller-owned configuration. Keep
network, remote code, and third-party plugin access disabled unless a separate
risk review explicitly authorizes them.

The in-process network guard is defense in depth, not a sandbox or host
firewall. Use container or infrastructure-level egress controls for an
adversarial workload.

Treat the container image as one layer of the trusted computing base. Patch and
harden the host, container engine, kernel, driver, firmware, and device access;
containers share host resources and do not by themselves supply execution
attestation or a complete security boundary.

## Authorize scorer extensions explicitly

When `comparison.scorer_extension` is selected, review the scorer
implementation and configuration before use. Pin its ID, version, descriptor
digest, and configuration digest in independently controlled policy bytes, and
pass the same explicit `ScorerExtensionRegistry` to evaluation and
verification. Installed code or a binding submitted inside evidence is not an
authorization source.

An acceptance scorer executes inside the evaluation and verifier trust
boundaries. Keep it deterministic and local, and reject implementations that
use a network, external model, human judgment, SQL or code execution,
model-based semantic similarity, or an LLM judge. Attach judge or review
results as authenticated observations when useful; observations remain outside
policy arithmetic. The extension declaration and replay checks do not provide
process isolation; enforce these restrictions with reviewed code and the
deployment sandbox.

### Use per-side OCI isolation

For run mode, pin the baseline and subject image digests independently. Common
image, device, and entrypoint options are convenience defaults; per-side
options select the exact worker configuration when the sides differ. Keep
schedule preparation, output validation, and evidence signing on the host.

The launcher runs CPU workers together when resources permit. It runs workers
with the same explicit CUDA index or a generic `cuda` selection sequentially
to avoid accidental contention. Distinct explicit indexes such as `cuda:0`
and `cuda:1` permit parallel worker execution. Device isolation remains a
property of the container engine, driver, and host configuration.

## Establish independent anchors

The verification operator should obtain these values independently of the
submitted pack:

- exact acceptance-policy bytes;
- expected baseline artifact-identity digest;
- expected subject artifact-identity digest;
- expected canonical schedule digest;
- expected baseline runtime digest;
- expected subject runtime digest;
- expected evidence-signer fingerprint;
- verifier key authorization; and
- verifier identity.

Store them in protected CI or release configuration. A closed
`invarlock/trust-inputs-v1` profile keeps the values reviewable as one unit;
protect the referenced verifier key separately and do not populate the profile
by parsing the evidence immediately before verification.

`invarlock evaluate` always completes its execution-free validation before it
allocates accelerators. Run it with `--preflight` when you want to stop at that
boundary and inspect the exact request, signing key, and runtime-image options.
The v2 preflight output can confirm a policy's minimum record count. It reports
the interval-width requirement as `pending_execution`; do not treat that state
as proof that precision or the final verdict will pass.
The check is local
and non-mutating: it does not pull images, start containers, load model
weights, run inference, publish evidence, or sign a result. Treat it as
configuration qualification, not as evidence that execution or policy will
pass. For a scorer-bound request, the same preflight must load the explicitly
authorized scorer and validate its exact descriptor, configuration schema, and
task compatibility before compute is allocated.

For release qualification, first use `runtime-qualification-canary` or the
provider's `qualify-canary` target to run one representative, strictly verified
transaction through the exact image digest. Retain that canary's evidence,
signed receipt, and verifier-owned trust profile. The maintained readiness and
evidence targets reverify all three and reject a canary bound to another image.
Changing the image digest requires a new signed canary.

Then use the maintained readiness and evidence targets with the actual Git
source archive, its digest and commit, a runtime image built with the matching
source labels, and fresh target evidence, receipt, report, and private-summary
destinations. The driver runs from the authenticated source archive through a
private snapshot and empty working directory. It verifies each caller-digested
candidate wheel against that archive, then runs the candidate core and maintained
add-ins through an isolated interpreter bootstrap that does not process installed
`.pth` files or `sitecustomize`. The private summary records the candidate-wheel
and interpreter digests. It compares preflight's complete
normalized-request digest with the strictly verified pack, independently
validates the signed receipt, and requires the renderer to report the same pack
identity. Fixed request material can remain read-only because the driver places
its byte snapshot in private temporary storage while retaining the original
request root for relative inputs. All output parents must already exist, be
real directories without symlink ancestors, and remain outside the evidence
destination. A readiness result authorizes the bounded execution attempt; it
is not itself an acceptance result.

The signed canary prevents broad image-level fan-out after a representative
transaction fails. It is not evidence that a different model will load, fit
the device, satisfy backend compatibility, or complete its own evaluation.
Readiness remains execution-free and model-specific failures remain possible.

The named commit must exist in the local qualification repository. The driver
recreates that commit's archive with system Git and compares the complete
execution-source inventory and bytes with the supplied bundle. A copied tar
whose PAX comment merely names another commit is rejected.

Run the complete verification transaction:

```bash
invarlock verify evidence/ \
  --trust-profile trust/trust-inputs.json \
  --receipt verification.receipt.json \
  --json
```

Treat a nonzero exit as nonacceptance. Retain a signed failure receipt when the
transaction produced one; it records which anchors and evidence were verified.

Before automation relies on a receipt, verify it against the expected verifier
identity and fingerprint from the same independent configuration class. The
receipt's embedded public key is not an authorization record.

## Precommit schedule and run selection

Pairing prevents the two sides from seeing different records. It does not
prevent cherry-picking. Before subject results are visible, approve:

- baseline and subject identities;
- schedule and expected outputs;
- metric and acceptance threshold;
- runtime and provider configuration;
- seed or decoding choices;
- run-selection and retry rules; and
- the treatment of failures and superseded attempts.

Use independent or held-out schedules when repeated tuning would otherwise
leak evaluation information.

When evidence informs a high-impact decision, separate the person or service
that selects the subject and retries from the person or service that controls
the schedule and accepts the receipt.

## Preserve canonical artifacts

Treat the published evidence directory as immutable. Never edit, normalize,
compress in place, or insert a receipt after publication. Store related output
beside it:

```text
review/
├── evidence/                    # signed immutable evidence bundle
├── verification.receipt.json   # verifier-signed acceptance statement
├── evidence.html               # optional, regenerable presentation
└── external-attestation/       # optional independently trusted material
```

Verify again after transport and before relying on the evidence. The HTML view
and copied summaries are not substitutes for the bundle and signed receipt.

Do not include secrets, access tokens, private host names, local filesystem
paths, environment dumps, or private keys in public evidence or reports.
Inspect imported provider output before publication; a cryptographically valid
pack can still disclose sensitive content intentionally placed in a record.

### Data handling and retention

Provider inputs and outputs can contain prompts, labels, model responses, or
other regulated data. Encrypt request roots, evidence, receipts, and backups at
rest when their classification requires it; use encrypted transport; restrict
read access independently from signing authority; and define retention and
deletion periods before evaluation. Public-evidence screening detects selected
disclosure patterns, not every secret, identifier, copyrighted passage, or
personal-data category. A human or domain-specific disclosure review remains
required before publication.

Keep the immutable evidence bundle for as long as the decision must remain
auditable. Retain receipts and the historical evidence-signer and verifier authorization
records for the same interval. Delete temporary materializations, caches, and
private keys according to the deployment policy without modifying the signed
bundle that remains under retention.

## Harden the execution environment

- run untrusted model and provider material in an isolated, patched runtime;
- deny network egress at the infrastructure boundary;
- disable remote model code and plugins that are not explicitly authorized;
- prefer formats and loaders that avoid executable deserialization;
- run with least filesystem, device, process, and secret privileges;
- pin and scan dependencies and images;
- keep signing keys host-side and unavailable to model/provider workers;
- collect host or workload attestation when the decision requires proof of
  execution; and
- independently rerun high-impact or boundary decisions.

InvarLock's runtime manifest and provider receipt improve traceability. They do
not secure the host or attest that the declared image executed.

## Respond to key or runtime compromise

When compromise is suspected:

1. stop evaluation or verification signing with the affected key;
2. preserve audit records and identify the exposure interval;
3. revoke the fingerprint in the authoritative identity source;
4. enumerate affected packs and receipts by recorded fingerprint;
5. invalidate downstream acceptance until source evidence is reverified;
6. rotate keys and independently distribute new fingerprints; and
7. regenerate evidence or receipts from trusted environments as required.

Changing an expected fingerprint locally without a recorded authorization
update hides the incident. Re-signing an old receipt under a new verifier key
can renew verifier accountability only after the underlying pack and anchors
are rechecked.

For a compromised runtime image, revoke the digest in the runtime-anchor
source, determine which packs bind it, and create new evidence under an authorized
replacement image. Do not treat an unchanged Dockerfile as evidence that the
new image is byte-identical or that the old result remains trustworthy.

## Review before acceptance

Use the one-screen [acceptance checklist](../assurance/acceptance-checklist.md).
Pay particular attention to:

- whether the finite schedule matches the intended decision scope;
- whether the policy threshold was chosen before results;
- whether a pass is close enough to the threshold to merit replication;
- whether baseline trust is justified independently; and
- whether execution attestation is required in addition to digest matching.

Read the [Threat model](threat-model.md) before using evidence for a
high-impact decision.

## References

- National Institute of Standards and Technology,
  [SP 800-57 Part 1 Revision 5: Recommendation for Key Management](https://doi.org/10.6028/NIST.SP.800-57pt1r5),
  2020.
- National Institute of Standards and Technology,
  [Digital Signature Standard](https://doi.org/10.6028/NIST.FIPS.186-5),
  2023.
- [RFC 8032: Edwards-Curve Digital Signature Algorithm](https://www.rfc-editor.org/rfc/rfc8032),
  2017.
- National Institute of Standards and Technology,
  [SP 800-190: Application Container Security Guide](https://doi.org/10.6028/NIST.SP.800-190),
  2017.
