# Security Policy

## Supported Versions

We actively support security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.x.x   | :white_check_mark: |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

### How to Report

If you discover a security vulnerability in InvarLock, please report it by:

1. **Email**: Send details to [security@invarlock.dev](mailto:security@invarlock.dev) (or create a private security advisory on GitHub)
2. **GitHub Security Advisory**: Use GitHub's private [security advisory feature](https://github.com/invarlock/invarlock/security/advisories/new)

### What to Include

Please include as much of the following information as possible:

- **Type of issue** (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- **Full paths of source file(s)** related to the issue
- **Location of the affected source code** (tag/branch/commit or direct URL)
- **Step-by-step instructions** to reproduce the issue
- **Proof-of-concept or exploit code** (if possible)
- **Impact of the issue**, including how an attacker might exploit it

### Response Timeline

- **Initial Response**: We aim to acknowledge reports within a few business days.
- **Status Updates**: We aim to share updates as assessment and remediation work
  progresses, with timing based on severity, exploitability, and maintainer
  availability.
- **Fix Timeline**: Remediation timing depends on the scope and risk of the
  issue. Critical or actively exploitable vulnerabilities are prioritized, while
  lower-risk issues may be handled in a scheduled release.

### What to Expect

1. **Acknowledgment**: We aim to confirm receipt of your vulnerability report.
2. **Assessment**: We triage reports and assess severity based on available
   information.
3. **Fix Development**: When a fix is needed, we work on remediation and may
   coordinate with the reporter when follow-up details are useful.
4. **Disclosure**: Public disclosure, when appropriate, is coordinated around
   user risk, fix availability, and release timing.

### Safe Harbor

We consider security research conducted in good faith and in accordance with this policy to be:

- Authorized under the Computer Fraud and Abuse Act (CFAA)
- Exempt from DMCA restrictions on circumvention
- Lawful, helpful, and welcome

We will not pursue legal action against researchers who:

- Act in good faith to avoid privacy violations and disruptions to others
- Only interact with accounts they own or with explicit permission
- Report vulnerabilities through this process before any public disclosure

## Security Best Practices for Users

When using InvarLock:

1. **Keep dependencies updated**: install a current InvarLock release and audit
   the complete environment, including optional provider runtimes.
2. **Verify with external trust anchors**: supply the policy, baseline and
   subject artifact-identity digests, canonical schedule digest, both expected
   runtime digests, and expected evidence-signer fingerprint independently of
   the evidence bundle. Write and retain a separately signed verifier receipt.
3. **Inspect authenticated evidence**: use `invarlock report` only after the
   bundle has passed independent verification.
4. **Isolate sensitive workloads**: use virtual environments or containers and
   protect evidence-signer and verifier signing keys separately. In OCI run
   mode, the host prepares the schedule and launches a separately pinned worker
   for each side. Model workers receive only read-only job, artifact, and support
   mounts plus an isolated writable output directory; the evidence-signing key
   remains in the host process and is never mounted into a model worker. A local
   key file is not isolation from compromise of that host process.
5. **Keep evaluation offline by default**: leave `INVARLOCK_ALLOW_NETWORK=0`
   unless an explicitly authorized provider operation requires network access.
6. **Audit requests**: review every request, referenced input, output
   destination, provider, and policy before evaluation.

## Security Features

InvarLock includes several security features:

- **Python-process network guard by default**: Python socket creation requires
  explicit opt-in; strict evidence additionally requires an independently
  enforced network-disabled container boundary.
- **Supply chain checks**: SBOM generation and dependency auditing in CI
- **Evidence-pack signatures**: Ed25519 manifest signatures authenticate the
  canonical pack when the verifier receives the expected evidence-signer
  fingerprint independently.
- **Closed evidence binding**: the signed manifest binds the normalized request,
  paired schedule and records, comparison report, and runtime-side provider
  evidence carried by the bundle. The verifier also requires an external
  policy, baseline and subject artifact-identity digests, canonical schedule
  digest, and both expected runtime digests.
- **Independent verifier receipts**: verification can write a separately signed
  receipt that binds the verifier identity, decision, policy, trust inputs, and
  evidence digest. The receipt is not stored inside the signed evidence pack.

These mechanisms authenticate claims and detect tampering; they do not attest
actual container execution. A compromised evaluation environment can fabricate
internally consistent evidence that names an expected digest. Use isolated
evaluation infrastructure, protect signing keys, and obtain policies, runtime
digests, artifact identities, schedule digests, signer fingerprints, and
verifier trust anchors through separate release or deployment channels.

See the [security practices](docs/security/best-practices.md) for the current
file-backed signer limitation, key custody, data retention, runtime isolation,
and incident response, and the [runtime-security API](docs/reference/runtime-security.md)
for the exact process-local controls and their limits.

## Acknowledgments

We thank the security researchers who have responsibly disclosed vulnerabilities:

Public acknowledgments will be listed here when applicable.

---

This policy is inspired by [GitHub's security policy guidelines](https://docs.github.com/en/code-security/getting-started/adding-a-security-policy-to-your-repository) and follows industry best practices.
