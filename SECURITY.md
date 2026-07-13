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

1. **Keep dependencies updated**: Run `pip install --upgrade invarlock` regularly
2. **Review evaluation reports**: Run `invarlock verify` with an independently
   pinned expected runtime-image digest, then inspect the underlying evidence.
3. **Isolate sensitive workloads**: Use virtual environments or containers
4. **Network isolation**: Set `INVARLOCK_ALLOW_NETWORK=0` (default) except when needed
5. **Audit configurations**: Review config files before running evaluation workflows

## Security Features

InvarLock includes several security features:

- **Network disabled by default**: External network access requires explicit opt-in
- **Supply chain checks**: SBOM generation and dependency auditing in CI
- **Evidence-pack signatures**: Ed25519 manifest signatures can authenticate a
  pack when the verifier receives a trusted signer fingerprint or trust store.
- **Report/manifest binding**: SHA-256 binding detects mismatches between a
  report and its runtime manifest. An independently supplied expected image
  digest additionally checks the manifest's image claim.

These mechanisms do not attest actual container execution and cannot make
evidence from a compromised evaluation environment trustworthy: the environment can fabricate a
consistent report and manifest that name an expected digest. Use isolated
evaluation infrastructure, protect signing keys, and obtain trust anchors
through a separate release/deployment channel.

## Acknowledgments

We thank the security researchers who have responsibly disclosed vulnerabilities:

Public acknowledgments will be listed here when applicable.

---

This policy is inspired by [GitHub's security policy guidelines](https://docs.github.com/en/code-security/getting-started/adding-a-security-policy-to-your-repository) and follows industry best practices.
