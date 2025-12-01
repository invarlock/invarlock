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

- **Initial Response**: Within 72 hours of receiving your report
- **Status Update**: Within 7 days with an assessment of the vulnerability
- **Fix Timeline**: Dependent on severity:
  - Critical: Within 7 days
  - High: Within 30 days
  - Medium: Within 90 days
  - Low: Next scheduled release

### What to Expect

1. **Acknowledgment**: We'll confirm receipt of your vulnerability report
2. **Assessment**: We'll evaluate the issue and determine its severity
3. **Fix Development**: We'll work on a fix and coordinate with you
4. **Disclosure**: We'll publicly disclose the vulnerability after a fix is available

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
2. **Review certificates**: Always verify certificate integrity before trusting results
3. **Isolate sensitive workloads**: Use virtual environments or containers
4. **Network isolation**: Set `INVARLOCK_ALLOW_NETWORK=0` (default) except when needed
5. **Audit configurations**: Review config files before running certification workflows

## Security Features

InvarLock includes several security features:

- **Network disabled by default**: External network access requires explicit opt-in
- **Supply chain verification**: SBOM generation and dependency auditing in CI
- **Certificate integrity**: Cryptographic verification of certification results
- **Minimal permissions**: Least-privilege design throughout the codebase

## Acknowledgments

We thank the security researchers who have responsibly disclosed vulnerabilities:

*No vulnerabilities have been reported yet.*

---

This policy is inspired by [GitHub's security policy guidelines](https://docs.github.com/en/code-security/getting-started/adding-a-security-policy-to-your-repository) and follows industry best practices.
