# Dependency-audit exceptions

Dependency auditing is a release and maintenance control. An exception is a
time-bounded, maintainer-approved statement that one known advisory does not currently
block the repository; it is not a declaration that the dependency is safe.

!!! warning "Security guidance"

    **In plain language:** A vulnerability exception buys limited remediation time; it must identify an owner, public tracking issue, exact advisory, rationale, and near-term expiry.
    **Objective:** Keep `pip-audit` exceptions narrow, accountable, and self-expiring while preserving a fail-closed dependency gate.
    **Assets or boundary:** Core and first-party add-in Python dependency sets audited by `make security`.
    **Use this page when:** Assessing a new advisory, authorizing a temporary exception, reviewing an existing entry, or removing a remediated exception.

## Default response

When `make security` reports an advisory:

1. confirm the affected distribution and installed version from the isolated
   audit environment;
2. read the upstream advisory and fixed-version information;
3. determine whether the vulnerable code is reachable in the core, HF extra,
   GGUF, TensorRT-LLM, diagnostics, build, or documentation surface;
4. prefer upgrading, removing, or constraining the dependency;
5. run the relevant package, runtime, and repository checks; and
6. remove obsolete allowlist entries in the same change.

Do not add an exception merely because exploitation was not reproduced. Lack
of a local proof is not evidence that a vulnerability is unreachable.

## Exception decision

An exception is appropriate only when all of these are true:

- no non-breaking fixed dependency set is currently available;
- the affected path and exposure are understood;
- compensating controls materially reduce the reachable risk;
- a named maintainer owns remediation;
- a public issue tracks the work and contains no sensitive exploit detail;
- the exception expires within 30 days; and
- release owners explicitly accept the residual risk.

Do not except an advisory with known active exploitation against the deployed
surface, a credential or signature compromise, or an unauthenticated code path
that the release intends to expose. Stop the release or remove the affected
surface instead.

## Allowlist entry

The source of truth is `scripts/security/pip_audit_allowlist.json`. The checker
validates exact keys and rejects missing ownership, issue links, malformed
dates, and expirations more than 30 days ahead. Follow the shape already
enforced by `scripts/security/run_pip_audit.py`; do not add undocumented fields
or wildcard advisory IDs.

Each entry must answer:

| Field | Review question |
| --- | --- |
| Advisory/package identity | Which exact finding is being excepted? |
| Rationale | Why is the vulnerable path not currently release-blocking? |
| Compensating control | What concrete boundary reduces exposure? |
| Owner | Who is accountable for removal or renewal? |
| Issue URL | Where is remediation tracked publicly? |
| Expiration | What date forces re-review, no more than 30 days away? |

Never place unpublished exploit details, credentials, private hosts, or private
artifact locations in the allowlist or issue.

## Review and removal

- Review every entry when dependency locks change and at least once before a
  release candidate is accepted.
- Treat an expired entry as a failed security gate, not an administrative
  warning.
- Renew only after repeating the decision analysis; update the rationale and
  issue rather than changing the date alone.
- Remove the entry as soon as the fixed dependency set passes the same tests
  and audit surface.
- Preserve Git history as the audit record; do not keep obsolete entries in the
  live allowlist as an archive.

Run the authoritative checks after any change:

```bash
make cve-audit
make security
```

## Incident boundary

An allowlist is not a vulnerability-response channel. Report suspected
exploitable defects through the private process in the repository security
policy. If an excepted advisory becomes actively exploitable, revoke the
exception, assess affected releases and evaluation environments, and rotate exposed
secrets or signing keys when the incident analysis requires it.

## Related documentation

- [Security practices](best-practices.md) defines the broader dependency and
  runtime hardening posture.
- [Release verification](../reference/release-verification.md) describes the
  coordinated distribution gate.
- [Threat model](threat-model.md) separates dependency compromise from evidence
  integrity and acceptance claims.
