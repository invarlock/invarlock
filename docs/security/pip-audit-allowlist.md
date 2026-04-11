# pip-audit Allowlist

The CI supply-chain jobs run `pip-audit` through
`scripts/security/run_pip_audit.py`, which owns allowlist parsing and
enforcement. The JSON source of truth is
`scripts/security/pip_audit_allowlist.json`.

The current exception is:

| Vulnerability ID      | Package | Expires      | Tracking Issue | Reason |
| --------------------- | ------- | ------------ | -------------- | ------ |
| `GHSA-4xh5-x5gv-qwph` | `pip`   | `2026-05-11` | [pip#13607](https://github.com/pypa/pip/issues/13607) | Temporary exception while the installer dependency chain is being refreshed and the upstream pip issue remains under review. |

All other findings must be remediated prior to release. Update this table and
the JSON allowlist entry whenever the allowlist changes.

## CVE Response Process

When a new CVE is discovered affecting InvarLock dependencies:

### 1. Discovery

New CVEs are detected via:

- `pip-audit` in CI (fails the build)
- GitHub Dependabot alerts
- Manual security reviews

### 2. Triage

Maintainer assesses exploitability:

- **Direct impact:** Vulnerability in code paths executed by InvarLock
- **Indirect impact:** Vulnerability in optional dependency or unused code path
- **No impact:** Dependency included transitively but never loaded

### 3. Decision Matrix

| Exploitability | Severity | Action |
| --- | --- | --- |
| Direct | Critical/High | Patch immediately, hotfix release |
| Direct | Medium/Low | Patch in next scheduled release |
| Indirect | Any | Add to allowlist with expiry, patch within 30 days |
| None | Any | Add to allowlist, track upstream |

### 4. Allowlist Entry Format

When adding to the allowlist:

```markdown
| `GHSA-xxxx-xxxx-xxxx` | `package` | YYYY-MM-DD | [owner/repo#123](https://github.com/owner/repo/issues/123) | [Reason] |
```

Include:

- Clear reason why it's acceptable to ignore
- Expiry date within 30 days
- Link to a GitHub tracking issue

### 5. Periodic Review

- Allowlist entries reviewed monthly
- Entries removed when upstream fix is available and upgraded
- Entries beyond 30 days are rejected by the allowlist loader

### 6. Documentation

For each allowlisted CVE:

1. Add entry to the JSON allowlist with a reason, expiry, and tracking issue.
2. Update the table above so the docs stay aligned with the JSON source.
3. Use a GitHub issue link that tracks the upstream fix or the repo follow-up.

## See Also

- [Security Best Practices](best-practices.md)
- [Threat Model](threat-model.md)
