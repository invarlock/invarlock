# pip-audit Allowlist

The CI supply-chain job runs `pip-audit` with the following exception:

| Vulnerability ID      | Package | Reason                                                                                               |
| --------------------- | ------- | ---------------------------------------------------------------------------------------------------- |
| `GHSA-4xh5-x5gv-qwph` | `pip`   | Upstream fix pending as of 2025-01; monitor for patched release and drop this ignore once available. |

All other findings must be remediated prior to release. Update this table and
the `--ignore-vuln` flag in `.github/workflows/ci.yml` whenever the allowlist
changes.

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
| `GHSA-xxxx-xxxx-xxxx` | `package` | [Reason]; expires YYYY-MM-DD or when [condition]. |
```

Include:

- Clear reason why it's acceptable to ignore
- Expiry date or removal condition
- Link to upstream tracking issue if available

### 5. Periodic Review

- Allowlist entries reviewed monthly
- Entries removed when upstream fix is available and upgraded
- Stale entries (> 90 days) escalated for re-triage

### 6. Documentation

For each allowlisted CVE:

1. Add entry to table above with reason
2. Update `.github/workflows/ci.yml` with `--ignore-vuln`
3. Create tracking issue linking to upstream fix

## See Also

- [Security Best Practices](best-practices.md)
- [Threat Model](threat-model.md)
