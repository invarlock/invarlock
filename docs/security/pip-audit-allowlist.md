# pip-audit Allowlist

The CI supply-chain job runs `pip-audit` with the following exception:

| Vulnerability ID      | Package | Reason                                                                                               |
| --------------------- | ------- | ---------------------------------------------------------------------------------------------------- |
| `GHSA-4xh5-x5gv-qwph` | `pip`   | Upstream fix pending as of 2025-01; monitor for patched release and drop this ignore once available. |

All other findings must be remediated prior to release. Update this table and
the `--ignore-vuln` flag in `.github/workflows/ci.yml` whenever the allowlist
changes.
