# Evaluation Capacity

!!! info "Reference"
    **Surface:** Bounded captured-evaluation workload and replay limits.
    **Stability:** Public core operational contract.
    **Use this page when:** Setting recipient-side replay budgets.

Captured evaluation enforces bounded input, record, artifact, and comparison
sizes before doing statistical work. The verifier also charges the complete
planned replay budget before evaluating a metric. These limits protect the
recipient process and are independent of native runtime resource limits.

| Bound | Captured limit |
| --- | --- |
| Records per run / planned case set | 50,000 |
| Metrics / metadata slices | 16 / 16, plus overall scope |
| IDs and short labels | 128 characters, no control characters |
| Metadata values / error messages | 4,096 characters |
| Record score or metadata maps | 100 properties |
| Authored or normalized request | 1 MiB |
| Canonical run, policy, or comparison payload | 128 MiB each |
| Manifest, checksums, or signature control file | 64 KiB each |
| Complete pack inventory | 384 MiB, including control bytes |
| External signed verification receipt | 1 MiB |
| Default complete comparison work | 102,400,000 planned bootstrap draws |

Use `--max-bootstrap-draws` on `evaluate` and independently on `verify` to select
each caller's integer allowance. The complete metric/slice schedule is charged
before the first scorer call, including overlapping slices. Preflight reports
required work without computing scores. Exceeding a budget exits `2` without a
new pack or recipient receipt; it is incomplete verification, not a policy
decision or integrity finding. Retry with a reviewed adequate allowance. Raising
the evaluation allowance does not raise the verifier allowance or change signed
policy. SDK `None` explicitly removes only this local work limit.

Per-role and aggregate byte checks apply to physical bytes as well as bounded
canonical serialization. Oversized missing-ID arrays are refused before costly
comparison work, and diagnostics are bounded. Do not drop cases, truncate IDs,
or change policy to bypass capacity errors. Native runtime schedule, evidence,
and resource ceilings remain separate and unchanged.
