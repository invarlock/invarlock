# Policy Provenance & Digest

> **Plain language:** The report embeds the policy it claims was evaluated and a
> short digest so auditors can check internal consistency. Recomputing a digest
> from the same report does not prove that the claimed policy was authorized;
> that requires an expected policy snapshot or digest obtained independently.

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Define how resolved policies and override order are captured, digested, and audited in reports. |
| **Audience** | Report verifier maintainers, release approvers, and operators checking policy drift. |
| **Contract scope** | `resolved_policy`, `policy_provenance`, policy digests, override ordering, and auditor recomputation. |
| **Source of truth** | `src/invarlock/reporting/policy_utils.py`, report generation code, and policy digest tests. |

## Resolved Policy → Digest

At runtime, the tier base (Balanced/Conservative/Aggressive) is resolved, guard‑level overrides are applied, and the result is materialized as `resolved_policy` in the report.
Additionally, a compact `policy_digest` object captures threshold floors and hysteresis knobs for stable auditing.

- Canonicalization: JSON serialize with sorted keys (standard JSON booleans and numbers; no locale‑specific formatting).
- Digest: `sha256(canonical)[0:16]` → `policy_provenance.policy_digest`.
- The canonical payload includes `resolved_policy` plus the ordered `overrides` list, so
  reordering overrides changes the digest.

Pseudocode to recompute the digest locally:

```python
import json, hashlib
canonical = json.dumps(
    {"resolved_policy": resolved_policy, "overrides": overrides},
    sort_keys=True,
    default=str,
)
digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
```

## Provenance (report fields)

- `resolved_policy`: per‑guard snapshot used during evaluation
- `policy_provenance`:
  - `tier` — policy tier name (e.g., `balanced`)
  - `overrides` — ordered list of override paths applied
  - `policy_digest` — short digest of `resolved_policy` plus ordered overrides
  - `resolved_at` — timestamp synchronized with report generation
- Convenience mirror: `auto.policy_digest`
- Thresholds digest and knobs: top‑level `policy_digest` with `{policy_version,tier_policy_name,thresholds_hash,hysteresis,min_effective,changed}`

## Auditor Checklist

1) Obtain the approved policy snapshot or digest from a separately controlled
   release record, signed policy pack, or acceptance configuration.
2) Extract `resolved_policy` and the ordered `policy_provenance.overrides` list.
3) Recompute the digest locally (see pseudocode).
4) Confirm the report mirrors agree **and** the resolved policy matches the
   independently obtained expectation.

For strict verification, this comparison is machine-enforced through
`--policy-pack`: the pack must be structurally valid, authorize a Balanced or
Conservative tier, and contain a `resolved_policy` exactly equal to the report's
resolved policy. The report-local truncated digest is not used as a substitute
for that exact policy comparison.

If the digest does not match, treat the evidence as drifted or altered and
rerun evaluation. A signed evidence pack authenticates the pack to a trusted
signer, but does not prove that the signer chose the approved policy unless the
verifier caller compares it with that external expectation.

## Notes

- The digest is a compact internal change detector, not a MAC, signature,
  authorization decision, or independent provenance anchor. Its 16-hex-character
  form is deliberately truncated and must not be treated as collision-resistant
  security for adversarial inputs.
- A malicious report author can alter both `resolved_policy` and its digest. External
  pinning or a separately signed approval record is required to detect that case.
- Keep tier tables and schema pages in sync when policy values change.

### Example (report fragment)

```json
{
  "auto": {"policy_digest": "4676d5d572e3b69c"},
  "policy_digest": {
    "policy_version": "policy-v1",
    "tier_policy_name": "balanced",
    "thresholds_hash": "9c0e3d0c5acb7e11",
    "hysteresis": {"ppl": 0.002, "accuracy_delta_pp": 0.1},
    "min_effective": {"ppl_ratio": 1.102},
    "changed": false
  },
  "resolved_policy": {
    "spectral": {
      "family_caps": {
        "ffn": {"kappa": 3.849},
        "attn": {"kappa": 3.018},
        "embed": {"kappa": 1.05},
        "other": {"kappa": 0.0}
      }
    }
  },
  "policy_provenance": {
    "tier": "balanced",
    "overrides": ["configs/overrides/spectral_balanced_local.example.yaml"],
    "policy_digest": "4676d5d572e3b69c",
    "resolved_at": "2025-10-13T01:22:45Z"
  }
}
```
