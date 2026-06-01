# Policy Provenance & Digest

> **Plain language:** The report embeds the exact policy evaluated and a
> short digest so auditors can recompute and verify there was no silent drift.

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

1) Extract `resolved_policy` and the ordered `policy_provenance.overrides` list.
2) Recompute the digest locally (see pseudocode).
3) Confirm it matches `policy_provenance.policy_digest` and `auto.policy_digest`.

If the digest does not match, treat the evidence as drifted or altered and
rerun evaluation. For tamper-evident distribution, use signed evidence packs.

## Notes

- The digest guards against silent changes to thresholds/caps between runs.
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
