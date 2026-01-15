# Guards

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Safety checks that validate edits against baseline-derived contracts. |
| **Audience** | Users tuning guard behavior and reviewing certificate evidence. |
| **Supported guards** | `invariants`, `spectral`, `rmt`, `variance` (plus optional plugin guards). |
| **Requires** | `invarlock[guards]` for torch/numpy guard math. |
| **Network** | Offline by default; guard logic itself is local. |
| **Inputs** | Model, adapter, calibration data, tier policy (`--tier`/`auto_config`). |
| **Outputs / Artifacts** | `report.guards` entries, certificate `resolved_policy`, `validation.*` flags. |
| **Source of truth** | `src/invarlock/guards/*.py`, `src/invarlock/guards/policies.py`, `src/invarlock/_data/runtime/tiers.yaml`. |

## Quick Start

```yaml
guards:
  order: ["invariants", "spectral", "rmt", "variance", "invariants"]
  spectral:
    sigma_quantile: 0.95
    deadband: 0.10
    scope: ffn
  rmt:
    epsilon_by_family: { ffn: 0.01, attn: 0.01, embed: 0.01, other: 0.01 }
  variance:
    min_gain: 0.0
    scope: ffn
```

> Most thresholds come from the tier defaults (see `tiers.yaml`). Use overrides
> sparingly and keep evidence in the certificate.

## Guard Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        GUARD PIPELINE FLOW                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│   │invariants│───▶│ spectral │───▶│   rmt    │───▶│ variance │         │
│   │(pre-edit)│    │ (weight) │    │(activatn)│    │  (A/B)   │         │
│   └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘         │
│        │               │               │               │                │
│        ▼               ▼               ▼               ▼                │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│   │ prepare  │    │ prepare  │    │ prepare  │    │ prepare  │         │
│   │(baseline)│    │(baseline)│    │(calibrtn)│    │(calibrtn)│         │
│   └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘         │
│        │               │               │               │                │
│        ▼               ▼               ▼               ▼                │
│   ┌──────────────────────────────────────────────────────────┐         │
│   │                    EDIT APPLIED                          │         │
│   └──────────────────────────────────────────────────────────┘         │
│        │               │               │               │                │
│        ▼               ▼               ▼               ▼                │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│   │ validate │    │ validate │    │ validate │    │ validate │         │
│   │(post-edt)│    │(κ-check) │    │(ε-band)  │    │(gain>0?) │         │
│   └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘         │
│        │               │               │               │                │
│        ▼               ▼               ▼               ▼                │
│   ┌──────────────────────────────────────────────────────────┐         │
│   │                GUARD RESULTS → CERTIFICATE               │         │
│   │     (passed/warned/failed + metrics + measurement_hash)  │         │
│   └──────────────────────────────────────────────────────────┘         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Concepts

- **Guard lifecycle**: the core runner calls `prepare(...)` (if implemented)
  and always calls `validate(...)`. Optional hooks (`before_edit`, `after_edit`,
  `finalize`) are only used when you manage guards manually (e.g., with
  `GuardChain`).
- **Tier policies**: `--tier balanced|conservative|aggressive` resolves a full
  policy bundle from `runtime/tiers.yaml`; overrides in config are merged on top.
- **Measurement contracts**: Spectral and RMT guards record estimator + sampling
  contracts in certificates and are enforced by `invarlock verify` in CI/Release.

### Guard hooks

| Hook | When called | Evidence |
| --- | --- | --- |
| `prepare` | Before edit (GuardWithPrepare only). | `report.guards[].actions`, `report.guards[].metrics`. |
| `validate` | After edit. | `report.guards[].passed`, `report.guards[].action`, `warnings`, `errors`. |

### Verify gate requirements

| Gate | Required fields | Applies |
| --- | --- | --- |
| Measurement contracts | `spectral.measurement_contract_hash`, `rmt.measurement_contract_hash`, `resolved_policy.*`. | CI/Release. |
| Guard overhead | `guard_overhead.*`. | Release only. |
| Validation allow‑list | `validation.*` booleans. | Schema validation. |

## Reference

### Guard summary

| Guard | Purpose | Key knobs (override) | Evidence (certificate/report) |
| --- | --- | --- | --- |
| `invariants` | Structural integrity + non-finite checks. | `strict_mode`, `on_fail`, `profile_checks`. | `validation.invariants_pass`, `invariants.*`. |
| `spectral` | Baseline-relative spectral norm stability. | `sigma_quantile`, `family_caps`, `deadband`, `scope`, `correction_enabled`, `estimator`, `degeneracy`, `multiple_testing`. | `validation.spectral_stable`, `spectral.*`, `resolved_policy.spectral`. |
| `rmt` | Activation edge-risk stability (ε-band). | `epsilon_default`, `epsilon_by_family`, `deadband`, `margin`, `correct`, `estimator`, `activation.sampling`. | `validation.rmt_stable`, `rmt.*`, `resolved_policy.rmt`. |
| `variance` | Variance equalization with A/B gate. | `min_gain`, `min_effect_lognll`, `max_calib`, `scope`, `clamp`, `deadband`, `predictive_gate`, `predictive_one_sided`, `calibration`, `tap`. | `variance.*`, `resolved_policy.variance`. |

### Guard evidence matrix

| Guard config | Report evidence | Certificate evidence | Verify gate |
| --- | --- | --- | --- |
| `guards.invariants.*` | `report.guards[name=invariants]` | `certificate.invariants`, `validation.invariants_pass` | Schema only. |
| `guards.spectral.*` | `report.guards[name=spectral]` | `certificate.spectral`, `resolved_policy.spectral`, `validation.spectral_stable` | Measurement contracts (CI/Release). |
| `guards.rmt.*` | `report.guards[name=rmt]` | `certificate.rmt`, `resolved_policy.rmt`, `validation.rmt_stable` | Measurement contracts (CI/Release). |
| `guards.variance.*` | `report.guards[name=variance]` | `certificate.variance`, `resolved_policy.variance` | Schema only. |
| `--profile release` | `report.guard_overhead` | `certificate.guard_overhead` | Required unless skipped. |

### Invariants Guard

```yaml
guards:
  invariants:
    strict_mode: false
    on_fail: warn   # warn | rollback | abort
```

### Spectral Guard (measurement contract)

```yaml
guards:
  spectral:
    sigma_quantile: 0.95
    deadband: 0.10
    scope: all
    family_caps: { ffn: 3.85, attn: 3.02, embed: 1.05, other: 0.0 }
    estimator: { iters: 4, init: ones }
    degeneracy:
      enabled: true
      stable_rank: { warn_ratio: 0.5, fatal_ratio: 0.25 }
      norm_collapse: { warn_ratio: 0.25, fatal_ratio: 0.10 }
```

### RMT Guard (activation edge-risk)

```yaml
guards:
  rmt:
    epsilon_by_family: { ffn: 0.01, attn: 0.01, embed: 0.01, other: 0.01 }
    epsilon_default: 0.01
    estimator: { iters: 3, init: ones }
    activation:
      sampling:
        windows: { count: 8, indices_policy: evenly_spaced }
```

### Variance Guard (A/B gate)

```yaml
guards:
  variance:
    scope: ffn
    min_gain: 0.0
    min_effect_lognll: 0.0
    max_calib: 200
    clamp: [0.85, 1.12]
    deadband: 0.02
    predictive_gate: true
    predictive_one_sided: false
    calibration:
      windows: 200
      min_coverage: 50
      seed: 123
```

### Guard order

`guards.order` defines the execution chain and is required in YAML presets. The
packaged presets include it by default; remove a guard from the list to skip it.

## Troubleshooting

- **Guard prepare failed**: set `INVARLOCK_GUARD_PREPARE_STRICT=0` for local
  debugging, or adjust tier policies for the guard that failed.
- **Spectral instability**: lower `sigma_quantile`, narrow `scope`, or increase
  deadband to reduce noise.
- **RMT ε-band violations**: tighten calibration (more windows) or adjust
  `epsilon_by_family` only if you are updating tier policy evidence.
- **Variance guard never enables**: A/B gate may fail; inspect
  `variance.metrics.predictive_gate` and `variance.metrics.ab_gain` in the report.

## Observability

- `report.guards` contains guard results by name.
- Certificates include `resolved_policy.{spectral,rmt,variance}` and evidence
  blocks (`spectral.*`, `rmt.*`, `variance.*`).
- Validation flags are recorded under `validation.*` (`invariants_pass`,
  `spectral_stable`, `rmt_stable`).

## Related Documentation

- [Tier Policy Catalog](tier-policy-catalog.md)
- [GPU/MPS-First Guards (Decision Memo)](../assurance/13-gpu-mps-first-guards.md)
- [Configuration Schema](config-schema.md)
- [Environment Variables](env-vars.md)
- [Guard Contracts & Primer](../assurance/04-guard-contracts.md)
