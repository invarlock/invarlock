# Guards

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Guard checks that validate edits against baseline-derived contracts. |
| **Audience** | Users tuning guard behavior and reviewing report evidence. |
| **Supported guards** | `invariants`, `spectral`, `rmt`, `variance` (plus optional plugin guards). |
| **Requires** | `invarlock[guards]` for torch/numpy guard math. |
| **Network** | Offline by default; guard logic itself is local. |
| **Inputs** | Model, adapter, calibration data, tier policy (`--tier`/`auto_config`). |
| **Outputs / Artifacts** | `report.guards` entries, report `resolved_policy`, `validation.*` flags. |
| **Source of truth** | `src/invarlock/guards/*.py`, `src/invarlock/guards/policies.py`, packaged `runtime/tiers.yaml`. |

See the [Glossary](../assurance/glossary.md) for definitions of guard terms such
as kappa threshold, epsilon band, and guard metric impact.

## Evidence Maturity and Enforcement

| Surface | Empirical maturity | Current strict behavior | Interpretation |
| --- | --- | --- | --- |
| Paired primary metric | **Implemented, recomputed gate** | Must satisfy the configured paired regression policy. | Main baseline-versus-subject decision; field sensitivity depends on the selected data, metric, and thresholds. It is documented with reports because it is not a guard plugin. |
| Invariants | **Stable blocking guard** | Structural and non-finite findings block. | Fail-closed integrity evidence. |
| Spectral | **Operational diagnostic** | Complete findings block under `enforce` and remain visible under `observe`. | Baseline-relative weight diagnostic with a versioned measurement contract; interpret only within calibrated scope. |
| RMT | **Experimental diagnostic** | Complete epsilon findings block under `enforce` and remain visible under `observe`. | Activation edge-risk diagnostic; interpret only within its evaluated sampling and family scope. |
| Variance/VE | **Experimental intervention** | The predictive gate must be evaluated; a complete failing predictive-gate outcome blocks under `enforce` and remains visible under `observe`. | A/B remediation/intervention evidence whose usefulness depends on workload-specific calibration and paired evidence. |

Maturity is not the same as enforcement. Current strict verification always
requires the canonical guard execution and complete evidence that can be replayed.
`resolved_policy.guard_authority` chooses whether a complete spectral, RMT, or
variance finding has `enforce` or `observe` acceptance authority. Observation
does not disable a guard and cannot waive missing, unsupported, degraded,
monitor-only, or incomplete evidence. Primary metrics, drift, invariants, and
guard-metric impact remain mandatory blockers. The maturity labels communicate
how broadly to interpret each signal; they do not introduce separate CLI modes.

All shipped tiers default spectral, RMT, and variance authority to `enforce`.
Selecting `observe` is an explicit policy override that requires deliberate
authorization for a complete, measured finding; it is not an evidence bypass.

The paired primary metric is the implemented behavioral acceptance surface. Guards
add structural checks and checkpoint-internal diagnostics; they do not replace
task evaluation, external benchmarks, or deployment monitoring.

## Quick Start

This example deliberately overrides the shipped all-`enforce` authority for
RMT and variance so their complete measured findings remain observable without
changing the mandatory evidence requirements.

```yaml
guards:
  order: ["invariants", "spectral", "rmt", "variance", "invariants"]
  authority:
    spectral: enforce
    rmt: observe
    variance: observe
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
> sparingly and keep evidence in the report.
>
> `contracts/support_matrix.json` records the maintained lanes and their current
> evidence status. Family-specific calibration studies can refine the shipped
> tier defaults for a selected workload.

## Guard Pipeline Flow

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                        GUARD PIPELINE FLOW                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   prepare model + prepare all guards                                   │
│                         │                                               │
│                         ▼                                               │
│   invariants(pre).validate                                              │
│                         │                                               │
│                         ▼                                               │
│   edit/noop stage                                                       │
│                         │                                               │
│                         ▼                                               │
│   spectral.validate → rmt.validate → variance.validate                  │
│                         │                                               │
│                         ▼                                               │
│   invariants(post).validate → evaluate → finalize                       │
│                         │                                               │
│                         ▼                                               │
│   report guard statuses, metrics, and measurement-contract hashes      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Concepts

- **Guard lifecycle**: the core runner calls `prepare(...)` (if implemented)
  and always calls `validate(...)`. Optional hooks (`before_edit`, `after_edit`,
  `finalize`) are only used when you manage guards manually (e.g., with
  `GuardChain`).
- **Tier policies**: `--tier balanced|conservative|aggressive` resolves a full
  policy bundle from packaged `runtime/tiers.yaml`; overrides
  in config are merged on top.
- **Measurement contracts**: Spectral and RMT guards record estimator + sampling
  contracts in reports and are enforced by `invarlock verify` in CI/Release,
  alongside required `runtime.manifest.json` runtime provenance for evaluation outputs.

### Guard hooks

| Hook | When called | Evidence |
| --- | --- | --- |
| `prepare` | Before edit (GuardWithPrepare only). | `report.meta.tier_policies`, `report.meta.guard_prepare_failures` (when prepare fails). |
| `validate` | After edit. | `report.guards[].passed`, `report.guards[].decision`, `report.guards[].diagnostics`, `report.guards[].violations`. |

### Verify gate requirements

| Gate | Required fields | Applies |
| --- | --- | --- |
| Measurement contracts | `spectral.measurement_contract_hash`, `rmt.measurement_contract_hash`, `resolved_policy.*`. | CI/Release. |
| Guard metric degradation | `guard_metric_impact.{metric_kind,direction,degradation_basis,bare_value,guarded_value,degradation,degradation_limit,display_value,display_unit}`. | Release only; kind-specific values are recomputed. |
| Validation allow‑list | `validation.*` booleans. | Schema validation. |

## Reference

### Guard summary

| Guard | Purpose | Key knobs (override) | Evidence (report/report) |
| --- | --- | --- | --- |
| `invariants` | Structural integrity + non-finite checks. | `strict_mode`, `on_fail`, `profile_checks`. | `validation.invariants_pass`, `invariants.*`. |
| `spectral` | Baseline-relative spectral norm stability. | `sigma_quantile`, `family_caps`, `deadband`, `scope`, `correction_enabled`, `estimator`, `degeneracy`, `multiple_testing`. | `validation.spectral_stable`, `spectral.*`, `resolved_policy.spectral`. |
| `rmt` | Activation edge-risk stability (ε-band). | `epsilon_default`, `epsilon_by_family`, `deadband`, `margin`, `correct`, `estimator`, `activation.sampling`. | `validation.rmt_stable`, `rmt.*`, `resolved_policy.rmt`. |
| `variance` | Variance equalization with A/B gate. | `min_gain`, `min_effect_lognll`, `max_calib`, `scope`, `clamp`, `deadband`, `predictive_gate`, `predictive_one_sided`, `calibration`, `tap`. | `variance.*`, `resolved_policy.variance`. |

### Guard evidence matrix

| Guard config | Report evidence | report evidence | Verify gate |
| --- | --- | --- | --- |
| `guards.invariants.*` | `report.guards[name=invariants]` | `report.invariants`, `validation.invariants_pass` | Schema only. |
| `guards.spectral.*` | `report.guards[name=spectral]` | `report.spectral`, `resolved_policy.spectral`, `validation.spectral_stable` | Measurement contracts (CI/Release). |
| `guards.rmt.*` | `report.guards[name=rmt]` | `report.rmt`, `resolved_policy.rmt`, `validation.rmt_stable` | Measurement contracts (CI/Release). |
| `guards.variance.*` | `report.guards[name=variance]` | `report.variance`, `resolved_policy.variance` | Schema only. |
| `--profile release` | `report.guard_metric_impact` | `report.guard_metric_impact` | Requires evaluated, passing paired evidence with the registered direction and degradation basis; skips fail. |

### Invariants Guard

```yaml
guards:
  invariants:
    strict_mode: false
    on_fail: monitor   # monitor | rollback | block
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

- **Guard prepare failed**: set `context.run.strict_guard_prepare: false` in
  your run config for local debugging, or adjust tier policies for the guard
  that failed.
- **Spectral instability**: inspect family dispersion and cap diagnostics before
  changing policy. `spectral.deadband` only affects the zero-standard-deviation
  fallback; it does not buffer the normal positive-variance z-score path.
- **RMT ε-band violations**: tighten calibration (more windows) or adjust
  `epsilon_by_family` only if you are updating tier policy evidence.
- **Variance guard never enables**: A/B gate may fail; inspect
  `variance.metrics.predictive_gate` and `variance.metrics.ab_gain` in the report.

## Observability

- `report.guards` contains guard results by name.
- reports include `resolved_policy.{spectral,rmt,variance}` and evidence
  blocks (`spectral.*`, `rmt.*`, `variance.*`).
- Validation flags are recorded under `validation.*` (`invariants_pass`,
  `spectral_stable`, `rmt_stable`).
- Reports may include `guard_warnings`. These are baseline-relative guard-signal
  changes that still pass the hard policy, such as a new capped spectral module
  while `caps_applied <= max_caps`. They are advisory by default and become
  verification failures only with `invarlock verify --warning-policy fail`.
- Evidence packs use the same guard observations but apply stricter scenario
  semantics. A public guard-value claim requires reproduced baseline-relative
  scenario evidence; an ordinary warning alone is not enough.

## Related Documentation

- [Tier Policy Catalog](tier-policy-catalog.md)
- [GPU/MPS-First Guard Measurement Contracts](../assurance/13-gpu-mps-first-guards.md)
- [Configuration Schema](config-schema.md)
- [Environment Variables](env-vars.md)
- [Guard Contracts & Primer](../assurance/04-guard-contracts.md)
