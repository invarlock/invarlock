# Threat Model

This document provides a high-level threat model for InvarLock deployments. It is
intentionally aligned with the **assurance case scope**: InvarLock’s primary goal
is to control **regression risk from weight edits relative to a baseline**
under specified configurations, not to provide a complete solution to model
security or alignment.

## Assumptions

- Users operate in isolated virtual environments or containers on Linux/macOS
  hosts with supported HF/PyTorch versions.
- Models and datasets may be sourced from public repositories, but are treated
  as potentially untrusted artifacts.
- Default runtime posture disables outbound network connections unless
  `INVARLOCK_ALLOW_NETWORK=1` is explicitly set.
- Certification runs use the pairing, windowing, and bootstrap profiles
  described in the assurance docs and configs.

## Security Flow Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      SECURITY BOUNDARY LAYERS                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    NETWORK LAYER                                │   │
│  │  ┌───────────────────────────────────────────────────────────┐ │   │
│  │  │ INVARLOCK_ALLOW_NETWORK=0 (default)                       │ │   │
│  │  │ ─────────────────────────────────────                     │ │   │
│  │  │ • Socket blocking via invarlock.security                  │ │   │
│  │  │ • Outbound connections denied by default                  │ │   │
│  │  │ • Downloads require explicit opt-in                       │ │   │
│  │  └───────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    ARTIFACT LAYER                               │   │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐ │   │
│  │  │ MODEL LOADING   │  │ DATASET LOADING  │  │ CONFIG LOADING │ │   │
│  │  │ ─────────────── │  │ ────────────────  │  │ ──────────────  │ │   │
│  │  │ • Adapter       │  │ • Provider       │  │ • YAML parsing │ │   │
│  │  │   validation    │  │   validation     │  │ • Schema       │ │   │
│  │  │ • Torch device  │  │ • Tokenizer hash │  │   validation   │ │   │
│  │  │   placement     │  │ • Window pairing │  │ • include      │ │   │
│  │  │ • No pickle     │  │   verification   │  │   restriction  │ │   │
│  │  │   execution     │  │                  │  │   to config dir│ │   │
│  │  └─────────────────┘  └──────────────────┘  └────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    VALIDATION LAYER                             │   │
│  │  ┌─────────────────────────────────────────────────────────────┐│   │
│  │  │ invarlock doctor → invarlock certify → invarlock verify    ││   │
│  │  │ ────────────────────────────────────────────────────────── ││   │
│  │  │ • Environment    • Guard checks      • Schema validation   ││   │
│  │  │   diagnostics    • Pairing math      • Ratio math check    ││   │
│  │  │ • Config check   • CI/Release gates  • Contract match      ││   │
│  │  └─────────────────────────────────────────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    EVIDENCE LAYER                               │   │
│  │  ┌────────────────────────────────────────────────────────────┐ │   │
│  │  │ evaluation.cert.json                                       │ │   │
│  │  │ ────────────────────────────────                           │ │   │
│  │  │ • seeds, device, policy_digest                             │ │   │
│  │  │ • tokenizer_hash, provider_digest                          │ │   │
│  │  │ • measurement_contract_hash (guards)                       │ │   │
│  │  │ • Events log for forensic review                           │ │   │
│  │  └────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Assets and adversaries (in scope)

**Assets**

- Baseline and subject model weights for supported task families.
- Evaluation datasets, pairing schedules, and seed bundles.
- Certification artifacts: reports, certificates, logs, and policy digests.

**Adversaries / failure modes**

- Malicious or malformed model artifacts (e.g., unsafe pickle payloads) used as
  baselines or subjects.
- Misconfigured edits or guard policies that silently degrade quality or break
  structural invariants while still “appearing to run”.
- Dependency vulnerabilities in the Python stack and transitive extras that
  could affect evaluation or guard logic.

## Mitigations (built-in + process)

- Network guard (`invarlock.security`) denies outbound sockets by default; network
  use must be opted into per command.
- Supply-chain checks in CI (SBOM generation, `pip-audit`, secret scanning).
- Strict configuration and certificate validation (`invarlock doctor`,
  `invarlock verify`) to detect misconfiguration and schema drift.
- Certificate fields for seeds, windowing, dataset/tokenizer hashes, and guard
  telemetry so reviewers can audit the safety case.

## Attack Scenarios

Concrete attack scenarios InvarLock is designed to address or explicitly
delegates to external processes:

### 1. Poisoned Baseline Model

**Threat:** Attacker provides a pre-backdoored baseline that passes all guards.

**Mitigation:** Baseline provenance is the caller's responsibility. InvarLock
compares subject to baseline but does not validate baseline correctness.

**Detection:** None — baseline is trusted by design. Use external model
provenance checks (e.g., model cards, hash verification) before certification.

### 2. Malformed Pickle in Subject Checkpoint

**Threat:** Unsafe deserialization executes arbitrary code during model load.

**Mitigation:** Use `weights_only=True` when available in PyTorch. Adapters
using `from_pretrained` inherit HF's safetensors preference.

**Detection:** Invariants guard checks for non-finite values post-load; does
not catch code execution during load itself.

### 3. Edit That Evades Guards

**Threat:** Carefully crafted edit stays within spectral/RMT bounds but causes
task-specific degradation not captured by primary metric.

**Mitigation:** Primary metric gate + guard ensemble provides layered defense.
Tighten tier (conservative) for high-stakes releases.

**Detection:** `validation.primary_metric_acceptable = false` or guard warnings
in certificate. Manual review of `report.guards[]` evidence.

### 4. Configuration Drift Attack

**Threat:** Attacker modifies config to weaken guards (larger ε, disabled
checks) hoping reviewers don't notice.

**Mitigation:** Certificates capture `resolved_policy.*` and `policy_digest`
for audit. `invarlock verify` enforces schema compliance.

**Detection:** Policy changes appear in `policy_digest.changed = true`.
Compare certificates side-by-side for unexpected policy drift.

### 5. Window Schedule Manipulation

**Threat:** Attacker provides crafted baseline windows that inflate subject
performance (cherry-picked easy examples).

**Mitigation:** Pairing enforcement requires `window_match_fraction = 1.0` and
`window_overlap_fraction = 0.0`. CI/Release profiles fail on pairing violations.

**Detection:** `[INVARLOCK:E001]` error on pairing schedule mismatch.

## Out of scope (security non-goals)

These match the assurance **non-goals**:

- Multi-tenant GPU isolation, kernel-level sandboxing, and host hardening.
- Protection against prompt-level attacks, content harms (toxicity, bias,
  jailbreaks), or general alignment failures.
- Guarantees for environments outside the documented support matrix (e.g.,
  native Windows, arbitrary CUDA stacks, unpinned dependency versions).

## See also

- [Assurance Overview and scope](../assurance/00-safety-case.md)
- [Security Best Practices](best-practices.md)
- [Security Architecture](architecture.md)
