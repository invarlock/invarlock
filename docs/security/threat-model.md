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

## Out of scope (security non-goals)

These match the assurance **non-goals**:

- Multi-tenant GPU isolation, kernel-level sandboxing, and host hardening.
- Protection against prompt-level attacks, content harms (toxicity, bias,
  jailbreaks), or general alignment failures.
- Guarantees for environments outside the documented support matrix (e.g.,
  native Windows, arbitrary CUDA stacks, unpinned dependency versions).

## See also

- Assurance Overview and scope: `../assurance/00-safety-case.md`
- Security Best Practices: `./best-practices.md`
- Security Architecture: `./architecture.md`
