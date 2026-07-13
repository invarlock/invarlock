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
- Default runtime posture keeps model-loading commands inside the runtime
  container unless a public host-side workflow uses `invarlock evaluate --execution-mode host`
  or an advanced/internal workflow explicitly sets `INVARLOCK_ALLOW_HOST_EXECUTION=1`.
- Evaluation runs use the pairing, windowing, and bootstrap profiles
  described in the assurance docs and configs.

## Security Flow Overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                     SECURITY BOUNDARY LAYERS                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ NETWORK LAYER                                                     │  │
│  │ INVARLOCK_ALLOW_NETWORK=0 by default; outbound blocked unless     │  │
│  │ explicitly enabled.                                               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                               │                                         │
│                               ▼                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ RUNTIME LAYER                                                     │  │
│  │ container execution by default; host execution only with          │  │
│  │ `evaluate --execution-mode host` (public) or advanced/internal host bypass │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                               │                                         │
│                               ▼                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ ARTIFACT LAYER                                                    │  │
│  │ model loading | dataset loading | config loading                  │  │
│  │ adapter checks | pairing checks | schema checks                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                               │                                         │
│                               ▼                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ VALIDATION LAYER                                                  │  │
│  │ doctor preflight | evaluate | verify                              │  │
│  │ strict verify also receives the raw baseline report, acceptance   │  │
│  │ policy pack, and expected runtime-image digest                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                               │                                         │
│                               ▼                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ EVIDENCE LAYER                                                    │  │
│  │ evaluation.report.json + runtime.manifest.json                    │  │
│  │ verifier result and exit status                                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Assets and adversaries (in scope)

**Assets**

- Baseline and subject model weights for supported task families.
- Evaluation datasets, pairing schedules, and seed bundles.
- Evaluation artifacts: reports, logs, and policy digests.

**Adversaries / failure modes**

- Malicious or malformed model artifacts (e.g., unsafe pickle payloads) used as
  baselines or subjects.
- Misconfigured edits or guard policies that silently degrade quality or break
  structural invariants while “appearing to run”.
- Dependency vulnerabilities in the Python stack and transitive extras that
  could affect evaluation or guard logic.
- A malicious or compromised evidence author that can author a mutually
  consistent report, runtime manifest, policy digest, and signing key.
- Post-hoc dataset/window selection that preserves pairing while biasing the
  evaluated sample.

## Mitigations (built-in + process)

- Network guard (`invarlock.security`) denies outbound sockets by default; network
  use must be opted into per command.
- Runtime security defaults keep model-loading commands containerized, third-party
  plugins disabled, and remote model code off unless explicitly allowed.
- Supply-chain checks in CI and PR validation (install-surface SBOM
  generation, `pip-audit` on the base/`hf`/`advanced` shipped surfaces,
  `gitleaks` git-delta JSON scanning), with tag-gated release checks and a
  scheduled full-history secret scan for drift detection.
- CodeQL scans shipped Python code plus repository helper scripts, and the
  analysis workflow fails closed if upload/analysis cannot complete.
- Release automation only rebuilds and publishes from validated tags resolved
  to an immutable commit SHA.
- Strict configuration and report validation (`invarlock doctor`,
  `invarlock verify`) to detect misconfiguration, schema drift, report/manifest
  mismatch, thresholds outside an independently maintained policy pack, raw-baseline
  replay failures, and mismatch against an independently supplied runtime-image
  digest.
- report fields for seeds, windowing, dataset/tokenizer hashes, and guard
  telemetry so auditors can audit the assurance evidence.

## Attack Scenarios

Concrete attack scenarios InvarLock is designed to address or explicitly
delegates to external processes:

### 1. Poisoned Baseline Model

**Threat:** Attacker provides a pre-backdoored baseline that passes all guards.

**Mitigation:** Baseline provenance is the caller's responsibility. InvarLock
compares subject to baseline but does not validate baseline correctness.

**Detection:** None — baseline is trusted by design. Use external model
provenance checks (e.g., model cards, hash verification) before evaluation.

### 2. Malformed Pickle in Subject Checkpoint

**Threat:** Unsafe deserialization executes arbitrary code during model load.

**Mitigation:** InvarLock does not use pickle-capable adapter snapshot restore
in the default path, and adapters using `from_pretrained` inherit HF's
safetensors preference.

**Detection:** Invariants guard checks for non-finite values post-load; does
not catch code execution during load itself.

### 3. Edit That Evades Guards

**Threat:** Carefully crafted edit stays within spectral/RMT bounds but causes
task-specific degradation not captured by primary metric.

**Mitigation:** Primary metric gate + guard ensemble provides layered defense.
Tighten tier (conservative) for high-stakes releases.

**Detection:** `validation.primary_metric_acceptable = false` or guard warnings
in report. Manual review of `report.guards[]` evidence.

### 4. Configuration Drift Attack

**Threat:** Attacker modifies config to weaken guards (larger ε, disabled
checks) hoping auditors do not notice.

**Mitigation:** reports capture `resolved_policy.*` and `policy_digest` for
audit. Strict verification requires an independently maintained policy pack and
checks the report against it. A report-local digest alone is only a
self-consistency check.

**Detection:** Supply the separately controlled CI/release policy pack to
`invarlock verify --policy-pack`. `policy_digest.changed` and side-by-side
comparison remain useful signals, but an evidence author can recompute a
report-local digest after changing policy.

### 5. Window Schedule Manipulation

**Threat:** Attacker provides crafted baseline windows that inflate subject
performance (cherry-picked easy examples).

**Mitigation:** Pairing enforcement requires `window_match_fraction = 1.0` and
`window_overlap_fraction = 0.0`, which prevents baseline/subject schedule drift.
It does not prevent cherry-picking. Representative dataset identity and a
window plan must be independently fixed before results are observed.

**Detection:** `[INVARLOCK:E001]` detects pairing mismatch. Detecting biased
selection requires an independently committed dataset/window-plan digest and
review of the sampling protocol.

### 6. Fabricated Evidence Bundle

**Threat:** An evidence source fabricates a report and matching runtime
manifest, or signs both with a key controlled only by that source.

**Mitigation:** Strict report verification requires the exact raw baseline, a
independently maintained policy pack, and an expected runtime image digest supplied
independently of the submitted manifest. Evidence-pack review separately pins
an accepted publisher key. High-assurance deployments should also use
externally issued execution attestations or reproducible reruns.

**Detection:** `manifest_bound` proves only report/manifest integrity.
`expected_image_digest_matched` proves the declaration matches the digest the
caller supplied, not that the image executed. A pinned evidence-pack signer
authenticates the publisher, not the runtime execution.

## Out of scope (security non-goals)

These match the assurance **non-goals**:

- Multi-tenant GPU isolation, kernel-level sandboxing, and host hardening.
- Protection against prompt-level attacks, content harms (toxicity, bias,
  jailbreaks), or general alignment failures.
- Guarantees for environments outside the documented support matrix (e.g.,
  native Windows, arbitrary CUDA stacks, unpinned dependency versions).

## See also

- [Assurance Overview and scope](../assurance/00-assurance-case.md)
- [Security Best Practices](best-practices.md)
- [Security Architecture](architecture.md)
