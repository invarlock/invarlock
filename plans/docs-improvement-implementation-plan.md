# Documentation Improvement Implementation Plan

## Overview

This phased plan consolidates all recommendations from the 5 review documents to improve InvarLock's documentation. The plan is organized into 4 phases with clear deliverables, effort estimates, and dependencies.

**Source Documents:**
- [`docs-reference-review.md`](docs-reference-review.md) - 14 recommendations
- [`docs-user-guide-review.md`](docs-user-guide-review.md) - 11 recommendations
- [`docs-assurance-review.md`](docs-assurance-review.md) - 12 recommendations
- [`docs-security-review.md`](docs-security-review.md) - 10 recommendations
- [`docs-comprehensive-review.md`](docs-comprehensive-review.md) - Cross-section issues

**Total Recommendations:** 57 items consolidated into 32 actionable tasks

---

## Phase 1: Critical Fixes (Day 1-2)

**Goal:** Fix navigation gaps and broken discoverability. All high-confidence, low-effort items.

**Estimated Total Effort:** 2-3 hours

### Task 1.1: Update docs/README.md Navigation
**Source:** Comprehensive Review, Security Review

**Description:** Add 10 missing files to the main documentation hub.

**Files to add:**
```markdown
## User Guide
+ - [Plugins](user-guide/plugins.md) — Extending adapters and guards
+ - [Bring Your Own Data](user-guide/bring-your-own-data.md) — Custom datasets
+ - [Proof Packs](user-guide/proof-packs.md) — Validation suite bundles
+ - [Proof Packs Internals](user-guide/proof-packs-internals.md) — Suite architecture

## Reference
+ - [API Guide](reference/api-guide.md) — Programmatic Python interface
+ - [Programmatic Quickstart](reference/programmatic-quickstart.md) — Minimal Python example
+ - [Environment Variables](reference/env-vars.md) — Runtime toggles

## Security
+ - [Threat Model](security/threat-model.md) — Assets and adversaries
+ - [Security Architecture](security/architecture.md) — Components and defaults
+ - [Best Practices](security/best-practices.md) — Operational recommendations
```

| Confidence | Effort | Impact |
|------------|--------|--------|
| 98% | 15 min | Critical |

**Acceptance Criteria:**
- [ ] All 10 files listed in docs/README.md
- [ ] Links are correct and working
- [ ] Order is logical within each section

---

### Task 1.2: Rename certificate_telemetry.md
**Source:** Reference Review, Comprehensive Review

**Description:** Fix naming inconsistency - only file using underscore instead of hyphens.

**Action:**
```bash
git mv docs/reference/certificate_telemetry.md docs/reference/certificate-telemetry.md
```

**Post-rename updates:**
- Update links in `certificate-schema.md`
- Update links in `artifacts.md`
- Update docs/README.md if listed

| Confidence | Effort | Impact |
|------------|--------|--------|
| 95% | 10 min | Low |

**Acceptance Criteria:**
- [ ] File renamed to `certificate-telemetry.md`
- [ ] All internal links updated
- [ ] No broken links in docs

---

### Task 1.3: Add TL;DR to Safety Case
**Source:** Assurance Review

**Description:** Add accessible plain-English summary at top of `00-safety-case.md`.

**Content to add:**
```markdown
## TL;DR

InvarLock certifies that **weight edits don't cause unacceptable regression** 
relative to a baseline under tested conditions. It does NOT certify:
- Model safety against adversarial prompts
- Alignment or content-harm prevention
- Non-weight changes like training data shifts

Key guarantees (with evidence):
1. Primary metric stays within tier limits (±5-10% PPL)
2. Spectral norms stay within calibrated bounds
3. Activation edge-risk stays within ε-bands
4. Guard overhead < 1% of baseline
```

| Confidence | Effort | Impact |
|------------|--------|--------|
| 85% | 15 min | High |

**Acceptance Criteria:**
- [ ] TL;DR section added at top after title
- [ ] Plain language, no jargon
- [ ] Links to detailed sections

---

### Task 1.4: Add TOC to Guard Contracts
**Source:** Assurance Review

**Description:** Add table of contents to `04-guard-contracts.md` for navigation.

**Content to add:**
```markdown
## Contents
1. [Guard Contracts](#1-guard-contracts)
   - [Invariants](#invariants-what-is-checked)
   - [Spectral](#spectral-guard)
   - [RMT](#rmt-guard)
   - [Variance](#variance-ve-guard)
2. [Statistical Method Primer](#2-statistical-method-primer)
3. [Calibration Requirements](#3-calibration--evaluation-slice-requirements)
4. [Reproducibility Kit](#4-reproducibility-kit)
5. [Device Tolerance](#5-device-tolerance-guidance)
6. [Threshold Rationale](#6-threshold-rationale-defaults)
7. [Known Limitations](#7-known-limitations)
8. [Coverage Reference](#8-coverage-reference)
```

| Confidence | Effort | Impact |
|------------|--------|--------|
| 90% | 15 min | Medium |

**Acceptance Criteria:**
- [ ] TOC added after overview
- [ ] All anchor links work
- [ ] Covers all major sections

---

### Task 1.5: Add Security Checklist
**Source:** Security Review

**Description:** Add actionable checklist to `best-practices.md`.

**Content to add:**
```markdown
## Pre-Deployment Security Checklist

- [ ] Environment isolated (pipx/venv/conda)
- [ ] Dependencies locked and audited
- [ ] `INVARLOCK_ALLOW_NETWORK` disabled by default
- [ ] Model/dataset sources vetted for provenance
- [ ] `invarlock verify` run on all certificates before promotion
- [ ] CI pipeline includes `pip-audit` step
- [ ] Secrets not present in configs or logs
- [ ] Temp directories use `secure_tempdir()`
```

| Confidence | Effort | Impact |
|------------|--------|--------|
| 85% | 15 min | Medium |

**Acceptance Criteria:**
- [ ] Checklist added to best-practices.md
- [ ] Markdown checkbox format
- [ ] All items actionable

---

### Task 1.6: Add User Journey Navigation
**Source:** User Guide Review

**Description:** Add learning paths to `getting-started.md`.

**Content to add:**
```markdown
## Learning Paths

**First-time setup:**
getting-started → quickstart → compare-and-certify

**Python developers:**
getting-started → primary-metric-smoke → API Guide (ref)

**Custom data users:**
getting-started → bring-your-own-data → config-gallery

**Plugin developers:**
getting-started → plugins → Guards Reference (ref)

**Validation engineers:**
getting-started → proof-packs → proof-packs-internals
```

| Confidence | Effort | Impact |
|------------|--------|--------|
| 85% | 20 min | High |

**Acceptance Criteria:**
- [ ] Learning paths section added
- [ ] Links are correct
- [ ] Covers main user personas

---

## Phase 2: High-Value Additions (Week 1)

**Goal:** Create missing documentation and improve cross-linking. Medium effort, high impact.

**Estimated Total Effort:** 8-10 hours

### Task 2.1: Create reference/calibration.md
**Source:** Comprehensive Review

**Description:** Document the undocumented `invarlock calibrate` CLI command.

**Outline:**
```markdown
# Calibration Reference

## Overview
| Aspect | Details |
| --- | --- |
| Purpose | Run calibration sweeps for tier policy values |
| Audience | Operators tuning guard thresholds |
| Commands | `calibrate null-sweep`, `calibrate ve-sweep` |
| Outputs | JSON, CSV, and Markdown reports |

## Quick Start
<!-- Note: These commands are proposed and may not exist yet -->
# invarlock calibrate null-sweep --model gpt2 --out calibration/spectral
# invarlock calibrate ve-sweep --model gpt2 --out calibration/ve

## Concepts
- When to recalibrate
- Null-sweep methodology (spectral κ)
- VE-sweep methodology (min_effect)
- Output interpretation

## Reference
### null-sweep options
### ve-sweep options
### Output formats

## Troubleshooting

## Related Documentation
- [Tier v1 Calibration](../assurance/09-tier-v1-calibration.md)
- [Tier Policy Catalog](tier-policy-catalog.md)
```

| Confidence | Effort | Impact |
|------------|--------|--------|
| 95% | 2 hrs | High |

**Acceptance Criteria:**
- [ ] Document created at `docs/reference/calibration.md`
- [ ] Both subcommands documented
- [ ] Examples included
- [ ] Added to docs/README.md

---

### Task 2.2: Create reference/index.md
**Source:** Reference Review

**Description:** Create navigation page for reference section with audience-based links.

**Content:**
```markdown
# Reference Documentation

## By Audience

### CLI Users
- [CLI Reference](cli.md) - Complete command documentation
- [Configuration Schema](config-schema.md) - YAML config options
- [Environment Variables](env-vars.md) - Runtime toggles

### Python Developers
- [API Guide](api-guide.md) - Programmatic interface
- [Programmatic Quickstart](programmatic-quickstart.md) - Minimal example
- [Model Adapters](model-adapters.md) - Loading models

### Operators & Auditors
- [Certificate Schema](certificate-schema.md) - Understanding certificates
- [Artifact Layout](artifacts.md) - Where outputs live
- [Tier Policy Catalog](tier-policy-catalog.md) - Guard thresholds

### Guard Configuration
- [Guards](guards.md) - Guard setup and tuning
- [Calibration](calibration.md) - Recalibrating thresholds
- [Datasets](datasets.md) - Evaluation data providers

## Quick Links
| Task | Document |
|------|----------|
| Run first certification | [CLI Reference](cli.md#compare--certify) |
| Understand a certificate | [Certificate Schema](certificate-schema.md) |
| Write Python integration | [API Guide](api-guide.md) |
| Configure guards | [Guards](guards.md) |
| Choose a tier | [Tier Policy Catalog](tier-policy-catalog.md) |
```

| Confidence | Effort | Impact |
|------------|--------|--------|
| 90% | 45 min | High |

**Acceptance Criteria:**
- [ ] Document created at `docs/reference/index.md`
- [ ] All links working
- [ ] Audience sections complete
- [ ] Quick links table accurate

---

### Task 2.3: Add Cross-Section Links
**Source:** Comprehensive Review, User Guide Review, Assurance Review

**Description:** Add targeted cross-links between sections.

**Links to add:**

| Source File | Add Link To | Reason |
|-------------|-------------|--------|
| `user-guide/compare-and-certify.md` | `assurance/02-coverage-and-pairing.md` | Pairing details |
| `user-guide/quickstart.md` | `assurance/04-guard-contracts.md` | Guard explanation |
| `user-guide/reading-certificate.md` | `assurance/00-safety-case.md` | Safety claims |
| `user-guide/primary-metric-smoke.md` | `assurance/01-eval-math-proof.md` | PM math |
| `reference/certificate-schema.md` | `assurance/00-safety-case.md` | Claims backing |
| `reference/api-guide.md` | `assurance/08-determinism-contracts.md` | Determinism |
| `reference/datasets.md` | `assurance/02-coverage-and-pairing.md` | Windows |
| `assurance/04-guard-contracts.md` | `reference/guards.md` | Implementation |
| `assurance/09-tier-v1-calibration.md` | `reference/tier-policy-catalog.md` | Policy details |

**Format for links:**
```markdown
## Related Documentation
...
- [Coverage & Pairing (Assurance)](../assurance/02-coverage-and-pairing.md) - Window requirements
```

| Confidence | Effort | Impact |
|------------|--------|--------|
| 90% | 1 hr | Medium |

**Acceptance Criteria:**
- [ ] All listed links added
- [ ] Links appear in Related Documentation sections
- [ ] Brief context for each link

---

### Task 2.4: Create user-guide/troubleshooting.md
**Source:** User Guide Review

**Description:** Consolidate error messages and fixes from scattered locations.

**Outline:**
```markdown
# Troubleshooting Guide

## Common Error Codes

| Code | Meaning | Common Fix |
|------|---------|------------|
| E001 | Pairing schedule mismatch | Ensure baseline windows match subject |
| E111 | Primary metric degraded/non-finite | Check device, force float32, reduce batch |

## Network Issues
- Downloads blocked
- Offline mode issues
- HuggingFace cache problems

## Pairing Failures
- Window count mismatches
- Overlap issues
- Coverage shortfalls

## Device Compatibility
- CPU fallback
- MPS issues on Apple Silicon
- CUDA determinism

## Non-Finite Metrics
- Causes and diagnosis
- dtype recommendations
- Batch size tuning

## Guard Failures
- Spectral caps exceeded
- RMT ε-band violations
- VE gate not enabling

## See Also
- [CLI Reference](../reference/cli.md#early-stops-cirelease)
- [Environment Variables](../reference/env-vars.md)
- [Guard Contracts](../assurance/04-guard-contracts.md)
```

| Confidence | Effort | Impact |
|------------|--------|--------|
| 85% | 2 hrs | High |

**Acceptance Criteria:**
- [ ] Document created
- [ ] Covers all common error codes
- [ ] Each section has actionable fixes
- [ ] Added to docs/README.md

---

### Task 2.5: Expand reading-certificate.md
**Source:** User Guide Review

**Description:** Expand the brief bullet list with examples and context.

**Content to add:**
- Complete JSON example showing key fields
- Visual diagram of certificate structure
- "What to look for first" guide
- Links to certificate-schema.md for details

| Confidence | Effort | Impact |
|------------|--------|--------|
| 85% | 1 hr | High |

**Acceptance Criteria:**
- [ ] At least one JSON example added
- [ ] Interpretation guidance for each section
- [ ] Links to detailed schema docs

---

### Task 2.6: Add Tier Quick Comparison
**Source:** Reference Review

**Description:** Add summary table to `tier-policy-catalog.md`.

**Content to add at top:**
```markdown
## Quick Comparison

| Aspect | Balanced | Conservative |
|--------|----------|--------------|
| PM Ratio Limit | ≤ 1.10 | ≤ 1.05 |
| Spectral Dead Band | 0.10 | 0.05 |
| Spectral Max Caps | 5 | 3 |
| VE Gate | One-sided | Two-sided |
| Min Windows | 180/180 | 220/220 |
| Use Case | Standard edits | High-stakes releases |
```

| Confidence | Effort | Impact |
|------------|--------|--------|
| 85% | 20 min | High |

**Acceptance Criteria:**
- [ ] Comparison table at top of file
- [ ] All key tier differences covered
- [ ] Values match tiers.yaml

---

## Phase 3: Consolidation (Week 2)

**Goal:** Merge related documents and improve organization. Higher effort, structural improvements.

**Estimated Total Effort:** 6-8 hours

### Task 3.1: Consolidate Certificate Documentation
**Source:** Reference Review

**Description:** Merge 3 certificate-related files into one comprehensive document.

**Files to merge:**
- `certificate-schema.md` (289 lines) → Base document
- `certificate-telemetry.md` (48 lines) → New section
- `exporting-certificates-html.md` (57 lines) → New section

**New structure:**
```markdown
# Certificate Reference

## Overview
## Quick Start
## Schema (v1)
  - [existing certificate-schema.md content]
## Telemetry Fields
  - [moved from certificate-telemetry.md]
## Exporting
  - HTML export
  - [moved from exporting-certificates-html.md]
## Validation
## Troubleshooting
## Related Documentation
```

**Post-merge actions:**
- Create redirects or update all links
- Update docs/README.md
- Delete merged files

| Confidence | Effort | Impact |
|------------|--------|--------|
| 85% | 2 hrs | High |

**Acceptance Criteria:**
- [ ] Single `certificates.md` file created
- [ ] All content from 3 files preserved
- [ ] All links updated
- [ ] Old files removed (or redirected)

---

### Task 3.2: Relocate gpu-mps-first-guards.md
**Source:** Reference Review, Assurance Review

**Description:** Move decision memo from reference to assurance section.

**Actions:**
```bash
git mv docs/reference/gpu-mps-first-guards.md docs/assurance/13-gpu-mps-first-guards.md
```

**Post-move updates:**
- Update links in `reference/guards.md`
- Add brief reference in guards.md: "For design rationale, see [GPU/MPS-First Guards](../assurance/13-gpu-mps-first-guards.md)"
- Update docs/README.md

| Confidence | Effort | Impact |
|------------|--------|--------|
| 80% | 30 min | Medium |

**Acceptance Criteria:**
- [ ] File moved to assurance/
- [ ] Numbered consistently with existing files
- [ ] All links updated
- [ ] Brief reference added to guards.md

---

### Task 3.3: Standardize Overview Tables
**Source:** Reference Review

**Description:** Add Overview tables to files missing them in reference section.

**Files needing Overview tables:**
- `cli.md` (add Source of truth)
- `exporting-certificates-html.md` (before merge; more complete)

**Template:**
```markdown
## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | ... |
| **Audience** | ... |
| **Requires** | ... |
| **Network** | Offline by default; ... |
| **Source of truth** | `src/invarlock/...` |
```

| Confidence | Effort | Impact |
|------------|--------|--------|
| 90% | 45 min | Medium |

**Acceptance Criteria:**
- [ ] All reference docs have Overview table
- [ ] Consistent format
- [ ] Network default always stated

---

### Task 3.4: Differentiate Getting Started vs Quickstart
**Source:** User Guide Review

**Description:** Clarify purposes and remove overlap between these files.

**Proposed split:**
- `getting-started.md` → **Setup only**: Install, environment, doctor, network config
- `quickstart.md` → **Workflows only**: Commands, plugins list, guards, reports

**Content moves:**
- Move Compare & Certify intro from getting-started to quickstart
- Keep "Run The Automation Loop" in getting-started as transition

| Confidence | Effort | Impact |
|------------|--------|--------|
| 80% | 1 hr | Medium |

**Acceptance Criteria:**
- [ ] Clear separation of concerns
- [ ] Minimal overlap
- [ ] Cross-links between files

---

### Task 3.5: Add Worked Example to Calibration Docs
**Source:** Assurance Review

**Description:** Add step-by-step recalibration example to `09-tier-v1-calibration.md`.

**Content to add:**
```markdown
## Worked Example: Recalibrating Spectral κ for a Custom Model

1. Run null baseline: `invarlock run -c custom_model.yaml --tier balanced`
2. Extract z-scores from report:
   ```python
   z_scores = report['guards']['spectral']['final_z_scores']
   ```
3. For FFN family (40 modules), allocate budget B(ffn) = ⌊5 × 40/120 + 0.5⌋ = 2
4. Sort |z| descending: [2.1, 1.8, 1.6, 1.5, ...]
5. Set κ(ffn) = 2nd largest |z| + margin = 1.8 + 0.1 = 1.9
6. Update local tiers override and re-run baseline
```

| Confidence | Effort | Impact |
|------------|--------|--------|
| 80% | 1 hr | High |

**Acceptance Criteria:**
- [ ] Complete worked example added
- [ ] Python code included
- [ ] Ties to existing methodology

---

## Phase 4: Enhancement (Week 3-4)

**Goal:** Visual improvements, advanced documentation, and polish. Lower priority, higher effort items.

**Estimated Total Effort:** 10-15 hours

### Task 4.1: Create reference/error-codes.md
**Source:** Comprehensive Review

**Description:** Consolidate all error codes into single reference document.

**Steps:**
1. Audit source code for all error codes
2. Extract meanings and contexts
3. Group by category (pairing, metrics, guards, etc.)
4. Add fix recommendations

| Confidence | Effort | Impact |
|------------|--------|--------|
| 80% | 2 hrs | Medium |

---

### Task 4.2: Create reference/observability.md
**Source:** Comprehensive Review

**Description:** Document the observability module capabilities.

**Outline:**
- Available metrics in reports
- Health check utilities
- Integration patterns
- Report fields reference

| Confidence | Effort | Impact |
|------------|--------|--------|
| 85% | 2 hrs | Medium |

---

### Task 4.3: Add Mermaid Diagrams
**Source:** Reference Review, Assurance Review, Security Review

**Description:** Add visual diagrams to key documentation.

**Diagrams to create:**

1. **Guards execution flow** (for guards.md):
```
graph LR
    A[prepare] --> B[validate]
    B --> C{passed?}
    C -->|Yes| D[continue]
    C -->|No| E[warn/rollback/abort]
```

2. **Evidence flow** (for guards.md):
```
graph LR
    A[guards.yaml] --> B[report.guards]
    B --> C[certificate.resolved_policy]
    C --> D[invarlock verify]
```

3. **Eval math** (for 01-eval-math-proof.md):
```
graph TD
    A[Window 1: t₁ tokens → Δℓ₁] --> E
    B[Window 2: t₂ tokens → Δℓ₂] --> E
    C[...] --> E
    D[Window n: tₙ tokens → Δℓₙ] --> E
    E[Weighted mean Δ̄] --> F[exp Δ̄ = ratio]
```

4. **Security component flow** (for architecture.md):
```
graph TD
    A[CLI Entry] --> B[enforce_default_security]
    B --> C{Network Check}
    C -->|Blocked| D[Offline Operation]
    C -->|ALLOW_NETWORK=1| E[Download Permitted]
```

| Confidence | Effort | Impact |
|------------|--------|--------|
| 80% | 3 hrs | Medium |

---

### Task 4.4: Split cli.md or Add TOC
**Source:** Reference Review

**Description:** Address the 690-line CLI reference file.

**Option A (Recommended): Add comprehensive TOC**
```markdown
## Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Commands](#reference)
  - [certify](#invarlock-certify)
  - [verify](#invarlock-verify)
  - [run](#invarlock-run)
  - [report](#invarlock-report-group)
  - [plugins](#invarlock-plugins-group)
  - [doctor](#invarlock-doctor)
- [JSON Output](#json-output-verify-and-plugins)
- [Profiles](#profile-reference-ci-vs-release)
- [Security](#security-defaults)
- [Troubleshooting](#troubleshooting)
```

**Option B: Split into focused files**
- `cli.md` - Core commands
- `cli-plugins.md` - Plugin management
- `cli-profiles.md` - Profile configuration
- `cli-json-output.md` - JSON schemas

| Confidence | Effort | Impact |
|------------|--------|--------|
| 70% | 3 hrs | Medium |

---

### Task 4.5: Expand Threat Model with Attack Scenarios
**Source:** Security Review

**Description:** Add concrete attack scenarios to `threat-model.md`.

**Content to add:**
```markdown
## Attack Scenarios

### 1. Poisoned Baseline Model
**Threat:** Attacker provides pre-backdoored baseline that passes guards.
**Mitigation:** Baseline provenance is caller's responsibility.
**Detection:** None - baseline is trusted by design.

### 2. Malformed Pickle in Subject Checkpoint
**Threat:** Unsafe deserialization executes arbitrary code.
**Mitigation:** Use `weights_only=True` when available.
**Detection:** Invariants guard checks for non-finite values post-load.

### 3. Edit That Evades Guards
**Threat:** Carefully crafted edit that stays within guard bounds but degrades tasks.
**Mitigation:** Primary metric gate catches regression.
**Detection:** `validation.primary_metric_acceptable = false`.
```

| Confidence | Effort | Impact |
|------------|--------|--------|
| 80% | 1 hr | High |

---

### Task 4.6: Expand Plugins Documentation
**Source:** User Guide Review

**Description:** Add complete examples to `plugins.md`.

**Content to add:**
- Full guard plugin with tests
- Adapter plugin example (currently missing)
- "Your First Plugin" walkthrough
- Plugin debugging tips

| Confidence | Effort | Impact |
|------------|--------|--------|
| 75% | 2 hrs | Medium |

---

### Task 4.7: Add Glossary
**Source:** Assurance Review

**Description:** Create terminology reference for technical terms.

**Location options:**
- A) Standalone `docs/glossary.md`
- B) Section in `assurance/00-safety-case.md`

**Terms to define:**
- BCa (Bias-Corrected and Accelerated)
- FWER (Family-Wise Error Rate)
- FDR (False Discovery Rate)
- κ (kappa - z-cap)
- ε (epsilon - acceptance band)
- VE (Variance Equalization)
- PM (Primary Metric)
- BYOE (Bring Your Own Edit)

| Confidence | Effort | Impact |
|------------|--------|--------|
| 75% | 45 min | Medium |

---

### Task 4.8: Add Config Gallery Explanations
**Source:** User Guide Review

**Description:** Expand `config-gallery.md` with context for each preset.

**Current:** File paths only (27 lines)
**Proposed:** Add use case and related links for each

| Confidence | Effort | Impact |
|------------|--------|--------|
| 80% | 30 min | Medium |

---

### Task 4.9: Add CVE Response Process
**Source:** Security Review

**Description:** Document CVE handling workflow in `pip-audit-allowlist.md`.

**Content to add:**
```markdown
## CVE Response Process

1. **Discovery:** New CVE found by `pip-audit` in CI
2. **Triage:** Maintainer assesses exploitability
3. **Decision:**
   - Exploitable: Patch immediately
   - Not exploitable: Add to allowlist with reason
4. **Tracking:** Entries reviewed monthly
5. **Removal:** Entry removed when upstream fix available
```

| Confidence | Effort | Impact |
|------------|--------|--------|
| 80% | 20 min | Medium |

---

### Task 4.10: Add Overview Tables to User Guide
**Source:** User Guide Review

**Description:** Add Overview tables to complex user-guide files (optional - lower priority).

**Files:**
- `proof-packs.md`
- `plugins.md`
- `bring-your-own-data.md`

| Confidence | Effort | Impact |
|------------|--------|--------|
| 70% | 1 hr | Low |

---

## Implementation Summary

### Phase Timeline

| Phase | Tasks | Effort | Timeline |
|-------|-------|--------|----------|
| 1: Critical Fixes | 6 tasks | 2-3 hrs | Day 1-2 |
| 2: High-Value Additions | 6 tasks | 8-10 hrs | Week 1 |
| 3: Consolidation | 5 tasks | 6-8 hrs | Week 2 |
| 4: Enhancement | 10 tasks | 10-15 hrs | Week 3-4 |

**Total: 27 tasks, ~30 hours of work**

### Confidence Distribution

| Confidence Range | Tasks |
|------------------|-------|
| 95-98% | 3 tasks |
| 85-90% | 12 tasks |
| 75-80% | 10 tasks |
| 70% | 2 tasks |

### Dependencies

```
Phase 1 (no dependencies)
    └── Task 1.1: Update README.md
    └── Task 1.2: Rename file
    └── Task 1.3-1.6: Content additions

Phase 2 (depends on Phase 1 completion)
    └── Task 2.1: calibration.md (needs README update first)
    └── Task 2.2: index.md (needs all reference docs stable)
    └── Task 2.3: Cross-links (needs stable file locations)
    └── Task 2.4-2.6: Content additions

Phase 3 (depends on Phase 2 completion)
    └── Task 3.1: Certificate consolidation (after all links stable)
    └── Task 3.2: File relocation (after cross-links)
    └── Task 3.3-3.5: Content standardization

Phase 4 (can start partially in parallel with Phase 3)
    └── Independent enhancement tasks
```

### Quick Reference: All Tasks by Confidence

**98% Confidence:**
- Task 1.1: Update docs/README.md

**95% Confidence:**
- Task 1.2: Rename certificate_telemetry.md
- Task 2.1: Create calibration.md

**90% Confidence:**
- Task 1.4: Add TOC to guard contracts
- Task 2.2: Create index.md
- Task 2.3: Add cross-section links
- Task 3.3: Standardize Overview tables

**85% Confidence:**
- Task 1.3: Add TL;DR to safety case
- Task 1.5: Add security checklist
- Task 1.6: Add user journey navigation
- Task 2.4: Create troubleshooting.md
- Task 2.5: Expand reading-certificate.md
- Task 2.6: Add tier quick comparison
- Task 3.1: Consolidate certificate docs
- Task 4.2: Create observability.md

**80% Confidence:**
- Task 3.2: Relocate gpu-mps-first-guards.md
- Task 3.4: Differentiate getting-started vs quickstart
- Task 3.5: Add worked calibration example
- Task 4.1: Create error-codes.md
- Task 4.3: Add Mermaid diagrams
- Task 4.5: Expand threat model
- Task 4.8: Expand config-gallery
- Task 4.9: Add CVE response process

**75% Confidence:**
- Task 4.6: Expand plugins.md
- Task 4.7: Add glossary

**70% Confidence:**
- Task 4.4: Split/TOC cli.md
- Task 4.10: Add overview tables to user-guide
