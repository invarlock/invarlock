# System Architecture

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Auditable strict-verification framework for ML model weight modifications. |
| **Audience** | Developers extending InvarLock, operators debugging pipelines, security auditors. |
| **Core components** | CLI shells, Core/runtime policy layer, Guard chain, Reporting/artifact subsystem. |
| **Design goals** | Torch-independent core, edit-stack-neutral guards, deterministic evaluation, explicit artifact contracts, and reviewable report-recorded provenance. |
| **Source of truth** | `src/invarlock/core/*.py`, `src/invarlock/reporting/*.py`, `src/invarlock/runtime_provenance.py`, `src/invarlock/runtime_verify.py`, `src/invarlock/cli/commands/*.py`, `src/invarlock/cli/run_*.py`, `src/invarlock/guards/*.py`. |

See the [Glossary](../assurance/glossary.md) for definitions of terms such as
the canonical guard chain, policy digest, and measurement contract.

## Contents

1. [Quick Reference](#quick-reference)
2. [High-Level Architecture](#high-level-architecture)
3. [Component Layers](#component-layers)
4. [Pipeline Flow](#pipeline-flow)
5. [Guard Chain Architecture](#guard-chain-architecture)
6. [Report Generation Flow](#report-generation-flow)
7. [Architecture Guardrails](#architecture-guardrails)
8. [Key Design Decisions](#key-design-decisions)
9. [Module Dependencies](#module-dependencies)
10. [Extension Points](#extension-points)
11. [Related Documentation](#related-documentation)

## Quick Reference

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INVARLOCK SYSTEM OVERVIEW                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  baseline + subject + dataset/config                                       │
│                    │                                                        │
│                    ▼                                                        │
│  invarlock evaluate → adapters/edit stage → canonical guard chain → eval   │
│                    │                                                        │
│                    ▼                                                        │
│  raw report.json files → evaluation.report.json + runtime.manifest.json     │
│                                      │                                      │
│                  ┌───────────────────┴───────────────────┐                  │
│                  ▼                                       ▼                  │
│  independent inputs → invarlock verify            invarlock report html    │
│  baseline + policy + image digest                  review rendering         │
│                  │                                                          │
│                  ▼                                                          │
│          verified or rejected                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## High-Level Architecture

InvarLock follows a layered architecture with clear separation of concerns:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CLI SHELL LAYER                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ evaluate │ │  verify  │ │  report  │ │  doctor  │ │ advanced │           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │
│       │            │            │            │            │                 │
├───────┴────────────┴────────────┴────────────┴────────────┴─────────────────┤
│                     CORE POLICY / CONTRACT LAYER                            │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │ evaluate_plan · report_inputs · doctor_findings                │        │
│  │ verify_contract · retry · run_snapshot_contract                │        │
│  │ run_report_contract · runtime_verify                          │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                       CORE RUNTIME / SERVICES                               │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │ runner.py + runner_*                                          │        │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐     │        │
│  │  │prepare │─▶│inv(pre)│─▶│  edit  │─▶│ guards │─▶│  eval  │     │        │
│  │  │+ guards│  │validate│  │ / noop │  │validate│  │+ final │     │        │
│  │  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘     │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                            GUARD / MODEL LAYER                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐             │
│  │ invariants │  │  spectral  │  │    rmt     │  │  variance  │             │
│  │ (integrity)│  │  (weights) │  │(activation)│  │   (A/B)    │             │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘             │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                          REPORTING / FILES LAYER                            │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────┐ ┌────────────┐            │
│  │ report_make  │ │report_bundle │ │ rendering  │ │report_schema│           │
│  │ + console    │ │ + manifest   │ │  (MD/HTML) │ │   (JSON)   │            │
│  └──────────────┘ └──────────────┘ └────────────┘ └────────────┘            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Layers

### CLI Layer (`src/invarlock/cli/`)

Typer-based command shells providing user-facing entry points. The command
modules should stay thin: parse arguments, call core/reporting owners, render
output, and map failures to exit codes.

Shell support modules such as `cli/config_execution.py`, `cli/run_execution.py`,
`cli/run_config.py`, `cli/run_pairing.py`, and `cli/run_metric_impact.py` belong to
this boundary layer as well. They can perform CLI-facing adaptation and
console/event rendering; policy ownership stays in the core and reporting
owners.

Public model-loading commands use the runtime container by default.
`invarlock evaluate --execution-mode host` is the explicit host-side path.

| Command | Purpose | Primary Output |
| --- | --- | --- |
| `evaluate` | Compare baseline vs subject with pinned windows | Raw run reports plus canonical report/manifest bundle |
| `verify` | Validate report against schema and pairing | Exit code + messages |
| `report` | Render, explain, validate, or export reports | MD/HTML/JSON artifacts |
| `doctor` | Environment diagnostics | Health check output |
| `advanced` | Maintenance workflows such as evidence packs, policy packs, plugins, and calibration | Exit code + workflow-specific artifacts |
| `version` | Emit package and schema version information | Version string |

### Core Policy / Contracts (`src/invarlock/core/`, `src/invarlock/reporting/`)

Deterministic policy, artifact-contract, and report-verification owners shared
by the CLI and non-CLI entrypoints.

| Module | Responsibility |
| --- | --- |
| `evaluate_contract.py` | Baseline-report validation and emitted run-artifact contract enforcement for `evaluate` |
| `evaluate_plan.py` | Evaluation result policy, degradation classification, and emitted outcome shaping |
| `report_inputs.py` | Canonical report path resolution and JSON-object validation |
| `doctor_findings.py` | Structured doctor findings and optional report cross-check analysis |
| `verify_contract.py` | Structured report-verification service used by `verify` and evidence-pack flows |
| `runtime_verify.py` + `runtime_provenance.py` | Report/manifest binding checks and optional caller-supplied runtime-image digest matching |
| `run_policy.py` | Shared run policy helpers such as split choice, PM thresholds, and overhead policy |
| `retry.py` | Retry controller, edit-parameter adjustment, attempt summaries, and retry state transitions |
| `run_snapshot_contract.py` | Snapshot planning, restore behavior, and retry transitions |
| `run_report_contract.py` | Run provenance finalization, payload shaping, and run-report assembly contracts |

### Runtime Evidence Verification Ownership

Runtime evidence uses a single verifier implementation:

- `runtime_verify.py` is the authoritative programmatic verifier for
  `runtime.manifest.json`, report-digest binding, and comparison with an
  optional caller-supplied expected image digest.
- `cli/commands/verify.py` owns the CLI entrypoints for both report verification
  and advanced runtime-manifest verification.
- `runtime_provenance.py` calls the same verifier when `invarlock verify`
  enforces runtime evidence policy on container-backed reports.
- Product behavior does not depend on finding an external verifier binary on
  `PATH`; verifier semantics are package-native and deterministic for the same
  package version and input bytes.

This boundary verifies artifact consistency and an image-identity claim. It is
not remote attestation: a compromised evaluation environment can fabricate a consistent
report and manifest that name an expected digest. The expected digest must come
from an independent release/deployment channel, and isolated evaluation
infrastructure is still required when the evaluation environment is outside the trust boundary.

### Core Runtime (`src/invarlock/core/`)

Pipeline orchestration without direct torch imports (torch-independent coordination).

| Module | Responsibility |
| --- | --- |
| `runner.py` + `runner_*.py` | Pipeline phases: prepare → guards → edit → eval → finalize |
| `api.py` | Protocol definitions for ModelAdapter, ModelEdit, Guard |
| `bootstrap.py` | BCa bootstrap CI computation for paired metrics |
| `checkpoint.py` | Snapshot/restore primitives for retry loops |
| `registry.py` | Plugin discovery and registration |

### Guard Layer (`src/invarlock/guards/`)

Four-guard pipeline for edit safety validation.

| Guard | Focus | Key Metric |
| --- | --- | --- |
| `invariants` | Structural integrity, NaN/Inf checks | `validation.invariants_pass` |
| `spectral` | Weight matrix spectral norm stability | κ-threshold violations |
| `rmt` | Activation edge-risk via Random Matrix Theory | ε-band compliance |
| `variance` | Variance equalization with A/B gate | Predictive gain |

### Reporting Layer (`src/invarlock/reporting/`)

Report generation, validation, persistence, and rendering.

| Module | Responsibility |
| --- | --- |
| `report_schema.py` | Evaluation report schema and structural validation |
| `validation/report.py` | Canonical validation-flag computation |
| `report_make.py` | Evaluation-report input normalization, build-section extraction, output shaping, and public report assembly |
| `report_make_assembly.py` | Policy/provenance/guard assembly and report build-context composition |
| `report_bundle.py` | Evaluation-bundle persistence, manifest writing, and evidence attachment |
| `report_contract.py` | Input loading and report-generation planning |
| `report_metric_impact.py` | Guard-metric-impact normalization, summary building, and report shaping |
| `report_summary.py` | Console validation blocks and shared executive-summary/view-model derivation for reporting surfaces |
| `rendering/markdown.py` | Markdown rendering for evaluation reports |
| `html.py` | HTML export with styling |
| `core/guard_evidence.py` | Canonical guard-evidence normalization before reporting assembly |
| `report_builder_support.py` | Report build context, telemetry extraction, telemetry payloads, artifacts, and baseline references |

## Pipeline Flow

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EVALUATION PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   PHASE 1: BASELINE RUN                                                     │
│   ─────────────────────                                                     │
│   load baseline → prepare guards → guard validation (noop) → evaluate       │
│                                                    → baseline/report.json    │
│                                                                             │
│   PHASE 2: SUBJECT RUN (with baseline window pinning)                       │
│   ───────────────────────────────────────────────                           │
│   load subject → prepare guards → guard validation (edit/noop)              │
│                → evaluate the same final windows → subject/report.json      │
│                                                                             │
│   PHASE 3: EVALUATION REPORT GENERATION                                     │
│   ──────────────────────────────────────                                    │
│   pair raw reports → recompute metrics → apply report-local policy          │
│                    → evaluation.report.json + runtime.manifest.json          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Guard Chain Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GUARD CHAIN EXECUTION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   CANONICAL ORDER: invariants → spectral → rmt → variance → invariants      │
│                                                                             │
│   prepare model + all guards                                                │
│          │                                                                  │
│          ▼                                                                  │
│   invariants(pre).validate → edit/noop stage                                │
│          │                                                                  │
│          ▼                                                                  │
│   spectral.validate → rmt.validate → variance.validate                      │
│          │                                                                  │
│          ▼                                                                  │
│   invariants(post).validate → evaluate → finalize run report                │
│                                                                             │
│   Report evidence includes guard statuses plus spectral and RMT             │
│   measurement-contract hashes.                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Report Generation Flow

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         REPORT GENERATION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INPUTS                                                                    │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│   │  Baseline   │  │   Subject   │  │   Policy    │  │   Profile   │        │
│   │   report    │  │   report    │  │ (tiers.yaml)│  │ (ci/release)│        │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│          └────────────────┴────────────────┴────────────────┘               │
│                                    │                                        │
│                                    ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         REPORT BUILDER                              │   │
│   │  1. Pair baseline/subject windows                                   │   │
│   │  2. Compute paired ΔlogNLL + BCa bootstrap                          │   │
│   │  3. Apply policy gates (PM ratio, drift, guard checks)              │   │
│   │  4. Emit validation flags + state                                   │   │
│   │  5. Attach provenance and mark verifier assurance pending           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│   BUNDLE OUTPUTS                                                            │
│   evaluation.report.json + runtime.manifest.json + optional Markdown       │
│                                    │                                        │
│                                    ▼                                        │
│   invarlock report html → evaluation.html (rebuildable review rendering)    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Architecture Guardrails

The shell/core split is enforced by design and by targeted architecture guard
tests. The intended invariants are:

- Package roots such as `adapters/__init__.py` and `guards/__init__.py` expose
  explicit canonical exports.
- RMT ownership lives in `rmt.py`, `rmt_analysis.py`, and `rmt_detection.py`.
- Public command shells stay thin; orchestration and typed dependencies live in
  their canonical owner modules.
- Canonical owner contracts use the current configuration objects. For example,
  lens-metric calculation takes a required `MetricsConfig`.
- Modules under `src/invarlock/core/` and `src/invarlock/reporting/` stay callable
  independently of `invarlock.cli`.

These guardrails keep the CLI as an imperative shell while policy, contracts,
and verdict computation remain reusable from non-CLI flows such as evidence-pack
verification and programmatic execution.

## Key Design Decisions

| Decision | Rationale | Implementation |
| --- | --- | --- |
| **Torch-independent core** | `runner.py` coordinates without importing torch; adapters encapsulate torch-specific logic. | Adapter protocol in `core/api.py` |
| **Edit-stack-neutral guards** | Guards work with subject checkpoints from quantization, pruning, LoRA merge, fine-tuning, or other weight-edit workflows. | Guard protocol validates model state, not edit toolchain |
| **Tier-based policies** | Explicit packaged defaults in `runtime/tiers.yaml`; run-local sweeps may recommend reviewed overrides. | Policy resolution in `guards/policies.py` |
| **Deterministic evaluation inputs** | Seed bundles and window schedules remove sampling drift; numerical results may still vary across devices, kernels, or dependency versions. | `meta.seeds`, `dataset.windows.stats` tracking |
| **Functional-core / imperative-shell split** | Keep policy, artifact contracts, and verdict computation reusable outside the CLI while CLI modules stay thin. | `core/*.py` + `reporting/*.py` owners called from `cli/commands/*.py` |
| **Single verifier ownership** | Runtime-manifest verification should not vary with host tooling, so it must use one product implementation. | `runtime_verify.py`, `runtime_provenance.py` |
| **Plugin architecture** | Entry points for guards, adapters, edits enable extension without core changes. | `importlib.metadata` discovery in `core/registry.py` |
| **Log-space primary metrics** | Paired ΔlogNLL with BCa bootstrap avoids ratio math bias. | `core/bootstrap.py` implementation |

Edit-stack neutral means the stable production boundary is BYOE: an external
quantization tool, pruner, adapter merge, or fine-tuning pipeline produces the
subject checkpoint, and InvarLock validates the resulting
baseline-vs-subject evidence.
The evaluator CLI keeps built-in editing to portable demo/smoke support. The
evidence-pack harness derives non-calibration active lanes from a scenario manifest; the
current validation-subject lanes cover magnitude pruning and synthetic
low-rank/dense perturbations, while compatible CUDA environments can opt into
packed bitsandbytes deployable quantization scenarios. Validation-subject lanes
represent deterministic perturbations rather than adapter training, adapter
merging, or model fine-tuning.

## Module Dependencies

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODULE DEPENDENCY GRAPH                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   cli/commands/*                                                            │
│          │                                                                  │
│          ▼                                                                  │
│   cli/run_config.py · run_pairing.py · run_execution.py                     │
│   cli/run_metric_impact.py                                                  │
│          │                                                                  │
│          ▼                                                                  │
│   core + reporting contracts                                                │
│   evaluate_plan.py · run_policy.py · retry.py                               │
│   run_snapshot_contract.py · verify_contract.py                             │
│   reporting/run_report_contract.py                                          │
│          │                                                                  │
│          ├──▶ core/runner.py ──▶ adapters/* · edits/* · guards/* · eval/*   │
│          │                                                                  │
│          └──▶ reporting/report_make.py ──▶ report_bundle.py · rendering/*   │
│                                                                             │
│   KEY: ───▶ imports/depends on                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Extension Points

InvarLock supports extension via entry points without modifying core code.

| Extension Type | Entry Point Group | Example |
| --- | --- | --- |
| Adapters | `invarlock.adapters` | `hf_causal`, `hf_mlm`, `hf_seq2seq`, `hf_multimodal`, `hf_auto`, `hf_bnb`, `hf_awq`, `hf_gptq`, `hf_torchao`, `hf_hqq`, `hf_quanto`, `hf_ct` |
| Guards | `invarlock.guards` | `invariants`, `spectral`, `rmt`, `variance` |
| Edits | `invarlock.edits` | `quant_rtn` (`noop` is built in for internal/catalog use, not a pyproject entry-point example) |

### Custom Adapter Example

```python
# my_adapter.py
from invarlock.core.api import ModelAdapter

class MyAdapter(ModelAdapter):
    name = "my_custom_adapter"

    def load(self, model_id: str, device: str) -> nn.Module:
        # Custom loading logic
        ...

    def describe(self, model: nn.Module) -> dict:
        # Return model metadata
        ...
```

```toml
# pyproject.toml
[project.entry-points."invarlock.adapters"]
my_custom_adapter = "my_adapter:MyAdapter"
```

## Troubleshooting

- **Import errors in torch-free context**: ensure `invarlock.core` imports stay
  torch-independent; use adapters for torch operations.
- **Guard preparation failures**: check tier policy compatibility; use
  `context.run.strict_guard_prepare: false` for debugging.
- **report generation errors**: verify baseline and subject reports exist
  and have compatible window structures.

## Observability

- Pipeline phases emit timing via `print_timing_summary()` in CLI.
- Guard results recorded in `report.guards[]` and report `validation.*` flags.
- Telemetry fields include `memory_mb_peak`, `latency_ms_*`, `duration_s`.

## Related Documentation

- [CLI Reference](cli.md) — Command usage and options
- [Guards Reference](guards.md) — Guard configuration and evidence
- [Configuration Schema](config-schema.md) — YAML config structure
- [reports](reports.md) — report schema and verification
- [Assurance Case Overview](../assurance/00-assurance-case.md) — Assurance claims and evidence
