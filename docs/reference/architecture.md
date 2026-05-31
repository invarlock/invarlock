# System Architecture

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Auditable release-gate framework for ML model weight modifications. |
| **Audience** | Developers extending InvarLock, operators debugging pipelines, security reviewers. |
| **Core components** | CLI shells, Core/runtime policy layer, Guard chain, Reporting/artifact subsystem. |
| **Design goals** | Torch-independent core, edit-stack-neutral guards, deterministic evaluation, explicit artifact contracts, full provenance. |
| **Source of truth** | `src/invarlock/core/*.py`, `src/invarlock/reporting/*.py`, `src/invarlock/runtime_provenance.py`, `src/invarlock/runtime_verify.py`, `src/invarlock/cli/commands/*.py`, `src/invarlock/cli/run_*.py`, `src/invarlock/guards/*.py`. |

See the [Glossary](../assurance/glossary.md) for definitions of terms such as
the canonical guard chain, policy digest, and measurement contract.

## Contents

1. [Quick Reference](#quick-reference)
2. [High-Level Architecture](#high-level-architecture)
3. [Component Layers](#component-layers)
4. [Pipeline Flow](#pipeline-flow)
5. [Guard Chain Architecture](#guard-chain-architecture)
6. [report Generation Flow](#report-generation-flow)
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
│  USER INPUT                    PROCESSING                      OUTPUT       │
│  ─────────                     ──────────                      ──────       │
│                                                                             │
│  ┌──────────┐    ┌────────────────────────────────┐    ┌──────────────┐     │
│  │  Config  │───▶│            CLI LAYER           │───▶│    report    │     │
│  │  (YAML)  │    │ evaluate | verify | report ... │    │    (JSON)    │     │
│  └──────────┘    └───────────────┬────────────────┘    └──────────────┘     │
│                                  │                                          │
│  ┌──────────┐    ┌───────────────▼────────────────┐    ┌──────────────┐     │
│  │  Model   │───▶│          CORE RUNTIME          │───▶│    report    │     │
│  │  (HF ID) │    │   runner.py + adapters + edits │    │    (JSON)    │     │
│  └──────────┘    └───────────────┬────────────────┘    └──────────────┘     │
│                                  │                                          │
│  ┌──────────┐    ┌───────────────▼────────────────┐    ┌──────────────┐     │
│  │ Dataset  │───▶│          GUARD CHAIN           │───▶│    Events    │     │
│  │(provider)│    │ inv(pre)→spectral→rmt→var→post │    │    (JSONL)   │     │
│  └──────────┘    └────────────────────────────────┘    └──────────────┘     │
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
│  │  │prepare │─▶│ guards │─▶│  edit  │─▶│ guards │─▶│  eval  │     │        │
│  │  │ model  │  │(before)│  │ apply  │  │(after) │  │ final  │     │        │
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
│  │ report_make  │ │report_bundle │ │   render   │ │  schema    │            │
│  │ + console    │ │ + evidence   │ │  (MD/HTML) │ │   (JSON)   │            │
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
`cli/run_config.py`, `cli/run_pairing.py`, and `cli/run_overhead.py` belong to
this boundary layer as well. They can perform
CLI-facing adaptation and console/event rendering, but they must not become
policy owners.

| Command | Purpose | Primary Output |
| --- | --- | --- |
| `evaluate` | Compare baseline vs subject with pinned windows | report JSON + MD |
| `verify` | Validate report against schema and pairing | Exit code + messages |
| `report` | Render/compare reports and reports | MD/HTML/JSON artifacts |
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
| `runtime_verify.py` + `runtime_provenance.py` | Authoritative runtime-manifest verification and runtime-provenance ownership for report verification |
| `run_policy.py` | Shared run policy helpers such as split choice, PM thresholds, and overhead policy |
| `retry.py` | Retry controller, edit-parameter adjustment, attempt summaries, and retry state transitions |
| `run_snapshot_contract.py` | Snapshot planning, restore behavior, and retry transitions |
| `run_report_contract.py` | Run provenance finalization, payload shaping, and run-report assembly contracts |

### Runtime Provenance Verification Ownership

Runtime provenance uses a single verifier implementation:

- `runtime_verify.py` is the authoritative programmatic verifier for
  `runtime.manifest.json` plus report-digest binding checks.
- `cli/commands/verify.py` owns the CLI entrypoints for both report verification
  and advanced runtime-manifest verification.
- `runtime_provenance.py` calls the same verifier when `invarlock verify`
  enforces runtime provenance on container-backed reports.
- Product behavior does not depend on finding an external verifier binary on
  `PATH`; verifier semantics are package-native and deterministic across
  installs.

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
| `report_validation.py` | Canonical validation-flag computation |
| `report_make.py` | Evaluation-report input normalization, build-section extraction, output shaping, and public report assembly |
| `report_make_assembly.py` | Policy/provenance/guard assembly and report build-context composition |
| `report_bundle.py` | Evaluation-bundle persistence, manifest writing, and evidence attachment |
| `report_contract.py` | Input loading and report-generation planning |
| `report_overhead.py` | Guard-overhead normalization, summary building, and report shaping |
| `report_summary.py` | Console validation blocks and shared executive-summary/view-model derivation for reporting surfaces |
| `render.py` | Markdown rendering for evaluation reports |
| `html.py` | HTML export with styling |
| `evidence.py` | Evidence file normalization and attachment helpers |
| `report_builder_support.py` | Report build context, telemetry extraction, telemetry payloads, artifacts, and baseline references |

## Pipeline Flow

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EVALUATION PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   PHASE 1: BASELINE RUN                                                     │
│   ─────────────────────                                                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│   │  Load    │───▶│ Evaluate │───▶│  Record  │───▶│  Save    │              │
│   │  Model   │    │  Windows │    │  Guards  │    │  Report  │              │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│                                                                             │
│   PHASE 2: SUBJECT RUN (with baseline window pinning)                       │
│   ───────────────────────────────────────────────                           │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│   │  Load    │───▶│  Apply   │───▶│ Evaluate │───▶│  Record  │              │
│   │  Model   │    │  Edit    │    │  Paired  │    │  Guards  │              │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│                                                                             │
│   PHASE 3: EVALUATION REPORT GENERATION                                     │
│   ──────────────────────────────────────                                    │
│   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│   │ Normalize  │─▶│  Compare   │─▶│  Apply     │─▶│ Persist +  │            │
│   │ inputs     │  │  metrics   │  │  policy    │  │ render     │            │
│   └────────────┘  └────────────┘  └────────────┘  └────────────┘            │
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
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                    BEFORE EDIT                                  │       │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │       │
│   │  │  INVARIANTS  │  │   SPECTRAL   │  │     RMT      │           │       │
│   │  │   prepare()  │  │   prepare()  │  │   prepare()  │           │       │
│   │  │  ──────────  │  │  ──────────  │  │  ──────────  │           │       │
│   │  │ • NaN check  │  │ • Baseline σ │  │ • Baseline ε │           │       │
│   │  │ • Shape check│  │ • Family caps│  │ • Activation │           │       │
│   │  │ • Tying check│  │ • z-scores   │  │ • Calibration│           │       │
│   │  └──────────────┘  └──────────────┘  └──────────────┘           │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                               │                                             │
│                               ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                      EDIT APPLIED                               │       │
│   │          (quant_rtn, noop, or external BYOE checkpoint)         │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                               │                                             │
│                               ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                     AFTER EDIT                                  │       │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │       │
│   │  │  INVARIANTS  │  │   SPECTRAL   │  │     RMT      │           │       │
│   │  │  validate()  │  │  validate()  │  │  validate()  │           │       │
│   │  │  ──────────  │  │  ──────────  │  │  ──────────  │           │       │
│   │  │ • Post-edit  │  │ • κ-check    │  │ • ε-band     │           │       │
│   │  │   integrity  │  │ • Caps count │  │   compliance │           │       │
│   │  │ • NaN detect │  │ • Stability  │  │ • Δ tracking │           │       │
│   │  └──────────────┘  └──────────────┘  └──────────────┘           │       │
│   │                                                                 │       │
│   │  ┌──────────────┐                                               │       │
│   │  │   VARIANCE   │  (A/B test: bare vs VE-enabled)               │       │
│   │  │  validate()  │                                               │       │
│   │  │  ──────────  │                                               │       │
│   │  │ • Gain check │                                               │       │
│   │  │ • CI overlap │                                               │       │
│   │  │ • Enable/skip│                                               │       │
│   │  └──────────────┘                                               │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                               │                                             │
│                               ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                    GUARD RESULTS                                │       │
│   │                                                                 │       │
│   │  • validation.invariants_pass: bool                             │       │
│   │  • validation.spectral_stable: bool                             │       │
│   │  • validation.rmt_stable: bool                                  │       │
│   │  • measurement_contract_hash: str (CI/Release verification)     │       │
│   │                                                                 │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Report Generation Flow

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         report GENERATION                                   │
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
│   │                         report BUILDER                              │   │
│   │  1. Pair baseline/subject windows                                   │   │
│   │  2. Compute paired ΔlogNLL + BCa bootstrap                          │   │
│   │  3. Apply policy gates (PM ratio, drift, guard checks)              │   │
│   │  4. Emit validation flags + state                                   │   │
│   │  5. Attach provenance (seeds)                                       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│   OUTPUTS                                                                   │
│   ┌────────────────────┐  ┌───────────────────┐  ┌────────────────────┐     │
│   │ evaluation.report  │  │ evaluation_report │  │  evaluation.html   │     │
│   │ .json              │  │ .md               │  │                    │     │
│   └────────────────────┘  └───────────────────┘  └────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Architecture Guardrails

The shell/core split is enforced by design and by targeted architecture guard
tests. The intended invariants are:

- No lazy exports in package roots such as `adapters/__init__.py` or
  `guards/__init__.py`. Package roots should expose only explicit canonical
  exports.
- No `rmt_legacy` references in production source. RMT ownership lives in
  `rmt.py`, `rmt_analysis.py`, and `rmt_detection.py`.
- No dependency-map orchestration in command shells. Public command owners must
  stay thin and must not rebuild giant `deps` dictionaries or inject callables
  to recreate removed indirection.
- No compatibility-only command signatures once a canonical owner contract
  exists. Example: lens-metric calculation takes a required `MetricsConfig`
  instead of deprecated per-call overrides.
- No CLI imports inside owner layers. Modules under `src/invarlock/core/` and
  `src/invarlock/reporting/` must stay callable without importing
  `invarlock.cli`.

These guardrails keep the CLI as an imperative shell while policy, contracts,
and verdict computation remain reusable from non-CLI flows such as evidence-pack
verification and programmatic execution.

## Key Design Decisions

| Decision | Rationale | Implementation |
| --- | --- | --- |
| **Torch-independent core** | `runner.py` coordinates without importing torch; adapters encapsulate torch-specific logic. | Adapter protocol in `core/api.py` |
| **Edit-stack-neutral guards** | Guards work with subject checkpoints from quantization, pruning, LoRA merge, fine-tuning, or other weight-edit workflows. | Guard protocol validates model state, not edit toolchain |
| **Tier-based policies** | Calibrated thresholds in `tiers.yaml` for balanced/conservative/aggressive safety profiles. | Policy resolution in `guards/policies.py` |
| **Deterministic evaluation** | Seed bundle + window pairing schedules ensure reproducible metrics. | `meta.seeds`, `dataset.windows.stats` tracking |
| **Functional-core / imperative-shell split** | Keep policy, artifact contracts, and verdict computation reusable outside the CLI while CLI modules stay thin. | `core/*.py` + `reporting/*.py` owners called from `cli/commands/*.py` |
| **Single verifier ownership** | Runtime-manifest verification should not vary with host tooling, so it must use one product implementation. | `runtime_verify.py`, `runtime_provenance.py` |
| **Plugin architecture** | Entry points for guards, adapters, edits enable extension without core changes. | `importlib.metadata` discovery in `core/registry.py` |
| **Log-space primary metrics** | Paired ΔlogNLL with BCa bootstrap avoids ratio math bias. | `core/bootstrap.py` implementation |

Edit-stack neutral does not mean every edit stack is bundled as an InvarLock
edit plugin. The stable production boundary is BYOE: an external quantization
tool, pruner, adapter merge, or fine-tuning pipeline produces the subject
checkpoint, and InvarLock validates the resulting baseline-vs-subject evidence.
Built-in edit generation is limited to demo/smoke support.

## Module Dependencies

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODULE DEPENDENCY GRAPH                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                           ┌─────────────┐                                   │
│                           │     CLI     │                                   │
│                           │  commands/* │                                   │
│                           └──────┬──────┘                                   │
│                                  │                                          │
│                                  ▼                                          │
│                   ┌──────────────────────────────┐                           │
│                   │ cli shell support modules    │                           │
│                   │ run_config/run_pairing/      │                           │
│                   │ run_execution/run_overhead   │                           │
│                   └─────────────┬────────────────┘                           │
│                                 │                                            │
│                                 ▼                                            │
│                     ┌───────────────────────────┐                            │
│                     │ core/reporting contracts  │                            │
│                     │ evaluate_plan,            │                            │
│                     │ report_inputs,            │                            │
│                     │ doctor_findings,          │                            │
│                     │ verify_contract,          │                            │
│                     │ run_policy, run_retry,    │                            │
│                     │ run_snapshot, run_report  │                            │
│                     └─────────────┬─────────────┘                            │
│                                   │                                          │
│              ┌────────────────────┼────────────────────┐                     │
│              │                    │                    │                     │
│              ▼                    ▼                    ▼                     │
│       ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                │
│       │ core/runner │────▶│  guards/*   │────▶│ reporting/* │                │
│       │  + services │     │             │     │ build/files │                │
│       └──────┬──────┘     └──────┬──────┘     └─────────────┘                │
│              │                   │                                           │
│              ▼                   ▼                                           │
│       ┌─────────────┐     ┌─────────────┐                                    │
│       │  adapters/  │     │   edits/    │                                    │
│       │   hf_*.py   │     │ quant_rtn.py│                                    │
│       └──────┬──────┘     └─────────────┘                                    │
│              │                                                               │
│              ▼                                                               │
│       ┌─────────────┐                                                        │
│       │    eval/    │  (metrics, datasets, tasks)                            │
│       │  *.py       │                                                        │
│       └─────────────┘                                                        │
│                                                                             │
│   KEY: ───▶ imports/depends on                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Extension Points

InvarLock supports extension via entry points without modifying core code.

| Extension Type | Entry Point Group | Example |
| --- | --- | --- |
| Adapters | `invarlock.adapters` | `hf_causal`, `hf_mlm`, `hf_causal` |
| Guards | `invarlock.guards` | `invariants`, `spectral`, `rmt`, `variance` |
| Edits | `invarlock.edits` | `quant_rtn`, `noop` |

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
