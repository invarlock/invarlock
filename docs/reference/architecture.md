# System Architecture

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Edit-agnostic safety evaluation framework for ML model weight modifications. |
| **Audience** | Developers extending InvarLock, operators debugging pipelines, security reviewers. |
| **Core components** | CLI shells, Core/runtime policy layer, Guard chain, Reporting/artifact subsystem. |
| **Design goals** | Torch-independent core, edit-agnostic guards, deterministic evaluation, explicit artifact contracts, full provenance. |
| **Source of truth** | `src/invarlock/core/*.py`, `src/invarlock/reporting/*.py`, `src/invarlock/cli/commands/*.py`, `src/invarlock/guards/*.py`. |

See the [Glossary](../assurance/glossary.md) for definitions of terms such as
the canonical guard chain, policy digest, and measurement contract.

## Contents

1. [Quick Reference](#quick-reference)
2. [High-Level Architecture](#high-level-architecture)
3. [Component Layers](#component-layers)
4. [Pipeline Flow](#pipeline-flow)
5. [Guard Chain Architecture](#guard-chain-architecture)
6. [report Generation Flow](#report-generation-flow)
7. [Key Design Decisions](#key-design-decisions)
8. [Module Dependencies](#module-dependencies)
9. [Extension Points](#extension-points)
10. [Related Documentation](#related-documentation)

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
│  │  (YAML)  │    │   evaluate | run | verify ...  │    │    (JSON)    │     │
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
│  │ evaluate │ │   run    │ │  verify  │ │  report  │ │  doctor  │           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │
│       │            │            │            │            │                 │
├───────┴────────────┴────────────┴────────────┴────────────┴─────────────────┤
│                     CORE POLICY / CONTRACT LAYER                            │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │ evaluate_plan · report_inputs · doctor_findings                │        │
│  │ verify_contract · run_*_policy                                 │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                       CORE RUNTIME / SERVICES                               │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │ runner.py + runner_* + config_execution.py                     │        │
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
│  │ report_build │ │ report_files │ │   render   │ │  manifest  │            │
│  │ + policy     │ │ + evidence   │ │  (MD/HTML) │ │   (JSON)   │            │
│  └──────────────┘ └──────────────┘ └────────────┘ └────────────┘            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Layers

### CLI Layer (`src/invarlock/cli/`)

Typer-based command shells providing user-facing entry points. The command
modules should stay thin: parse arguments, call core/reporting owners, render
output, and map failures to exit codes.

| Command | Purpose | Primary Output |
| --- | --- | --- |
| `evaluate` | Compare baseline vs subject with pinned windows | report JSON + MD |
| `run` | Single-model evaluation pipeline | Report JSON + Events JSONL |
| `verify` | Validate report against schema and pairing | Exit code + messages |
| `report` | Render/compare reports and reports | MD/HTML/JSON artifacts |
| `doctor` | Environment diagnostics | Health check output |
| `plugins` | List adapters, guards, edits | Plugin inventory |

### Core Policy / Contracts (`src/invarlock/core/`, `src/invarlock/reporting/`)

Deterministic policy, artifact-contract, and report-verification owners shared
by the CLI and non-CLI entrypoints.

| Module | Responsibility |
| --- | --- |
| `evaluate_plan.py` | Evaluation result policy, degradation classification, and emitted outcome shaping |
| `report_inputs.py` | Canonical report path resolution and JSON-object validation |
| `doctor_findings.py` | Structured doctor findings and optional report cross-check analysis |
| `verify_contract.py` | Structured report-verification service used by `verify` and proof-pack flows |
| `run_*_policy.py` | Snapshot, retry, and timing-policy helpers injected into `run` |

### Core Runtime (`src/invarlock/core/`)

Pipeline orchestration without direct torch imports (torch-independent coordination).

| Module | Responsibility |
| --- | --- |
| `runner.py` + `runner_*.py` | Pipeline phases: prepare → guards → edit → eval → finalize |
| `api.py` | Protocol definitions for ModelAdapter, ModelEdit, Guard |
| `bootstrap.py` | BCa bootstrap CI computation for paired metrics |
| `checkpoint.py` | Snapshot/restore primitives for retry loops |
| `config_execution.py` | Explicit run-from-config execution contract |
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
| `report_builder.py` | Evaluation report assembly from paired baseline/subject runs |
| `report_validation.py` | Schema and semantic validation |
| `render.py` | Markdown report rendering |
| `html.py` | HTML export with styling |
| `report_files.py` | Report/manifest persistence and artifact writing |
| `evidence.py` | Evidence file normalization and attachment helpers |
| `telemetry.py` | Performance metrics collection |

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

## Key Design Decisions

| Decision | Rationale | Implementation |
| --- | --- | --- |
| **Torch-independent core** | `runner.py` coordinates without importing torch; adapters encapsulate torch-specific logic. | Adapter protocol in `core/api.py` |
| **Edit-agnostic guards** | Guards work with any weight modification (quantization, pruning, LoRA merge). | Guard protocol validates model state, not edit type |
| **Tier-based policies** | Calibrated thresholds in `tiers.yaml` for balanced/conservative/aggressive safety profiles. | Policy resolution in `guards/policies.py` |
| **Deterministic evaluation** | Seed bundle + window pairing schedules ensure reproducible metrics. | `meta.seeds`, `dataset.windows.stats` tracking |
| **Functional-core / imperative-shell split** | Keep policy, artifact contracts, and verdict computation reusable outside the CLI while CLI modules stay thin. | `core/*.py` + `reporting/*.py` owners called from `cli/commands/*.py` |
| **Plugin architecture** | Entry points for guards, adapters, edits enable extension without core changes. | `importlib.metadata` discovery in `core/registry.py` |
| **Log-space primary metrics** | Paired ΔlogNLL with BCa bootstrap avoids ratio math bias. | `core/bootstrap.py` implementation |

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
│                     ┌───────────────────────────┐                            │
│                     │ core/reporting contracts  │                            │
│                     │ evaluate_plan,            │                            │
│                     │ report_inputs,            │                            │
│                     │ doctor_findings,          │                            │
│                     │ verify_contract, run_*    │                            │
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
