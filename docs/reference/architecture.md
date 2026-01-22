# System Architecture

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Edit-agnostic safety certification framework for ML model weight modifications. |
| **Audience** | Developers extending InvarLock, operators debugging pipelines, security reviewers. |
| **Core components** | CLI layer, Core runtime, Guard chain, Reporting/certificate subsystem. |
| **Design goals** | Torch-independent core, edit-agnostic guards, deterministic evaluation, full provenance. |
| **Source of truth** | `src/invarlock/core/runner.py`, `src/invarlock/cli/commands/*.py`, `src/invarlock/guards/*.py`. |

See the [Glossary](../assurance/glossary.md) for definitions of terms such as
four-guard pipeline, policy digest, and measurement contract.

## Contents

1. [Quick Reference](#quick-reference)
2. [High-Level Architecture](#high-level-architecture)
3. [Component Layers](#component-layers)
4. [Pipeline Flow](#pipeline-flow)
5. [Guard Chain Architecture](#guard-chain-architecture)
6. [Certificate Generation Flow](#certificate-generation-flow)
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
│  USER INPUT                    PROCESSING                      OUTPUT      │
│  ─────────                     ──────────                      ──────      │
│                                                                             │
│  ┌─────────┐     ┌────────────────────────────────┐     ┌──────────────┐   │
│  │ Config  │────▶│         CLI LAYER              │────▶│ Certificate  │   │
│  │ (YAML)  │     │ certify │ run │ verify │ ...   │     │   (JSON)     │   │
│  └─────────┘     └───────────────┬────────────────┘     └──────────────┘   │
│                                  │                                          │
│  ┌─────────┐     ┌───────────────▼────────────────┐     ┌──────────────┐   │
│  │ Model   │────▶│       CORE RUNTIME             │────▶│   Report     │   │
│  │ (HF ID) │     │ runner.py + adapters + edits   │     │   (JSON)     │   │
│  └─────────┘     └───────────────┬────────────────┘     └──────────────┘   │
│                                  │                                          │
│  ┌─────────┐     ┌───────────────▼────────────────┐     ┌──────────────┐   │
│  │ Dataset │────▶│       GUARD CHAIN              │────▶│   Events     │   │
│  │(provider)│     │ invariants→spectral→rmt→var   │     │  (JSONL)     │   │
│  └─────────┘     └────────────────────────────────┘     └──────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## High-Level Architecture

InvarLock follows a layered architecture with clear separation of concerns:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLI LAYER                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ certify  │ │   run    │ │  verify  │ │  report  │ │  doctor  │          │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘          │
│       │            │            │            │            │                 │
├───────┴────────────┴────────────┴────────────┴────────────┴─────────────────┤
│                            CORE RUNTIME                                     │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │                        runner.py                                  │      │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐     │      │
│  │  │prepare │─▶│ guards │─▶│  edit  │─▶│ guards │─▶│  eval  │     │      │
│  │  │ model  │  │(before)│  │ apply  │  │(after) │  │ final  │     │      │
│  │  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘     │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                            GUARD LAYER                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ invariants │  │  spectral  │  │    rmt     │  │  variance  │            │
│  │ (integrity)│  │  (weights) │  │(activation)│  │   (A/B)    │            │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘            │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                          REPORTING LAYER                                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │   report   │  │certificate │  │   render   │  │  manifest  │            │
│  │   (JSON)   │  │   (JSON)   │  │   (MD/HTML)│  │   (JSON)   │            │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Layers

### CLI Layer (`src/invarlock/cli/`)

Typer-based command-line interface providing user-facing entry points.

| Command | Purpose | Primary Output |
| --- | --- | --- |
| `certify` | Compare baseline vs subject with pinned windows | Certificate JSON + MD |
| `run` | Single-model evaluation pipeline | Report JSON + Events JSONL |
| `verify` | Validate certificate against schema and pairing | Exit code + messages |
| `report` | Render/compare reports and certificates | MD/HTML/JSON artifacts |
| `doctor` | Environment diagnostics | Health check output |
| `plugins` | List adapters, guards, edits | Plugin inventory |

### Core Runtime (`src/invarlock/core/`)

Pipeline orchestration without direct torch imports (torch-independent coordination).

| Module | Responsibility |
| --- | --- |
| `runner.py` | Pipeline phases: prepare → guards → edit → eval → finalize |
| `api.py` | Protocol definitions for ModelAdapter, ModelEdit, Guard |
| `bootstrap.py` | BCa bootstrap CI computation for paired metrics |
| `checkpoint.py` | Snapshot/restore for retry loops |
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

Certificate generation, validation, and rendering.

| Module | Responsibility |
| --- | --- |
| `certificate.py` | Certificate schema and validation |
| `render.py` | Markdown certificate rendering |
| `html.py` | HTML export with styling |
| `report.py` | Report generation and manifest |
| `telemetry.py` | Performance metrics collection |

## Pipeline Flow

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CERTIFICATION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   PHASE 1: BASELINE RUN                                                     │
│   ─────────────────────                                                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │  Load    │───▶│ Evaluate │───▶│  Record  │───▶│  Save    │             │
│   │  Model   │    │  Windows │    │  Guards  │    │  Report  │             │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘             │
│                                                                             │
│   PHASE 2: SUBJECT RUN (with baseline window pinning)                       │
│   ───────────────────────────────────────────────                           │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │  Load    │───▶│  Apply   │───▶│ Evaluate │───▶│  Record  │             │
│   │  Model   │    │  Edit    │    │  Paired  │    │  Guards  │             │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘             │
│                                                                             │
│   PHASE 3: CERTIFICATE GENERATION                                           │
│   ───────────────────────────────                                           │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │  Pair    │───▶│ Compute  │───▶│  Apply   │───▶│  Render  │             │
│   │  Windows │    │  Ratios  │    │  Gates   │    │  Cert    │             │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Guard Chain Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GUARD CHAIN EXECUTION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   CANONICAL ORDER: invariants → spectral → rmt → variance → invariants     │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                    BEFORE EDIT                                   │      │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │      │
│   │  │  INVARIANTS  │  │   SPECTRAL   │  │     RMT      │           │      │
│   │  │   prepare()  │  │   prepare()  │  │   prepare()  │           │      │
│   │  │  ──────────  │  │  ──────────  │  │  ──────────  │           │      │
│   │  │ • NaN check  │  │ • Baseline σ │  │ • Baseline ε │           │      │
│   │  │ • Shape check│  │ • Family caps│  │ • Activation │           │      │
│   │  │ • Tying check│  │ • z-scores   │  │ • Calibration│           │      │
│   │  └──────────────┘  └──────────────┘  └──────────────┘           │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                               │                                             │
│                               ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                      EDIT APPLIED                                │      │
│   │          (quant_rtn, noop, or external BYOE checkpoint)          │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                               │                                             │
│                               ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                     AFTER EDIT                                   │      │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │      │
│   │  │  INVARIANTS  │  │   SPECTRAL   │  │     RMT      │           │      │
│   │  │  validate()  │  │  validate()  │  │  validate()  │           │      │
│   │  │  ──────────  │  │  ──────────  │  │  ──────────  │           │      │
│   │  │ • Post-edit  │  │ • κ-check    │  │ • ε-band     │           │      │
│   │  │   integrity  │  │ • Caps count │  │   compliance │           │      │
│   │  │ • NaN detect │  │ • Stability  │  │ • Δ tracking │           │      │
│   │  └──────────────┘  └──────────────┘  └──────────────┘           │      │
│   │                                                                  │      │
│   │  ┌──────────────┐                                                │      │
│   │  │   VARIANCE   │  (A/B test: bare vs VE-enabled)               │      │
│   │  │  validate()  │                                                │      │
│   │  │  ──────────  │                                                │      │
│   │  │ • Gain check │                                                │      │
│   │  │ • CI overlap │                                                │      │
│   │  │ • Enable/skip│                                                │      │
│   │  └──────────────┘                                                │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                               │                                             │
│                               ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                    GUARD RESULTS                                 │      │
│   │                                                                  │      │
│   │  • validation.invariants_pass: bool                             │      │
│   │  • validation.spectral_stable: bool                             │      │
│   │  • validation.rmt_stable: bool                                  │      │
│   │  • measurement_contract_hash: str (CI/Release verification)     │      │
│   │                                                                  │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Certificate Generation Flow

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CERTIFICATE GENERATION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INPUTS                                                                    │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│   │  Baseline   │  │   Subject   │  │   Policy    │  │   Profile   │       │
│   │   Report    │  │   Report    │  │(tiers.yaml) │  │ (ci/release)│       │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │
│          │                │                │                │               │
│          └────────────────┴────────────────┴────────────────┘               │
│                                    │                                        │
│                                    ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                    CERTIFICATE BUILDER                           │      │
│   │                                                                  │      │
│   │  1. Pair evaluation windows (baseline ↔ subject)                │      │
│   │  2. Compute log-space ΔlogNLL with BCa bootstrap                │      │
│   │  3. Apply tier policy gates (PM ratio, drift, guards)           │      │
│   │  4. Generate validation flags and overall status                │      │
│   │  5. Attach provenance (seeds, hashes, contracts)                │      │
│   │                                                                  │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                    │                                        │
│                                    ▼                                        │
│   OUTPUTS                                                                   │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│   │ evaluation  │  │ evaluation  │  │ evaluation  │  │  manifest   │       │
│   │ .cert.json  │  │_cert.md     │  │_cert.html   │  │   .json     │       │
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │
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
│              ┌───────────────────┼───────────────────┐                      │
│              │                   │                   │                      │
│              ▼                   ▼                   ▼                      │
│       ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
│       │    core/    │    │   guards/   │    │ reporting/  │                │
│       │  runner.py  │───▶│  *.py       │───▶│ *.py        │                │
│       └──────┬──────┘    └──────┬──────┘    └─────────────┘                │
│              │                  │                                           │
│              ▼                  ▼                                           │
│       ┌─────────────┐    ┌─────────────┐                                   │
│       │  adapters/  │    │   edits/    │                                   │
│       │   hf_*.py   │    │ quant_rtn.py│                                   │
│       └──────┬──────┘    └─────────────┘                                   │
│              │                                                              │
│              ▼                                                              │
│       ┌─────────────┐                                                      │
│       │    eval/    │  (metrics, datasets, tasks)                          │
│       │  *.py       │                                                      │
│       └─────────────┘                                                      │
│                                                                             │
│   KEY: ───▶ imports/depends on                                             │
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
  `INVARLOCK_GUARD_PREPARE_STRICT=0` for debugging.
- **Certificate generation errors**: verify baseline and subject reports exist
  and have compatible window structures.

## Observability

- Pipeline phases emit timing via `print_timing_summary()` in CLI.
- Guard results recorded in `report.guards[]` and certificate `validation.*` flags.
- Telemetry fields include `memory_mb_peak`, `latency_ms_*`, `duration_s`.

## Related Documentation

- [CLI Reference](cli.md) — Command usage and options
- [Guards Reference](guards.md) — Guard configuration and evidence
- [Configuration Schema](config-schema.md) — YAML config structure
- [Certificates](certificates.md) — Certificate schema and verification
- [Safety Case Overview](../assurance/00-safety-case.md) — Assurance claims and evidence
