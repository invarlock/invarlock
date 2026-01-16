# InvarLock Architecture

## Overview

InvarLock is an edit-agnostic safety certification framework for ML model weight edits.

## System Architecture

### High-Level Flow

```text
CLI -> Core Runtime -> Guards -> Reporting
```

### Component Layers

1. **CLI Layer** (Typer-based)
   - `certify`, `run`, `verify`, `doctor`, `calibrate`, `report`, `export html`, `plugins`, `explain gates`
2. **Core Runtime** (`runner.py`)
   - Pipeline orchestration: prepare -> edit -> guards -> eval -> finalize
3. **Guards Layer**
   - Four-guard pipeline: invariants -> spectral -> RMT -> variance
4. **Reporting Layer**
   - Certificate generation, validation, rendering

## Key Design Decisions

1. Torch-independent core - `runner.py` coordinates without direct torch imports
2. Edit-agnostic design - guards work with any weight modification
3. Tier-based policies - calibrated thresholds in `tiers.yaml`
4. Deterministic evaluation - seed bundle + pairing schedules
5. Plugin architecture - entry points for guards, adapters, edits

## Data Flow

```mermaid
graph TD
    CLI[CLI Commands] --> CORE[CoreRunner]
    CORE --> EDIT[Edit Application]
    CORE --> GUARDS[Guard Chain]
    CORE --> EVAL[Evaluation]
    EVAL --> REPORT[Report + Certificate]
    GUARDS --> REPORT
```

```mermaid
sequenceDiagram
    participant CLI as CLI
    participant Core as CoreRunner
    participant Guard as Guard Chain
    participant Report as Reporting
    CLI->>Core: load config + model
    Core->>Guard: prepare + validate
    Core->>Core: apply edit
    Core->>Guard: before/after edit checks
    Core->>Report: emit report + evidence
```

```mermaid
graph LR
    Report[report.json] --> Cert[certificate.json]
    Cert --> Render[markdown/html renderer]
    Render --> Artifacts[cert md/html + manifest]
```

## Module Dependencies

```mermaid
graph TD
    CLI[cli/commands] --> Core[core/runner]
    Core --> Guards[guards/*]
    Core --> Eval[eval/*]
    Core --> Reporting[reporting/*]
    Reporting --> Cert[reporting/certificate]
```
