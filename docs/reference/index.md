# Reference Documentation

This section provides detailed technical reference documentation for InvarLock's
configuration, CLI, APIs, guards, and supporting infrastructure.

## Quick Navigation

| Document | Purpose | Audience |
| --- | --- | --- |
| [CLI Reference](cli.md) | Command-line interface and options | All users |
| [Tier Policy Tuning CLI (Calibration)](calibration.md) | Tier policy sweep harnesses | Operators recalibrating thresholds |
| [Configuration Schema](config-schema.md) | YAML config structure and precedence | CLI users |
| [Guards](guards.md) | Guard configuration and evidence | Users tuning guards |
| [Model Adapters](model-adapters.md) | Adapter selection and capabilities | CLI and API users |
| [Model Family Catalog](model-family-catalog.md) | Authoritative support inventory and backlog | Reviewers, tool authors |
| [Public Contracts](contracts.md) | Stable public contracts for reports, verification, and policy artifacts | Tool authors, reviewers |
| [Datasets](datasets.md) | Dataset providers and pairing | CLI users |
| [reports](reports.md) | v1 schema, telemetry, and HTML export | Operators, tool authors |
| [Architecture](architecture.md) | System layers, data flow, and dependencies | Builders, reviewers |
| [Tier Policy Catalog](tier-policy-catalog.md) | Guard threshold explanations | Operators auditing policies |
| [Environment Variables](env-vars.md) | Runtime toggles and flags | Operators |
| [API Guide](api-guide.md) | Programmatic interface | Python developers |
| [Observability](observability.md) | Monitoring and telemetry | Operators |

## By Task

### Running Evaluations

1. [CLI Reference](cli.md) — `evaluate`, `verify`, `report`, and `advanced`
2. [Configuration Schema](config-schema.md) — YAML presets and profiles
3. [Datasets](datasets.md) — Provider configuration
4. [Model Adapters](model-adapters.md) — Adapter selection
5. [Model Family Catalog](model-family-catalog.md) — Support inventory and backlog
6. [Public Contracts](contracts.md) — Machine-readable trust contracts

### Understanding reports

1. [reports](reports.md) — v1 schema, telemetry, and HTML export
2. [Artifact Layout](artifacts.md) — File organization

### Tuning Guards

1. [Guards](guards.md) — Configuration and evidence
2. [Tier Policy Catalog](tier-policy-catalog.md) — Threshold rationale
3. [Tier Policy Tuning CLI (Calibration)](calibration.md) — Sweep harnesses for recalibrating thresholds
4. [GPU/MPS-First Guard Measurement Contracts](../assurance/13-gpu-mps-first-guards.md) — Accelerator guard evidence contracts

### Programming Against InvarLock

1. [API Guide](api-guide.md) — Advanced/non-stable Python integration surface
2. [Programmatic Quickstart](programmatic-quickstart.md) — Minimal examples
3. [Architecture](architecture.md) — System layers and data flow
4. [Model Family Catalog](model-family-catalog.md) — Support inventory and backlog
5. [Public Contracts](contracts.md) — Evidence and policy contract surfaces
6. [Observability](observability.md) — Monitoring infrastructure

### Operations

1. [Environment Variables](env-vars.md) — Runtime configuration
2. [Artifact Layout](artifacts.md) — Evidence retention
3. [Observability](observability.md) — Health checking and telemetry
4. [CLI Reference](cli.md) — Advanced namespaces such as `evidence-pack`, `policy`, and `plugins`

## Related Documentation

- [User Guide](../user-guide/getting-started.md) — Task-oriented workflows
- [Assurance](../assurance/00-assurance-case.md) — Assurance rationale and derivations
- [Security](../security/pip-audit-allowlist.md) — Security policies
