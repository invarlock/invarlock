# Reference Documentation

This section provides detailed technical reference documentation for InvarLock's
configuration, CLI, APIs, guards, and supporting infrastructure.

## Quick Navigation

| Document | Purpose | Audience |
| --- | --- | --- |
| [CLI Reference](cli.md) | Command-line interface and options | All users |
| [Configuration Schema](config-schema.md) | YAML config structure and precedence | CLI users |
| [Guards](guards.md) | Safety check configuration and evidence | Users tuning guards |
| [Model Adapters](model-adapters.md) | Adapter selection and capabilities | CLI and API users |
| [Datasets](datasets.md) | Dataset providers and pairing | CLI users |
| [Certificates](certificates.md) | v1 schema, telemetry, and HTML export | Operators, tool authors |
| [Tier Policy Catalog](tier-policy-catalog.md) | Guard threshold explanations | Operators auditing policies |
| [Environment Variables](env-vars.md) | Runtime toggles and flags | Operators |
| [API Guide](api-guide.md) | Programmatic interface | Python developers |
| [Observability](observability.md) | Monitoring and telemetry | Operators |
| [Calibration](calibration.md) | Guard threshold calibration | Guard maintainers |

## By Task

### Running Certifications

1. [CLI Reference](cli.md) — `certify`, `verify`, `run` commands
2. [Configuration Schema](config-schema.md) — YAML presets and profiles
3. [Datasets](datasets.md) — Provider configuration
4. [Model Adapters](model-adapters.md) — Adapter selection

### Understanding Certificates

1. [Certificates](certificates.md) — v1 schema, telemetry, and HTML export
2. [Artifact Layout](artifacts.md) — File organization

### Tuning Guards

1. [Guards](guards.md) — Configuration and evidence
2. [Tier Policy Catalog](tier-policy-catalog.md) — Threshold rationale
3. [GPU/MPS-First Guards](../assurance/13-gpu-mps-first-guards.md) — Design decisions
4. [Calibration](calibration.md) — Threshold calibration

### Programming Against InvarLock

1. [API Guide](api-guide.md) — `CoreRunner.execute` and helpers
2. [Programmatic Quickstart](programmatic-quickstart.md) — Minimal examples
3. [Observability](observability.md) — Monitoring infrastructure

### Operations

1. [Environment Variables](env-vars.md) — Runtime configuration
2. [Artifact Layout](artifacts.md) — Evidence retention
3. [Observability](observability.md) — Health checking and telemetry

## Related Documentation

- [User Guide](../user-guide/getting-started.md) — Task-oriented workflows
- [Assurance](../assurance/00-safety-case.md) — Safety claims and proofs
- [Security](../security/pip-audit-allowlist.md) — Security policies
