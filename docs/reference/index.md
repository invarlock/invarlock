# Reference Documentation

This section provides detailed technical documentation for InvarLock's CLI,
configuration, APIs, and internal components.

## Quick Navigation

| If you want to... | Start here |
| --- | --- |
| Run InvarLock from terminal | [CLI Reference](cli.md) |
| Write a YAML config | [Configuration Schema](config-schema.md) |
| Use Python API directly | [API Guide](api-guide.md) |
| Understand certificates | [Certificate Schema](certificate-schema.md) |
| Configure guards | [Guards](guards.md) |
| Choose an adapter | [Model Adapters](model-adapters.md) |

## By Audience

### Operators (CLI Users)

- [CLI Reference](cli.md) — Commands, flags, and examples
- [Configuration Schema](config-schema.md) — YAML config structure
- [Environment Variables](env-vars.md) — Runtime toggles
- [Artifact Layout](artifacts.md) — Where outputs live
- [Dataset Providers](datasets.md) — Supported data sources

### Developers (Python API)

- [API Guide](api-guide.md) — CoreRunner and programmatic usage
- [Programmatic Quickstart](programmatic-quickstart.md) — Minimal Python example
- [Model Adapters](model-adapters.md) — Adapter interfaces and capabilities
- [Guards](guards.md) — Guard configuration and lifecycle

### Auditors (Certificate Consumers)

- [Certificate Schema](certificate-schema.md) — v1 certificate contract
- [Certificate Telemetry](certificate-telemetry.md) — Latency/memory fields
- [Exporting Certificates (HTML)](exporting-certificates-html.md) — Human-readable exports
- [Tier Policy Catalog](tier-policy-catalog.md) — Policy keys and rationale

### Contributors (Internals)

- [Calibration Reference](calibration.md) — Recalibrating guard thresholds
- [GPU/MPS-First Guards](gpu-mps-first-guards.md) — Guard design decisions
- [Tier Policy Catalog](tier-policy-catalog.md) — Policy resolution and overrides

## All Reference Pages

| Page | Purpose |
| --- | --- |
| [API Guide](api-guide.md) | Programmatic interface for CoreRunner |
| [Artifact Layout](artifacts.md) | Run outputs and certificate locations |
| [Calibration Reference](calibration.md) | Guard threshold recalibration |
| [Certificate Schema](certificate-schema.md) | v1 certificate contract |
| [Certificate Telemetry](certificate-telemetry.md) | Telemetry fields in certificates |
| [CLI Reference](cli.md) | Command-line interface |
| [Configuration Schema](config-schema.md) | YAML config structure |
| [Dataset Providers](datasets.md) | Supported data sources |
| [Environment Variables](env-vars.md) | Runtime toggles |
| [Exporting Certificates (HTML)](exporting-certificates-html.md) | HTML certificate exports |
| [GPU/MPS-First Guards](gpu-mps-first-guards.md) | Guard design decisions |
| [Guards](guards.md) | Guard configuration and policies |
| [Model Adapters](model-adapters.md) | Adapter interfaces |
| [Programmatic Quickstart](programmatic-quickstart.md) | Minimal Python example |
| [Tier Policy Catalog](tier-policy-catalog.md) | Policy keys and rationale |

## Related Sections

- [User Guide](../user-guide/getting-started.md) — Tutorials and workflows
- [Assurance Notes](../assurance/00-safety-case.md) — Mathematical foundations
- [Security](../security/architecture.md) — Threat model and best practices
