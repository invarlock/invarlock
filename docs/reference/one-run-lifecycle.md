# One Run Lifecycle

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Map one `evaluate -> verify -> report` journey to code and artifact owners. |
| **Audience** | Maintainers, reviewers auditing the assurance boundary, contributors tracing failures. |
| **Contract scope** | Current strict assurance flow and report v1 artifacts. |
| **Source of truth** | `src/invarlock/cli/commands/evaluate.py`, `src/invarlock/core/evaluate_plan.py`, `src/invarlock/core/assurance_contract.py`, `src/invarlock/reporting/verify_contract.py`. |

This page maps one `evaluate -> verify -> report` journey to the code and
artifact surfaces reviewers inspect.

## Quick Start

The minimal end-to-end trace for a single comparison:

```bash
invarlock evaluate --allow-network \
  --baseline gpt2 \
  --subject distilgpt2 \
  --adapter auto \
  --profile ci \
  --assurance strict \
  --report-out reports/eval

invarlock verify --assurance strict reports/eval/evaluation.report.json
invarlock report html -i reports/eval/evaluation.report.json -o reports/eval/evaluation.html
```

Each stage emits artifacts the next stage consumes; reviewers can pause at any
stage to inspect the surface in the table below. The `evaluate` command uses
the runtime container by default for model-loading work; host execution must be
an explicit non-assurance bypass.

## Stage Map

| Stage | Code surface | Artifact surface |
| --- | --- | --- |
| CLI planning | `invarlock.cli.commands.evaluate`, `invarlock.core.evaluate_plan` | selected profile, tier, preset, adapter, runtime policy |
| Runtime policy | `invarlock.runtime_security`, `invarlock.cli.evaluate_phases` | `runtime.manifest.json` |
| Config loading | `invarlock.core.config_loader` | normalized run config, `context.assurance` |
| Component resolution | `invarlock.cli.run_execution`, guard/adapter/edit registries | resolved adapter, edit, and guard order |
| Guard execution | `invarlock.core.runner`, `invarlock.guards.*` | guard evidence and statuses |
| Metric computation | `invarlock.core.bootstrap`, runner metric helpers | paired delta log-loss, ratio, CI fields |
| Report assembly | `invarlock.reporting.report_make_output` | `evaluation.report.json` |
| Verification | `invarlock.reporting.verify_contract` | verifier pass/fail details |
| Human report | `invarlock report html` | rendered HTML report |

## Assurance Boundary

The strict assurance boundary starts at CLI planning and ends at verifier
acceptance. Strict mode is not inferred from profile names alone; it is
recorded in `assurance.mode` and checked by the verifier.

## Debugging Rule

When a strict report fails verification, fix the earliest source evidence that
caused the failure. Do not patch the report artifact by hand. The stage table
above lets you trace a failure back to the owning code path and the artifact
where the evidence is recorded.

## Related Documentation

- [System Architecture](architecture.md)
- [Reports Reference](reports.md)
- [CLI Reference](cli.md)
- [Trust Model](../assurance/trust-model.md)
- [Strict Assurance Checklist](../assurance/strict-assurance-checklist.md)
- [Runtime Provenance Guide](../security/runtime-provenance-guide.md)
- [Failure Examples](../user-guide/failure-examples.md)
