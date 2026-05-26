# Alternatives Comparison

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Position InvarLock against adjacent evaluation, MLOps, and compression tools. |
| **Audience** | Prospective users, evaluators, maintainers writing integration guidance. |
| **Scope** | Current project positioning; not a vendor compatibility contract. |
| **Source of truth** | `docs/assurance/14-trust-model.md`, `docs/reference/reports.md`, `docs/reference/guards.md`. |

InvarLock is not a general model benchmark harness or MLOps monitoring system.
It is a paired, verifier-friendly regression assurance workflow for edited
weights.

## When To Use InvarLock

- You produced an edited checkpoint (quantization, pruning, fine-tune) and
  need a machine-verifiable artifact that proves it did not regress beyond
  configured bounds.
- A release gate needs a fail-closed `evaluate -> verify` contract with
  pairing, guard evidence, and runtime provenance.
- Reviewers expect a self-contained report bundle they can re-verify offline.

## When To Reach For Something Else

- You want broad downstream benchmark scores (use lm-evaluation-harness or
  LightEval).
- You want production drift monitoring, dashboards, or experiment tracking
  (use MLflow, Evidently, Deepchecks).
- You need the tool to *produce* the compressed checkpoint (use Optimum,
  Intel Neural Compressor, GPTQModel, etc.); InvarLock validates the artifact
  afterwards.

## Tool Comparison

| Tool family | Use it for | How InvarLock differs |
| --- | --- | --- |
| lm-evaluation-harness, LightEval | Broad benchmark quality and task scores. | InvarLock focuses on paired baseline-vs-subject windows, guard evidence, runtime provenance, and a standalone report verifier. |
| OpenAI Evals | Custom LLM and system evaluations. | InvarLock operates on local checkpoint comparisons and weight-edit evidence. |
| MLflow, Evidently, Deepchecks | Experiment validation, monitoring, drift, and dashboards. | InvarLock ships a narrow fail-closed artifact contract for weight edits rather than a broad observability platform. |
| Hugging Face Optimum, Intel Neural Compressor, GPTQModel | Producing optimized or compressed model artifacts. | InvarLock validates the artifact after the edit instead of performing the compression. |

## Recommended Combined Workflow

1. Use compression or edit tooling to produce the subject checkpoint.
2. Use lm-eval / LightEval for broad downstream benchmark confidence.
3. Use an MLOps platform for tracking, monitoring, and dashboards.
4. Use InvarLock at the release gate to produce a machine-verifiable report
   that says a specific edited checkpoint did not exceed configured regression
   and guard thresholds relative to a fixed baseline.

## Related Documentation

- [Trust Model](../assurance/14-trust-model.md) — What a strict pass does and does not mean
- [Strict Assurance Checklist](../assurance/15-strict-assurance-checklist.md) — Reviewer acceptance criteria
- [Reports Reference](reports.md) — Schema, telemetry, and HTML export
- [Guards Reference](guards.md) — Configuration and evidence
- [Model Family Catalog](model-family-catalog.md) — Authoritative support inventory
- [Compare & evaluate (BYOE)](../user-guide/compare-and-evaluate.md) — Primary BYOE workflow
