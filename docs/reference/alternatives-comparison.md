# Alternatives Comparison

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Position InvarLock against adjacent evaluation, MLOps, and compression tools. |
| **Audience** | Prospective users, evaluators, maintainers writing integration guidance. |
| **Scope** | Current project positioning and handoff guidance for adjacent toolchains. |
| **Source of truth** | `docs/assurance/14-trust-model.md`, `docs/reference/reports.md`, `docs/reference/guards.md`. |

InvarLock is a paired, verifier-friendly regression assurance workflow for
edited weights. It fits after edit or compression tooling has produced a subject
checkpoint and alongside broader benchmark, monitoring, and registry systems.

## When To Use InvarLock

- You produced an edited checkpoint (quantization, pruning, fine-tune) and
  need a machine-verifiable artifact showing it stayed within
  configured bounds.
- A strict verification workflow needs a fail-closed `evaluate -> verify` contract with
  pairing, guard evidence, and runtime provenance.
- Auditors expect a self-contained report bundle they can re-verify offline.

## When To Reach For Something Else

- You want broad downstream benchmark scores (use lm-evaluation-harness,
  LightEval, or an in-house evaluation stack).
- You want production drift monitoring, dashboards, or experiment tracking
  (use MLflow, Evidently, Deepchecks).
- You need the tool to *produce* the compressed checkpoint (use Optimum,
  Intel Neural Compressor, GPTQModel, etc.); InvarLock validates the artifact
  afterwards.

## Tool Comparison

| Tool family | Use it for | How InvarLock differs |
| --- | --- | --- |
| lm-evaluation-harness, LightEval, custom eval runners | Broad benchmark quality and task scores. | InvarLock focuses on paired baseline-vs-subject windows, guard evidence, runtime provenance, and a standalone report verifier. |
| OpenAI Evals | Custom LLM and system evaluations. | InvarLock operates on local checkpoint comparisons and weight-edit evidence. |
| MLflow, Evidently, Deepchecks | Experiment validation, monitoring, drift, and dashboards. | InvarLock adds a narrow fail-closed artifact contract for weight edits. |
| Hugging Face Optimum, Intel Neural Compressor, GPTQModel | Producing optimized or compressed model artifacts. | InvarLock validates the produced artifact after the edit. |

## Recommended Combined Workflow

1. Use compression or edit tooling to produce the subject checkpoint.
2. Use lm-eval, LightEval, or a custom eval stack for broad downstream benchmark confidence.
3. Use an MLOps platform for tracking, monitoring, and dashboards.
4. Use InvarLock in the strict verification workflow to produce a machine-verifiable report
   that says a specific edited checkpoint stayed within configured regression
   and guard thresholds relative to a fixed baseline.
5. Export the InvarLock result into existing workflow surfaces with
   `invarlock report export --format mlflow-tags`,
   `--format model-card-md`, or `--format release-review-md`.

## Related Documentation

- [Trust Model](../assurance/14-trust-model.md) — Strict pass scope
- [Strict Assurance Checklist](../assurance/15-strict-assurance-checklist.md) — Evidence acceptance criteria
- [Reports Reference](reports.md) — Schema, telemetry, and HTML export
- [Guards Reference](guards.md) — Configuration and evidence
- [Model Family Catalog](model-family-catalog.md) — Support tiers and backlog
- [Compare & evaluate (BYOE)](../user-guide/compare-and-evaluate.md) — Primary BYOE workflow
