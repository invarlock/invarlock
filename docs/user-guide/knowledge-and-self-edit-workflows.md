---
title: Knowledge & self-edit workflows
---

## Knowledge & self-edit workflows

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Evaluate a subject checkpoint produced by an external knowledge-edit, self-edit, LoRA, fine-tune, pruning, or quantization workflow. |
| **Audience** | Teams that already have an upstream edit workflow and need baseline-vs-subject evidence before release. |
| **Workflow** | External edit tool -> subject checkpoint -> InvarLock Compare & evaluate (BYOE) -> report -> verification -> optional evidence pack. |
| **Subject artifact** | A reproducible checkpoint, merged adapter output, or declared runtime artifact produced before evaluation. |
| **Output** | `evaluation.report.json`, `evaluation_report.md`, and `runtime.manifest.json` for container-backed runs. |

Knowledge-edit and self-edit workflows usually end with a changed subject:
a rewritten checkpoint, merged adapter, self-adapted checkpoint, generated
adapter state, or another reproducible artifact. InvarLock’s v1 workflow starts
from that artifact. The external editor creates the subject, and Compare &
evaluate measures the declared baseline against that subject under the selected
dataset windows, tier, profile, guard policy, and runtime policy.

The optional metadata in this guide records how the subject was produced and how
the evaluation lanes were organized. In the current release, those fields are
reporting context for the existing weight-edit regression contract; a future
named profile would be required before scenario labels or edit provenance become
claim-bearing.

## TL;DR

- Produce or reference the baseline checkpoint.
- Produce the edited subject checkpoint with your upstream workflow.
- Run Compare & evaluate (BYOE) with `--baseline` and `--subject`.
- Record optional edit provenance metadata when you need reviewer context.
- Verify `evaluation.report.json` together with its `runtime.manifest.json`.

```bash
INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate --allow-network \
  --baseline <BASELINE_MODEL> \
  --subject <SUBJECT_MODEL_OR_PATH> \
  --baseline-adapter auto \
  --subject-adapter auto \
  --edit-label custom \
  --profile ci \
  --out runs/knowledge_edit_eval \
  --report-out reports/knowledge_edit_eval
```

By default, `evaluate` uses the runtime container. Use `--execution-mode host`
only for host-side workflows that intentionally bypass container execution.

## Upstream Artifact Contract

The upstream workflow is responsible for edit creation. InvarLock expects the
resulting subject to be reproducible enough for evaluation and review.

| Upstream workflow | InvarLock role |
| --- | --- |
| Knowledge-edit method | Evaluate the produced subject artifact against the baseline. |
| Self-edit or self-adaptation loop | Evaluate the resulting checkpoint and record optional source metadata. |
| LoRA merge or fine-tune | Treat the merged or fine-tuned checkpoint as the subject. |
| Quantization or pruning | Treat the compressed or pruned checkpoint as the subject. |
| Dynamic runtime adapter | Evaluate only when the runtime behavior can be represented by a declared subject artifact and metadata. |

If behavior depends on runtime-generated weights or context rather than a stable
checkpoint, record that with `dynamic_runtime_required: true` and keep the
evidence scope narrow. Context-conditioned behavior is not a separate artifact
class in this workflow.

## Optional Edit Provenance

Use optional metadata to make the upstream subject-generation process easier to
audit. Current verifiers preserve and validate these fields when present while
keeping their verdicts tied to the existing baseline-vs-subject evidence.

| Field | Purpose |
| --- | --- |
| `edit_family` | Broad family such as `lora_merge`, `knowledge_edit`, `self_edit`, `magnitude_prune`, `quantization_dequantized`, or `custom`. |
| `edit_method` | Method label supplied by the producer, such as `custom`. |
| `edit_count` | Number of target edits or edit steps represented by the subject. |
| `target_set_digest` | Digest of the target-edit set without exposing sensitive contents. |
| `editor_artifact_digest` | Digest of the upstream editor, recipe, or generator artifact when available. |
| `self_edit_data_digest` | Digest of self-generated data or directives when applicable. |
| `dynamic_runtime_required` | Whether evaluation depends on runtime-generated edit behavior. |

## Optional Edit-Impact Scenarios

Scenario labels organize evaluation lanes for knowledge-edit and self-edit
reviews. They give readers a compact map of target, neighbor, locality, sentinel,
portability, and sequential-edit checks.

| Scenario type | Purpose |
| --- | --- |
| `target_success` | The intended target behavior was evaluated. |
| `near_neighbor` | Equivalent or paraphrased target-adjacent behavior was evaluated. |
| `near_confuser` | Semantically adjacent but wrong targets were checked. |
| `unrelated_locality` | Unrelated behavior was checked for baseline-relative preservation. |
| `general_ability_sentinel` | A general task sentinel was checked for regression. |
| `multilingual_portability` | Language-specific behavior was checked where required. |
| `sequential_edit_stress` | Batch or sequential edit accumulation was checked. |

For v1, keep these labels descriptive in reports, examples, and evidence packs.
Turning them into strict gates would require a future profile with thresholds,
calibration, and assurance evidence.

## Evidence Packs

Evidence packs can carry the resulting reports, checksums, runtime manifests,
optional signatures, and edit metadata. They keep model weights external by
default. For external subject workflows, include checkpoint references or
digests that let reviewers understand which baseline and subject were compared.

## Related Documentation

- [Compare & evaluate (BYOE)](compare-and-evaluate.md) — Primary baseline-vs-subject workflow
- [Bring Your Own Data](bring-your-own-data.md) — Custom datasets for scenario lanes
- [Reading a report](reading-report.md) — Report interpretation
- [Evidence Packs](evidence-packs.md) — Portable evidence artifacts
- [Assurance Case](../assurance/00-assurance-case.md) — Current scoped assurance boundary
