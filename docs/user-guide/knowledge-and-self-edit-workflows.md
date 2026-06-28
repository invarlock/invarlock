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
- Record optional edit provenance metadata when you need reader context.
- Record evaluation realism metadata when the metric is a proxy for live generation.
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

## Evaluation Realism

Knowledge-edit and self-edit reviews often mix live generation checks with proxy
metrics such as teacher-forced log-probability. Record the evaluation mode so a
evidence reader can tell what behavior was actually exercised.

| Field | Purpose |
| --- | --- |
| `mode` | Evaluation mode: `generation`, `logprob`, `teacher_forced`, `classification`, or `benchmark_harness`. |
| `prompt_template_hash` | Digest of the prompt template used for generation or scoring. |
| `decoding_config` | Generation settings such as temperature, sampling, beams, or stop rules. |
| `max_tokens` | Maximum generated or scored tokens for the task. |
| `truncation_policy` | How prompts, contexts, or completions were truncated. |
| `dataset_or_task_id` | Dataset, task, or benchmark lane identifier. |
| `metric_is_generation_realistic` | Whether the primary metric reflects live generation behavior. |
| `proxy_metric_warning` | Short warning when the metric is useful but not live-generation realistic. |

Use this metadata as reader context: a teacher-forced or log-prob proxy is a
regression signal, while live generation behavior belongs in a generation-mode
lane.

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

## Optional Topology Metadata

Most v1 examples are baseline checkpoint versus subject checkpoint. Some edit
systems may produce adapter packages, merged adapters, memory modules, dynamic
weight modules, runtime configs, or prompt wrappers. Use optional topology
metadata when the subject is more than one static checkpoint.

| Field | Purpose |
| --- | --- |
| `artifact_kind` | Subject artifact kind such as `checkpoint`, `adapter`, `merged_adapter`, `memory_module`, `dynamic_weight_module`, `runtime_config`, or `prompt_wrapper`. |
| `module_hashes` | Hashes for adapter, memory, generator, routing, or wrapper modules. |
| `runtime_activation_policy` | Declared activation or routing condition for runtime-dependent modules. |
| `training_or_edit_data_ref` | Public reference or hash-only pointer to training/edit data when applicable. |

Topology metadata is descriptive; verifier verdicts still come from the selected
baseline-vs-subject evaluation policy.

## Delta And Privacy Exposure

Public evidence packs should not require raw deltas, adapter weights, parameter
patches, or other recovered-subject-sensitive artifacts by default. Those
materials may expose sensitive or proprietary edit information depending on the
method and threat model.

Use `delta_privacy` metadata to tell evidence readers whether raw edit material is
absent, private, public, or hash-only:

| Field | Purpose |
| --- | --- |
| `delta_available` | `none`, `private`, `public`, or `hash_only`. |
| `privacy_sensitivity` | `public`, `internal`, `customer_controlled`, or `sensitive`. |
| `public_raw_delta_approved` | Whether public disclosure of raw deltas or adapter weights was explicitly approved. |

InvarLock preserves this disclosure metadata for review. Treat privacy analysis
as upstream release review, and prefer hash-only or manifest-only evidence for
public bundles unless raw artifact disclosure is intentional.

## Evidence Packs

Evidence packs can carry the resulting reports, checksums, runtime manifests,
optional signatures, and edit metadata. They keep model weights external by
default. For external subject workflows, include checkpoint references or
digests that let evidence readers understand which baseline and subject were compared.

## Related Documentation

- [Compare & evaluate (BYOE)](compare-and-evaluate.md) — Primary baseline-vs-subject workflow
- [Bring Your Own Data](bring-your-own-data.md) — Custom datasets for scenario lanes
- [Reading a report](reading-report.md) — Report interpretation
- [Evidence Packs](evidence-packs.md) — Portable evidence artifacts
- [Assurance Case](../assurance/00-assurance-case.md) — Current scoped assurance boundary
