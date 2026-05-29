# Public Evidence Walkthrough

## Purpose

This walkthrough shows the shipped public evidence floor that reviewers can
verify without downloading model weights. It is deliberately BYOE-oriented:
InvarLock validates a baseline/subject comparison artifact; it does not produce
deployable quantized checkpoints.

## Published-basis pass

The repository ships strict-pass public-basis examples for GPT-2-style causal LM
and BERT-style masked LM lanes:

```bash
invarlock verify --profile release --assurance strict \
  public_evidence/published_basis/gpt2/evaluation.report.json

invarlock verify --profile release --assurance strict \
  public_evidence/published_basis/bert/evaluation.report.json
```

Each directory includes:

| File | Role |
| --- | --- |
| `evaluation.report.json` | Canonical verifier input with primary metric, guard evidence, policy digest, and assurance section. |
| `runtime.manifest.json` | Container runtime provenance manifest bound to the report by SHA-256. |
| `evidence_pack_recipe.json` | Recipe pointer for rebuilding a full validation evidence pack. |

The support matrix records these paths under
`contracts/support_matrix.json` as the `published_basis` evidence floor.

## Caught regression

The caught-regression fixture keeps the naive primary metric acceptable
(`ratio_vs_baseline = 1.0`) but marks the spectral guard as failed:

```bash
invarlock verify --profile release --assurance strict \
  public_evidence/caught_regressions/spectral_guard_failure/evaluation.report.json
```

Expected outcome: verification fails. The failure is not a perplexity failure;
it is a guard/policy failure:

```text
Release verification requires validation.spectral_stable == true
spectral did not pass
```

That is the intended release-gate behavior: a clean summary metric is not enough
when a guard detects unstable weight geometry.

## Applying this to your checkpoint

Use your own edited checkpoint from a quantization, pruning, distillation, or
fine-tuning pipeline, then run `invarlock evaluate` or generate an
`evaluation.report.json` from paired run reports:

```bash
invarlock report generate \
  --run runs/subject/report.json \
  --baseline-run-report runs/baseline/report.json \
  --format report \
  -o reports/eval

invarlock verify --profile release --assurance strict \
  reports/eval/evaluation.report.json
```

Keep `evaluation.report.json` and `runtime.manifest.json` together. Use
`invarlock advanced runtime-verify` only when you specifically want to inspect
the manifest/report binding; use `invarlock verify` for the full release gate.
