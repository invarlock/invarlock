# Integration Examples

## Purpose

The source tree now has a first-party scaffold for optional integration
examples under `examples/integrations/`. These examples are for checkpoint-edit
toolchains that want to attach InvarLock regression evidence after producing an
edited subject checkpoint.

The integration surface is intentionally source-tree only. It is not part of the
runtime package and should not add required dependencies to the core install.

## Current Scope

| Surface | Status |
| --- | --- |
| Shared evidence wording | Present under `examples/integrations/_shared/evidence-scope.md`. |
| Expected artifact checklist | Present under `examples/integrations/_shared/expected-artifacts.md`. |
| Shared compare wrapper | Present under `examples/integrations/_shared/run_invarlock_compare.sh`. |
| PEFT LoRA merge | Runnable example under `examples/integrations/peft_lora/`. |
| torchao int8 export | Runnable example under `examples/integrations/torchao_int8_export/`. |
| LM Evaluation Harness sidecar | Exploratory host example under `examples/integrations/lm_eval_harness/`. |
| Additional target-specific examples | Added one target at a time after backend compatibility is validated. |

Browse the integration scaffold in the repository:
<https://github.com/invarlock/invarlock/tree/staging/next/examples/integrations>

## Target Example Readiness

Target examples should declare one of these labels in their README:

| Label | Meaning |
| --- | --- |
| `runnable` | Commands are expected to generate `evaluation.report.json`, `verify.json`, and `evaluation.html` in the documented environment. |
| `exploratory-host` | Commands run with `--execution-mode host --assurance off` for backend bring-up or local debugging. |
| `compatibility-investigation` | The external artifact cannot yet be loaded or verified through the documented InvarLock path; the README records the blocker. |

## Preflight

Before adding or publishing a target example:

```bash
invarlock doctor
invarlock advanced plugins list --json
```

For optional backend targets, also check the Python module:

```bash
python -c "import importlib.util; print(importlib.util.find_spec('gptqmodel') is not None)"
```

Replace `gptqmodel` with the backend module for the target example.

## Shared Workflow

For a baseline and subject that are already loadable through an InvarLock
adapter:

```bash
examples/integrations/_shared/run_invarlock_compare.sh \
  --baseline sshleifer/tiny-gpt2 \
  --subject ./models/tiny-gpt2-subject \
  --report-out ./reports/integration-smoke \
  --allow-network
```

The generated local output should include:

| Artifact | Role |
| --- | --- |
| `evaluation.report.json` | Canonical verifier input. |
| `verify.json` | Machine-readable verifier result. |
| `evaluation.html` | Human-readable report. |
| `runtime.manifest.json` | Runtime provenance for strict container-backed runs. |
| `run_command.txt` | Wrapper invocation plus evaluate, verify, and render commands. |

Generated reports, models, runs, HTML, and artifacts under
`examples/integrations/**` are ignored by git.

## PEFT LoRA Merge

The first target example materializes a tiny PEFT LoRA-merged subject checkpoint
and feeds it through the shared compare wrapper:

```bash
examples/integrations/peft_lora/run_tiny_peft_lora.sh \
  --allow-network \
  --force
```

This command requires PEFT in the example environment. It writes the merged
subject checkpoint under `examples/integrations/peft_lora/models/` and the
InvarLock artifacts under `examples/integrations/peft_lora/reports/`.
The runner defaults to the `release` profile so strict verification has enough
evaluation tokens for a stable primary-metric verdict.

For host-side dependency bring-up, use:

```bash
examples/integrations/peft_lora/run_tiny_peft_lora.sh \
  --allow-network \
  --force \
  --execution-mode host \
  --assurance off
```

## torchao Int8 Export

The `torchao` example creates a deterministic tiny local HF baseline, applies a
`torchao` int8 weight-only quantization pass, exports a dequantized HF-loadable
subject checkpoint, and feeds the baseline/subject pair through the shared
compare wrapper:

```bash
examples/integrations/torchao_int8_export/run_tiny_torchao_int8_export.sh \
  --allow-network \
  --force
```

This command requires `torchao` in the example environment. It writes the
generated baseline and exported subject under
`examples/integrations/torchao_int8_export/models/` and the InvarLock artifacts
under `examples/integrations/torchao_int8_export/reports/`.

The materializer records the native quantized HF save probe in
`external_edit_summary.json`; the runnable comparison path uses the exported
HF-loadable subject.

For host-side dependency bring-up, use:

```bash
examples/integrations/torchao_int8_export/run_tiny_torchao_int8_export.sh \
  --allow-network \
  --force \
  --execution-mode host \
  --assurance off
```

## LM Evaluation Harness Sidecar

The LM Evaluation Harness example records broad task metrics beside an
InvarLock regression-evidence run. It does not generate
`evaluation.report.json`, `verify.json`, or `evaluation.html`; those remain the
outputs of the InvarLock compare path.

Run a tiny baseline-only smoke task:

```bash
examples/integrations/lm_eval_harness/run_tiny_lm_eval_sidecar.sh \
  --allow-network \
  --force
```

Run the same sidecar with a subject checkpoint that is already HF-loadable:

```bash
examples/integrations/lm_eval_harness/run_tiny_lm_eval_sidecar.sh \
  --baseline sshleifer/tiny-gpt2 \
  --subject ./examples/integrations/peft_lora/models/tiny-gpt2-peft-lora-merged \
  --allow-network \
  --force
```

The runner writes raw LM Eval JSON plus
`reports/tiny-lm-eval-sidecar/lm_eval_sidecar_summary.json`. Use that summary
for task-score context after the matching InvarLock comparison has produced the
release-gate artifacts.

## Public Evidence Anchors

Use the shipped public evidence as the credibility floor while target examples
are being built:

```bash
invarlock verify --profile release --assurance strict \
  public_evidence/published_basis/gpt2/evaluation.report.json

invarlock verify --profile release --assurance strict \
  public_evidence/byoe_examples/lora_merge_byoe/evaluation.report.json

invarlock verify --profile release --assurance strict \
  public_evidence/real_runs/tiny_gpt2_quant_rtn/evaluation.report.json
```

See the [Public Evidence Walkthrough](public-evidence-walkthrough.md) for the
full artifact taxonomy.
