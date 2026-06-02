# LM Evaluation Harness Sidecar Example

Status: `exploratory-host`

This example shows how to place LM Evaluation Harness task metrics beside
InvarLock regression evidence for the same checkpoint-edit workflow. It runs a
tiny `wikitext` task against a baseline model, and optionally a subject model,
then normalizes the harness JSON into a compact sidecar summary.

The sidecar summary is not an InvarLock verifier input. Keep
`evaluation.report.json`, `verify.json`, and `evaluation.html` from an InvarLock
comparison as the evidence artifacts for regression claims; use the LM Eval
outputs as broader task-score context.

The example is source-tree only. It does not add LM Evaluation Harness to the
core InvarLock install.

## Prerequisites

Install InvarLock with the Hugging Face stack and add LM Evaluation Harness to
the same example environment:

```bash
python -m pip install "invarlock[hf]" "lm_eval[hf]"
```

From a repository checkout, an existing `.venv` with `invarlock[hf]` is also
fine:

```bash
.venv/bin/python -m pip install "lm_eval[hf]"
```

## Run

## Lane Support

| Artifact lane label | Command shape | Notes |
| --- | --- | --- |
| `cuda-container-strict` | Paired InvarLock comparison with `--lane cuda` | Primary verifier-evidence path; not produced by the sidecar itself. |
| `cuda-host-off` | `--device cuda` | Secondary CUDA sidecar task-score run with host dependencies. |
| `cpu-host-off` | `--device cpu` | Portable sidecar smoke run. |
| `mps-host-off` | `--device mps` | Apple Silicon sidecar task-score run when the harness environment supports MPS. |

Treat the paired InvarLock `cuda-container-strict` run as the primary evidence
path. The sidecar host lanes are secondary task-score context and run
prerequisite preflight before LM Evaluation Harness is invoked. The
`cuda-host-off` lane checks `torch.cuda.is_available()` before the harness run.

### cuda-container-strict InvarLock evidence

Run an InvarLock comparison for the same baseline and subject when the edit
artifact is HF-loadable:

```bash
make runtime-image-cuda

INVARLOCK_RUNTIME_IMAGE=invarlock-runtime:cuda-local \
examples/integrations/_shared/run_invarlock_compare.sh \
  --baseline sshleifer/tiny-gpt2 \
  --subject ./examples/integrations/peft_lora/models/tiny-gpt2-peft-lora-merged \
  --report-out ./examples/integrations/lm_eval_harness/reports/tiny-invarlock-pair \
  --lane cuda \
  --allow-network
```

Use the InvarLock verifier result for the release-gate claim, and use
`lm_eval_sidecar_summary.json` for task-score context. For local debug, use the
same comparison with `--lane host`. Do not use an identical baseline and subject
as a placeholder for this paired run; the verifier can correctly fail that as a
non-edit comparison instead of producing useful regression evidence.

### cpu-host-off lane

From the repository root:

```bash
examples/integrations/lm_eval_harness/run_tiny_lm_eval_sidecar.sh \
  --allow-network \
  --force \
  --device cpu
```

To compare a subject checkpoint that is already loadable by Hugging Face, pass
the subject path or model ID:

```bash
examples/integrations/lm_eval_harness/run_tiny_lm_eval_sidecar.sh \
  --baseline sshleifer/tiny-gpt2 \
  --subject ./examples/integrations/peft_lora/models/tiny-gpt2-peft-lora-merged \
  --allow-network \
  --force
```

The default is `--device cpu` for portable smoke runs.

### cuda-host-off lane

Run this lane on a host where LM Evaluation Harness and the selected model can
use a CUDA device:

```bash
examples/integrations/lm_eval_harness/run_tiny_lm_eval_sidecar.sh \
  --allow-network \
  --force \
  --device cuda \
  --batch-size auto
```

### mps-host-off lane

Use `--device mps` on Apple Silicon when the local harness environment supports
it. That writes an `mps-host-off` sidecar summary. This sidecar does not produce
InvarLock runtime provenance; pair it with a `cuda-container-strict` InvarLock
run for the release-gate evidence claim.

## Outputs

The runner writes generated outputs under ignored local directories:

| Path | Role |
| --- | --- |
| `reports/tiny-lm-eval-sidecar/baseline/` | Raw LM Eval output directory for the baseline. |
| `reports/tiny-lm-eval-sidecar/subject/` | Raw LM Eval output directory for the optional subject. |
| `reports/tiny-lm-eval-sidecar/lm_eval_sidecar_summary.json` | Compact sidecar summary with task metrics, lane label, and optional baseline-vs-subject deltas. |
| `reports/tiny-lm-eval-sidecar/run_command.txt` | Runner invocation and LM Eval commands. |
| `reports/tiny-lm-eval-sidecar/run_summary.txt` | Concise success or failure status, lane label, verifier status, runtime provenance status, and primary output paths. |

The default `--limit 1` setting is intentionally a smoke-test setting. Remove or
raise the limit only when you want meaningful task metrics and have recorded the
corresponding InvarLock regression artifacts separately.
