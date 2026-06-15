# LM Evaluation Harness Sidecar Example

Status: `exploratory-host`

This example shows how to place LM Evaluation Harness task metrics beside
InvarLock regression evidence for the same checkpoint-edit workflow. It runs a
tiny `wikitext` task against a baseline model, and optionally a subject model,
then normalizes the harness JSON into a compact sidecar summary.

Use `evaluation.report.json`, `verify.json`, and `evaluation.html` from an
InvarLock comparison as the evidence artifacts for regression claims. The LM
Eval sidecar provides broader task-score context beside those artifacts.

The example keeps LM Evaluation Harness in the example environment rather than
the core InvarLock install.

The same sidecar pattern applies to LightEval or in-house evaluation runners:
write their task metrics beside the InvarLock report as context, keep their
dependencies outside the core install, and do not treat those sidecar scores as
verifier inputs unless a future report schema explicitly defines that contract.

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

If the checkout environment was created by `uv sync`, install the optional
sidecar dependency into that environment with:

```bash
uv pip install --python .venv/bin/python "lm_eval[hf]"
```

From a source checkout, you can also keep the optional dependency scoped to the
example command:

```bash
uv run --extra hf --with "lm_eval[hf]" \
  examples/integrations/lm_eval_harness/run_tiny_lm_eval_sidecar.sh --help
```

## Run

### Lane Support

| Artifact lane label | Command shape | Notes |
| --- | --- | --- |
| `cuda-container-strict` | Paired InvarLock comparison with `--lane cuda` | Primary verifier-evidence path; not produced by the sidecar itself. |
| `cuda-host-off` | `--device cuda` | Secondary CUDA sidecar task-score run with host dependencies. |
| `cpu-host-off` | `--device cpu` | Portable sidecar quick check. |
| `mps-host-off` | `--device mps` | Apple Silicon sidecar task-score run when the harness environment supports MPS. |

Treat the paired InvarLock `cuda-container-strict` run as the primary evidence
path. The sidecar host lanes are secondary task-score context and run
prerequisite preflight before LM Evaluation Harness is invoked. The
`cuda-host-off` lane checks `torch.cuda.is_available()` before the harness run.

### cuda-container-strict InvarLock evidence

Run the PEFT example first when you want the sidecar to use that generated
subject and fixture:

```bash
make runtime-image-cuda

INVARLOCK_RUNTIME_IMAGE=invarlock-runtime:cuda-local \
uv run --extra hf --with peft \
examples/integrations/peft_lora/run_tiny_peft_lora.sh \
  --allow-network \
  --force \
  --lane cuda
```

Then run an InvarLock comparison for the same baseline, subject, and PEFT
fixture when you want a paired report under the LM Eval example directory:

```bash
INVARLOCK_RUNTIME_IMAGE=invarlock-runtime:cuda-local \
examples/integrations/_shared/run_invarlock_compare.sh \
  --baseline sshleifer/tiny-gpt2 \
  --subject ./examples/integrations/peft_lora/models/tiny-gpt2-peft-lora-merged \
  --report-out ./examples/integrations/lm_eval_harness/reports/tiny-invarlock-pair \
  --lane cuda \
  --profile release \
  --preset ./examples/integrations/peft_lora/artifacts/tiny-peft-lora-fixture/preset.yaml \
  --edit-label peft_lora_merge \
  --allow-network
```

Use the InvarLock verifier result as the strict regression evidence, and use
`lm_eval_sidecar_summary.json` for task-score context. For a host-side
comparison, use the same command with `--lane host`. Do not use an identical
baseline and subject as a placeholder for this paired run; the verifier can
correctly fail that as a non-edit comparison instead of producing useful
regression evidence. For very small edits, a default `ci` profile comparison on
an unrelated dataset can also fail policy because the measured delta is too close
to the baseline.

### cpu-host-off lane

From the repository root:

```bash
uv run --extra hf --with "lm_eval[hf]" \
examples/integrations/lm_eval_harness/run_tiny_lm_eval_sidecar.sh \
  --allow-network \
  --force \
  --device cpu
```

To compare a subject checkpoint that is already loadable by Hugging Face, pass
the subject path or model ID:

```bash
uv run --extra hf --with "lm_eval[hf]" \
examples/integrations/lm_eval_harness/run_tiny_lm_eval_sidecar.sh \
  --baseline sshleifer/tiny-gpt2 \
  --subject ./examples/integrations/peft_lora/models/tiny-gpt2-peft-lora-merged \
  --allow-network \
  --force
```

The default is `--device cpu` for portable quick checks.

### cuda-host-off lane

Run this lane on a host where LM Evaluation Harness and the selected model can
use a CUDA device:

```bash
uv run --extra hf --with "lm_eval[hf]" \
examples/integrations/lm_eval_harness/run_tiny_lm_eval_sidecar.sh \
  --allow-network \
  --force \
  --device cuda \
  --batch-size auto
```

### mps-host-off lane

Use `--device mps` on Apple Silicon when the local harness environment supports
it. That writes an `mps-host-off` sidecar summary for task-score context. Pair
it with a `cuda-container-strict` InvarLock run for the strict evidence record.

## Outputs

The runner writes generated outputs under local output directories:

| Path | Role |
| --- | --- |
| `reports/tiny-lm-eval-sidecar/<artifact-lane>/baseline/` | Raw LM Eval output directory for the baseline. |
| `reports/tiny-lm-eval-sidecar/<artifact-lane>/subject/` | Raw LM Eval output directory for the optional subject. |
| `reports/tiny-lm-eval-sidecar/<artifact-lane>/lm_eval_sidecar_summary.json` | Compact sidecar summary with task metrics, lane label, and optional baseline-vs-subject deltas. |
| `reports/tiny-lm-eval-sidecar/<artifact-lane>/run_command.txt` | Runner invocation and LM Eval commands. |
| `reports/tiny-lm-eval-sidecar/<artifact-lane>/run_summary.txt` | Concise success or failure status, lane label, verifier status, runtime provenance status, and primary output paths. |

The default `--limit 1` setting is intentionally minimal. Remove or raise the
limit only when you want meaningful task metrics and have recorded the
corresponding InvarLock regression artifacts separately.
