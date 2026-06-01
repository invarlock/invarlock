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

From the repository root:

```bash
examples/integrations/lm_eval_harness/run_tiny_lm_eval_sidecar.sh \
  --allow-network \
  --force
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

Use `--device mps` on Apple Silicon when the local harness environment supports
it. The default is `--device cpu` for portable smoke runs.

## Outputs

The runner writes generated outputs under ignored local directories:

| Path | Role |
| --- | --- |
| `reports/tiny-lm-eval-sidecar/baseline/` | Raw LM Eval output directory for the baseline. |
| `reports/tiny-lm-eval-sidecar/subject/` | Raw LM Eval output directory for the optional subject. |
| `reports/tiny-lm-eval-sidecar/lm_eval_sidecar_summary.json` | Compact sidecar summary with task metrics and optional baseline-vs-subject deltas. |
| `reports/tiny-lm-eval-sidecar/run_command.txt` | Runner invocation and LM Eval commands. |

The default `--limit 1` setting is intentionally a smoke-test setting. Remove or
raise the limit only when you want meaningful task metrics and have recorded the
corresponding InvarLock regression artifacts separately.

## Pairing With InvarLock Evidence

Run an InvarLock comparison for the same baseline and subject when the edit
artifact is HF-loadable:

```bash
examples/integrations/_shared/run_invarlock_compare.sh \
  --baseline sshleifer/tiny-gpt2 \
  --subject ./examples/integrations/peft_lora/models/tiny-gpt2-peft-lora-merged \
  --report-out ./examples/integrations/lm_eval_harness/reports/tiny-invarlock-pair \
  --allow-network
```

Use the InvarLock verifier result for the release-gate claim, and use
`lm_eval_sidecar_summary.json` for task-score context.
