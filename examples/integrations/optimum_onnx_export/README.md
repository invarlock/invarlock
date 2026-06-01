# Hugging Face Optimum ONNX Export Compatibility Example

Status: `compatibility-investigation`

This example probes the Hugging Face Optimum ONNX export path for a tiny causal
LM. It exports `sshleifer/tiny-gpt2` to ONNX, checks that the exported artifact
can be opened through ONNX Runtime and Optimum's ORT model class, and records
why the artifact is not yet a direct input to the shared InvarLock comparison
wrapper.

The exported ONNX directory is useful runtime evidence for an Optimum deployment
lane. It is not a HF PyTorch checkpoint: `transformers.AutoModelForCausalLM`
does not load it as a standard checkpoint, so the shared
`run_invarlock_compare.sh` path should still use the original HF-loadable
baseline or a separately materialized HF-loadable subject.

The example is source-tree only. It does not add Optimum, ONNX, or ONNX Runtime
to the core InvarLock install.

## Prerequisites

Install InvarLock with the Hugging Face stack and add Optimum ONNX Runtime
dependencies to the same example environment:

```bash
python -m pip install "invarlock[hf]" "optimum-onnx[onnxruntime]"
```

From a repository checkout, an existing `.venv` with `invarlock[hf]` is also
fine:

```bash
.venv/bin/python -m pip install "optimum-onnx[onnxruntime]"
```

## Run

From the repository root:

```bash
examples/integrations/optimum_onnx_export/run_tiny_optimum_onnx_probe.sh \
  --allow-network \
  --force
```

The export defaults to CPU, batch size 1, and sequence length 8 so it remains a
small local compatibility probe.

## Outputs

The runner writes generated outputs under ignored local directories:

| Path | Role |
| --- | --- |
| `models/tiny-gpt2-optimum-onnx/` | Optimum ONNX export directory. |
| `reports/tiny-optimum-onnx/compatibility_probe.json` | Machine-readable compatibility result. |
| `reports/tiny-optimum-onnx/run_command.txt` | Runner invocation, export command, and inspection command. |

The compatibility report includes file hashes, ONNX Runtime provider details,
an Optimum ORT load probe, and the expected HF PyTorch checkpoint load result.

## Pairing With InvarLock Evidence

For a release-gate comparison, keep using an HF-loadable model path with the
shared wrapper:

```bash
examples/integrations/_shared/run_invarlock_compare.sh \
  --baseline sshleifer/tiny-gpt2 \
  --subject ./models/hf-loadable-subject \
  --report-out ./examples/integrations/optimum_onnx_export/reports/tiny-invarlock-pair \
  --allow-network
```

Use `compatibility_probe.json` to document the Optimum deployment artifact, and
use the InvarLock verifier result for the baseline-vs-subject regression claim.
