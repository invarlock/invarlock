# Inspect AI

This example runs a native [Inspect AI](https://inspect.aisi.org.uk/) Task and
its exact-match scorer over two pinned model evaluations, then completes
`invarlock evaluate`, `invarlock verify`, and `invarlock report`. InvarLock
recomputes the paired result from every imported record instead of trusting the
evaluator aggregate.

The default `quick` profile compares Qwen3.5 0.8B Base with its post-trained
checkpoint over 102 local records on CPU. The retained `deployment` profile
uses the same checkpoints with a tokenizer-qualified 400-record LAMBADA
completion corpus on CUDA. The retained `flagship` profile
compares revision-pinned Qwen3.5 9B Base and post-trained checkpoints over 400
balanced MMLU-Pro records on CUDA. Every required snapshot file, tokenizer
contract, task setting, evaluator version, runtime image, and per-record output
is digest-bound.

## Run the integration

From a clean committed checkout with Docker or Podman available:

```bash
make example-inspect-ai EXAMPLE_ARGS="--evidence-signing-key /secure/keys/evidence.pem --verifier-signing-key /secure/keys/verifier.pem --builder-signing-key /secure/keys/builder.pem --builder-public-key /secure/keys/builder-public.pem --trust-root /secure/trust/inspect-ai"
```

Select the current-model flagship with:

```bash
make example-inspect-ai EXAMPLE_ARGS="--corpus-profile flagship --evidence-signing-key /secure/keys/evidence.pem --verifier-signing-key /secure/keys/verifier.pem --builder-signing-key /secure/keys/builder.pem --builder-public-key /secure/keys/builder-public.pem --trust-root /secure/trust/inspect-ai"
```

Run the compact deployment-approval profile with:

```bash
make example-inspect-ai EXAMPLE_ARGS="--corpus-profile deployment --evidence-signing-key /secure/keys/evidence.pem --verifier-signing-key /secure/keys/verifier.pem --builder-signing-key /secure/keys/builder.pem --builder-public-key /secure/keys/builder-public.pem --trust-root /secure/trust/inspect-ai"
```

The shared `portability` profile can also run the Gemma 4 12B instruction and
official QAT-Q4 source checkpoints through Inspect AI. The compact retained
Gemma transaction uses LM Evaluation Harness; the retained Qwen3.5 transactions
use both evaluators and therefore isolate cross-evaluator agreement separately
from cross-family portability.

The GPU profiles require an NVIDIA CUDA runtime and enough memory for one model
at a time. A 32 GB GPU is a practical minimum for the BF16 singleton runs.
Pass `EXAMPLE_ARGS="--workspace PATH"` to retain the complete transaction at a
new path. Signing keys and the trust root remain caller-owned and outside the
transaction.

## Frozen 400-record suite

The flagship profile uses one revision- and hash-pinned 400-item semantic
selection from
[TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro),
balanced across all 14 subject categories and answer labels A–J. It binds the
Qwen no-thinking chat rendering, derived JSONL digest, stable record order, and
1,024-token input ceiling. After model and image preparation, both evaluator
runs execute without network access.

Inspect's causal Hugging Face decoder removes leading completion whitespace.
The native task therefore uses an authenticated raw-text chat template, and the
bridge records its explicit boundary recovery before strict replay. Missing or
reordered IDs, changed inputs or outputs, scorer disagreement, and modified
post-run samples fail closed.

The 400-record policy requires all records, at least 20% accuracy on each side,
a paired 95% interval no wider than 10 percentage points, and a lower bound of
at least −2 percentage points. A verified policy rejection is retained only
with explicit `--allow-policy-fail`; malformed or untrusted evidence remains an
error.

The retained Qwen3.5 transaction measured 55.5% baseline accuracy and 53.0%
subject accuracy, with a −2.5-point estimate and a 7.85-point interval width.
Its evidence and receipt passed integrity verification, while the regression
policy rejected the comparison because the confidence lower bound crossed the
declared floor.

The separate deployment profile deterministically selects 400 records from a
revision- and hash-pinned EleutherAI LAMBADA source. Every target is one
lossless token under both Qwen3.5 0.8B tokenizers, prompts fit the 256-token
ceiling, and four prompt-length strata contribute 100 records each. Its
policy declared in advance requires 400 records, at least 5% accuracy on each side, an
interval no wider than 10 percentage points, and a lower bound of at least
−20 percentage points. The retained transaction measured 49.25% baseline and
43.50% subject accuracy, a −5.75-point estimate, and a 6.52-point interval
width; its signed policy verdict passed.

LM Evaluation Harness and Inspect AI produced identical ordered output records
for all 400 baseline and all 400 subject examples under the singleton profile.
The retained comparison reports this agreement without creating another
acceptance decision.

This is a reproducible integration and regression-policy demonstration, not a
general model-quality ranking. Production use should select datasets and
thresholds that represent the intended deployment.
