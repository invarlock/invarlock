# LM Evaluation Harness

This example runs [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
over two pinned checkpoints, imports every output, and completes `invarlock
evaluate`, `invarlock verify`, and `invarlock report`. InvarLock recomputes the
paired exact-match result instead of trusting the Harness aggregate.

Four profiles serve different purposes:

| Profile | Models | Records | Runtime | Purpose |
| --- | --- | ---: | --- | --- |
| `quick` | Qwen3.5 0.8B Base → post-trained | 102 local records | CPU/float32, batch 8 | Short local workflow |
| `deployment` | Qwen3.5 0.8B Base → post-trained | 400 tokenizer-qualified LAMBADA records | CUDA/BF16, batch 8 | Compact deployment-approval workflow |
| `flagship` | Qwen3.5 9B Base → post-trained | 400 balanced MMLU-Pro records | CUDA/BF16, batch 1 | Current-model evaluator comparison |
| `portability` | Gemma 4 12B IT → official QAT-Q4 source checkpoint | The same 400 semantic MMLU-Pro items | CUDA/BF16, batch 1 | Cross-family deployment-change evidence |

Every model revision and required snapshot file is bound by byte length and
SHA-256. Each run records its model tree, tokenizer contract, evaluator and
backend, generation settings, seed, runtime image, task configuration, and
per-record output digests. Remote model code and checkpoint generation defaults
are disabled.

## Run a profile

From a clean committed checkout with Docker or Podman available, the quick
profile is:

```bash
make example-lm-evaluation-harness EXAMPLE_ARGS="--evidence-signing-key /secure/keys/evidence.pem --verifier-signing-key /secure/keys/verifier.pem --builder-signing-key /secure/keys/builder.pem --builder-public-key /secure/keys/builder-public.pem --trust-root /secure/trust/lm-evaluation-harness"
```

Select the Qwen3.5 flagship with:

```bash
make example-lm-evaluation-harness EXAMPLE_ARGS="--corpus-profile flagship --evidence-signing-key /secure/keys/evidence.pem --verifier-signing-key /secure/keys/verifier.pem --builder-signing-key /secure/keys/builder.pem --builder-public-key /secure/keys/builder-public.pem --trust-root /secure/trust/lm-evaluation-harness"
```

Select the compact deployment profile with:

```bash
make example-lm-evaluation-harness EXAMPLE_ARGS="--corpus-profile deployment --evidence-signing-key /secure/keys/evidence.pem --verifier-signing-key /secure/keys/verifier.pem --builder-signing-key /secure/keys/builder.pem --builder-public-key /secure/keys/builder-public.pem --trust-root /secure/trust/lm-evaluation-harness"
```

Select the Gemma 4 portability profile with:

```bash
make example-lm-evaluation-harness EXAMPLE_ARGS="--corpus-profile portability --evidence-signing-key /secure/keys/evidence.pem --verifier-signing-key /secure/keys/verifier.pem --builder-signing-key /secure/keys/builder.pem --builder-public-key /secure/keys/builder-public.pem --trust-root /secure/trust/lm-evaluation-harness"
```

The GPU profiles require an NVIDIA CUDA runtime and enough memory for one model
at a time. A 32 GB GPU is a practical minimum for these BF16 singleton
runs. Allow roughly 50 GB of workspace disk per prepared profile, plus model
cache and container-image storage.

Pass `--workspace PATH` to retain the complete transaction at a new path. The
launcher otherwise creates a temporary workspace and prints it at completion.
It removes the exact temporary base and evaluator image tags it creates,
including when the workspace is retained.

## Frozen 400-record suite

The flagship and portability profiles derive model-specific prompts from one
revision- and hash-pinned 400-item semantic selection of
[TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro). The
selection balances all 14 subject categories and answer labels A–J. Stable
record IDs preserve pairing across model families, while each profile binds
its own official no-thinking chat rendering and derived JSONL digest.

The source can be supplied from a previously downloaded file with
`--benchmark-source PATH`; the same byte length and SHA-256 checks apply. Both
model runs execute without network access after snapshots and images are
prepared.

The 400-record policy was fixed before model execution. It requires all
records, at least 20% accuracy on each side, a paired 95% interval no wider than
10 percentage points, and a confidence-interval lower bound of at least −2
percentage points. A verified policy rejection is retained only when the
caller explicitly supplies `--allow-policy-fail`; integrity failures remain
errors.

The deployment profile instead uses a deterministic, revision- and hash-pinned
400-record selection from EleutherAI LAMBADA. It requires a lossless one-token
target under both Qwen3.5 0.8B tokenizers, a 256-token prompt ceiling, and 100
records from each of four prompt-length strata. Its independent policy uses a
5% side-accuracy floor, a 10-point maximum interval width, and a −20-point
lower-bound floor. The repository retains the passing Inspect AI transaction
for this profile; the Harness command is runnable but is not represented as a
second retained deployment decision.

The retained Qwen3.5 transaction measured 55.5% baseline accuracy and 53.0%
subject accuracy, with a −2.5-point estimate and a 7.85-point interval width.
The retained Gemma 4 transaction measured 44.0% and 42.5%, with a −1.5-point
estimate and a 4.68-point interval width. Both evidence packs passed integrity,
record-count, interval-width, and side-accuracy checks; both were rejected by
the conservative regression rule because their lower confidence bounds crossed
the declared −2-point floor.

## Trust boundary

The command builds a source-authenticated runtime and adds a cache-free package
derived from the hash-pinned `lm-eval` 0.4.12 wheel. The immutable inspected
image ID is bound into both runtime receipts and a builder-signed image
attestation before the evidence is signed.

The adapter requires one upstream sample for every schedule record, stable
dataset IDs, exact prompt and target hashes, the declared task configuration,
and a digest-bound run manifest. Aggregate-only files, missing or reordered
records, changed inputs, and post-run sample changes fail closed. Full upstream
samples and manifests remain authenticated provenance; the acceptance result is
replayed from raw responses.

This is a reproducible integration and regression-policy demonstration, not a
general model-quality ranking. Production use should select datasets and
thresholds that represent the intended deployment.
