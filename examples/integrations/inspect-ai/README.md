# Inspect AI

This example runs a native Inspect AI Task with its exact-match scorer over two
real, locally generated Qwen3 evaluations, then runs `invarlock evaluate`,
`invarlock verify`, and `invarlock report`. InvarLock recomputes the exact-match
comparison from the paired records and does not trust the evaluator aggregate
score. A clean CPU-only OCI run of this journey is retained as the Inspect AI
signed-transaction demonstration in the qualification matrix.

The maintained journey compares the public, revision-pinned
`Qwen/Qwen3-0.6B-Base` checkpoint with the public post-trained
`Qwen/Qwen3-0.6B` checkpoint. Every snapshot file is checked against a fixed
byte length and SHA-256 before execution. The default 102-record schedule
carries stable IDs and fixed prompts and targets; both upstream runs execute
offline after the snapshot and image downloads.

The curated snapshots contain the pinned weights, model configuration, and
tokenizer files required by the run. Optional checkpoint generation defaults
are excluded, so the task owns the sampling mode, newline stop, and
authenticated one-token limit. Every run manifest records the evaluator,
scorer, fixed CPU/float32 profile, batch size 8, seed, disabled remote code, and
per-record output digests.

From a clean committed checkout with Docker or Podman available:

```bash
make example-inspect-ai EXAMPLE_ARGS="--evidence-signing-key /secure/keys/evidence.pem --verifier-signing-key /secure/keys/verifier.pem --builder-signing-key /secure/keys/builder.pem --builder-public-key /secure/keys/builder-public.pem --trust-root /secure/trust/inspect-ai"
```

Release qualification uses the larger flagship profile:

```bash
make example-inspect-ai EXAMPLE_ARGS="--corpus-profile flagship --evidence-signing-key /secure/keys/evidence.pem --verifier-signing-key /secure/keys/verifier.pem --builder-signing-key /secure/keys/builder.pem --builder-public-key /secure/keys/builder-public.pem --trust-root /secure/trust/inspect-ai"
```

The flagship profile derives 400 records from an immutable
`EleutherAI/lambada_openai` test source. It freezes the source revision and
hash, deterministic stratified selection, selected indices, and derived JSONL
hash. Every selected final-word target is one losslessly decoded token under
both pinned Qwen tokenizers. Use `--benchmark-source PATH` to prepare from a
previously downloaded copy with the same byte and hash checks. This is a
paired generative exact-match projection, distinct from the standard
log-likelihood LAMBADA metric.

Pass `EXAMPLE_ARGS="--workspace PATH"` when you want to retain the transaction
at a specific new path. The evidence and verifier keys must be caller-owned and
outside the transaction; the trust root is created outside the transaction.

The command builds the source-authenticated CPU runtime and adds a hash-pinned
Inspect AI environment. It needs roughly 7 GB of temporary disk for the two
Qwen3 snapshots, runtime images, and outputs. Both model runs execute without
network access inside the inspected image. The immutable image ID is bound into
both runtime receipts before the import transaction is signed.

The adapter requires one upstream sample for each schedule record, stable IDs,
matching prompt and target hashes, the exact scorer and generation
configuration, and a digest-bound run manifest. Missing IDs, reordered records,
source-input changes, output changes, evaluator-score disagreement, and
post-run sample changes fail closed. Inspect's pinned HF decoder removes the
leading whitespace of a causal completion, so the native task uses an
authenticated raw-text chat template and the bridge records the explicit
target-leading-whitespace recovery before strict replay. Evaluator provenance
is attached as an authenticated observation, while acceptance is replayed from
the recovered raw responses.

The quick policy requires all 102 records, limits the paired 95% confidence
interval width to 20 percentage points, and requires 20% accuracy on each
side. The flagship policy requires all 400 records, tightens the interval-width
limit to 10 percentage points, and requires 5% accuracy on each side. Both
profiles reject a regression larger than 20 percentage points.

This is a small integration demonstration, not a model-quality benchmark.
