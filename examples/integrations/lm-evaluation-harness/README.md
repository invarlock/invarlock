# LM Evaluation Harness

This example imports complete per-record output from
[LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness),
then runs `invarlock evaluate`, `invarlock verify`, and `invarlock report`.
InvarLock recomputes the exact-match comparison from the paired records. It
does not trust the harness aggregate score.

The maintained journey compares the public, revision-pinned
`Qwen/Qwen3-0.6B-Base` checkpoint with the public post-trained
`Qwen/Qwen3-0.6B` checkpoint. Every snapshot file is checked against a fixed
byte length and SHA-256 before execution. The default 102-record schedule
carries stable IDs and fixed prompts and targets; both upstream runs execute
offline after the snapshot and image downloads.

The curated snapshots contain the pinned weights, model configuration, and
tokenizer files required by the run. Optional checkpoint generation defaults
are excluded, so the task owns the sampling mode, newline stop, and
authenticated one-token limit. Vocabulary and special-token behavior remain
bound to each checkpoint. Every run manifest records the Harness model and
backend, fixed CPU/float32 profile, batch size 8, seed, and disabled remote code
alongside the task and per-record output digests.

From a clean committed checkout with Docker or Podman available:

```bash
make example-lm-evaluation-harness
```

Supply caller-owned signing keys and a new trust root when running the signed
journey, for example:

```bash
make example-lm-evaluation-harness EXAMPLE_ARGS="--evidence-signing-key /secure/keys/evidence.pem --verifier-signing-key /secure/keys/verifier.pem --builder-signing-key /secure/keys/builder.pem --builder-public-key /secure/keys/builder-public.pem --trust-root /secure/trust/lm-evaluation-harness"
```

Release qualification uses the larger flagship profile:

```bash
make example-lm-evaluation-harness EXAMPLE_ARGS="--corpus-profile flagship --evidence-signing-key /secure/keys/evidence.pem --verifier-signing-key /secure/keys/verifier.pem --builder-signing-key /secure/keys/builder.pem --builder-public-key /secure/keys/builder-public.pem --trust-root /secure/trust/lm-evaluation-harness"
```

The flagship profile derives 400 records from the immutable
`EleutherAI/lambada_openai` test source. It freezes the source revision, source
hash, selection seed, four prompt-length strata, selected indices, and derived
JSONL hash. A selected target must be one losslessly decoded token under both
pinned Qwen tokenizers. The source can be supplied from a previously downloaded
file with `--benchmark-source PATH`; the same byte length and SHA-256 checks
apply. This generative exact-match projection uses LAMBADA's final-word
boundary, while the retained score remains InvarLock's paired generation
comparison rather than the standard log-likelihood LAMBADA metric.

Pass `--workspace PATH` when you want to retain the transaction at a specific
new path. Otherwise the launcher creates a temporary workspace and prints its
location.

The launcher removes the exact temporary base and Harness image tags it creates
after the journey, including when the workspace is retained.

The command builds the repository's source-authenticated CPU runtime and adds
a deterministic cache-free package derived from the hash-pinned upstream
`lm-eval` 0.4.12 wheel. The integration never reads or writes Harness response
caches, and its image contains neither the vulnerable `sqlitedict` package nor
a cache entry point. It needs roughly 7 GB of temporary
disk for the two Qwen3 snapshots, runtime images, and outputs. Both model runs
execute without network access inside that derived image. The inspected
immutable image ID is bound into both runtime receipts before the import
transaction is signed.

The adapter requires one upstream sample for each schedule record, stable
dataset IDs, matching prompt and target hashes, the exact task configuration,
and a digest-bound run manifest. Aggregate-only result files, missing IDs,
reordered records, source-input changes, and post-run sample changes fail
closed. Completion reruns both workers in the inspected image; prepared worker
output is not authoritative. The full upstream per-record snapshots and
manifests are attached as authenticated provenance, while acceptance is
replayed from the raw responses. Every target fits the authenticated one-token
generation bound. The signed policy applies the profile-specific side-accuracy
floor described below.

The quick schedule covers factual, numeric, temporal, spatial, scientific, and
common-language completions. Its fixed policy requires all 102 records and
limits the paired 95% confidence-interval width to 20 percentage points. The
flagship policy requires all 400 records, tightens that interval-width limit to
10 percentage points, and requires at least 5% accuracy on each side. Both
profiles reject a regression larger than 20 percentage points.

The retained 400-record flagship transaction achieved an 8.39-percentage-point
paired interval width under its 10-point maximum.

This is a reference integration demonstration, not a model-quality benchmark.
Use a representative pinned dataset and reviewed policy for a production claim.
