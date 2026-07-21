# LM Evaluation Harness

This example imports complete per-record output from
[LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness),
then runs `invarlock evaluate`, `invarlock verify`, and `invarlock report`.
InvarLock recomputes the exact-match comparison from the paired records. It
does not trust the harness aggregate score.

The maintained journey compares the public, revision-pinned
`Qwen/Qwen3-0.6B-Base` checkpoint with the public post-trained
`Qwen/Qwen3-0.6B` checkpoint. Every snapshot file is checked against a fixed
byte length and SHA-256 before execution. The 102-record schedule carries stable
IDs and fixed prompts and targets; both upstream runs execute offline after the
snapshot and image downloads.

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

Pass `--workspace PATH` when you want to retain the transaction at a specific
new path. Otherwise the launcher creates a temporary workspace and prints its
location.

The command builds the repository's source-authenticated CPU runtime and adds
the pinned optional `lm-eval` dependency. It needs roughly 7 GB of temporary
disk for the two Qwen3 snapshots, runtime images, and outputs. Both model runs
execute without network access inside that derived image. The inspected
immutable image ID is bound into both runtime receipts before the import
transaction is signed.

The adapter requires one upstream sample for each schedule record, stable
dataset IDs, matching prompt and target hashes, the exact task configuration,
and a digest-bound run manifest. Aggregate-only result files, missing IDs,
reordered records, source-input changes, and post-run sample changes fail
closed. The harness provenance is also attached as an authenticated
observation, while acceptance is replayed from the raw responses. Every target
fits the authenticated one-token generation bound, and the completed journey
requires each model to solve at least 20% of the fixed records.

The schedule covers factual, numeric, temporal, spatial, scientific, and
common-language completions. Its fixed policy requires all 102 records, limits
the paired 95% confidence-interval width to 20 percentage points, and rejects
a regression larger than 20 percentage points.

This is a small integration demonstration, not a model-quality benchmark. Use
a representative pinned dataset and reviewed policy for a production claim.
