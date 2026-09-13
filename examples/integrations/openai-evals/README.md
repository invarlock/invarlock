# OpenAI Evals

This example runs the pinned OpenAI Evals `basic.Match` evaluator over two real,
locally generated Qwen3.5 evaluations, then runs `invarlock evaluate`,
`invarlock verify`, and `invarlock report`. InvarLock recomputes the exact-match
comparison from the paired records and does not trust the evaluator aggregate
score. The end-to-end qualification row remains pending until a clean OCI run
is retained.

The maintained journey compares the public, revision-pinned
`Qwen/Qwen3.5-0.8B-Base` checkpoint with the public post-trained
`Qwen/Qwen3.5-0.8B` checkpoint. Every snapshot file is checked against a fixed
byte length and SHA-256 before execution. The 102-record schedule carries stable
IDs and fixed prompts and targets; both upstream runs execute offline after the
snapshot and image downloads.

The curated snapshots contain the pinned weights, model configuration, and
tokenizer files required by the run. Optional checkpoint generation defaults
are excluded, so the task owns the sampling mode, newline stop, and
authenticated one-token limit. Every run manifest records the evaluator,
scorer, fixed CPU/float32 profile, batch size 8, seed, disabled remote code, and
per-record output digests.

## Capture an existing OpenAI Evals workflow

The command below is the example-owned signed OCI bridge. An existing workflow
can instead map original stable case IDs, inputs, references and answers through
`invarlock.engine.capture_evaluator_run` and use
[captured evaluation](../../captured-results/README.md). This canonical path
does not depend on a dedicated installed OpenAI Evals export parser.

Select InvarLock exact match for string answers and references. A judge recipe
can use frozen task/answer text and complete retained calls; a native Match
score or an existing scalar rating is insufficient. Use an explicit projection
for structured task inputs, and opt in to `prompt.reference_mode: per_case` when
the judge should receive each string reference separately from the model input.

Normalized NLL requires an actual reference-continuation measurement with the
closed likelihood facts and identity bindings. The Match workflow described
here produces generated answers and match events, so it supplies no such
likelihoods. The [shortlist capture guide](../../evaluator-qualification/maintained/CAPTURE.md)
distinguishes canonical compatibility tests and the separate real Harness NLL
reference from this historical exact-match qualification.

## Run the signed OCI integration

From a clean committed checkout with Docker or Podman available:

```bash
make example-openai-evals EXAMPLE_ARGS="--evidence-signing-key /secure/keys/evidence.pem --verifier-signing-key /secure/keys/verifier.pem --builder-signing-key /secure/keys/builder.pem --builder-public-key /secure/keys/builder-public.pem --trust-root /secure/trust/openai-evals"
```

Pass `EXAMPLE_ARGS="--workspace PATH"` when you want to retain the transaction
at a specific new path. The evidence and verifier keys must be caller-owned and
outside the transaction; the trust root is created outside the transaction.

The command builds the source-authenticated CPU runtime and adds
`evals==3.0.1.post1+invarlock.match.1`, derived from the hash-pinned upstream
3.0.1.post1 wheel. Every upstream code file is unchanged; package metadata
removes the unused NLTK dependency and records the derived version. This image
supports `basic.Match` with the repository's HF completion adapter. NLTK-based
suites and data-generation helpers are unsupported; their upstream modules
can fail on missing optional dependencies.
The broader qualification profile and historical evidence keep their original
dependency identities.

The journey needs roughly 7 GB of temporary disk for the two
Qwen3.5 snapshots, runtime images, and outputs. Both model runs execute without
network access inside the inspected image. The immutable image ID is bound into
both runtime receipts before the import transaction is signed.

The adapter requires one upstream sample for each schedule record, stable IDs,
matching prompt and target hashes, the exact scorer and generation
configuration, and a digest-bound run manifest. Missing IDs, reordered records,
source-input changes, output changes, inconsistent native match events, and
post-run sample changes fail closed. The pinned `basic.Match` evaluator uses
prefix matching; the native result is retained and checked, while the signed
transaction deliberately replays byte-exact equality through InvarLock's
strict metric. Evaluator provenance is attached as an authenticated
observation, while acceptance is replayed from the raw responses.

The fixed policy requires all 102 records, limits the paired 95% confidence
interval width to 20 percentage points, rejects a regression larger than 20
percentage points, and requires each model to solve at least 20% of the corpus.

This is a small integration demonstration, not a model-quality benchmark.
