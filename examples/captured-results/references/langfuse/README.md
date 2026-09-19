# Langfuse experiment replay reference

> **Surface:** Offline Langfuse Python SDK experiment exports.
>
> **Stability:** Langfuse 4.14.1 and `invarlock/langfuse-export-v1`.
>
> **Use this page when:** Checking exact-match and normalized likelihood imports
> from retained model measurements through Langfuse experiments.

The four exports retain complete results from real
`Langfuse.run_experiment` calls with tracing disabled. Each has 400 records.
The task callback returns an existing recorded output; it makes no new model,
judge, API or GPU calls. These are real SDK executions over retained model
measurements, not new model inference or hosted Langfuse service qualification.

`exact_match-*.json` replays the retained
[Mistral 7B HTTP outputs](../../../hosted-service/references/mistral-7b-http/README.md).
`normalized_nll-*.json` replays the retained
[Mistral 7B likelihood measurements](../mistral-7b-likelihood/README.md).
Original source-file hashes and run identity bindings are recorded in
`reference.json`. Original signed evidence remains at those references.
The original policies produce exact-match acceptance and likelihood regression.

Likelihood values remain explicit per-record capture facts in
`item.metadata.invarlock_likelihood`, including token counts, byte counts,
model, tokenizer, configuration, input and reference bindings. They are never
inferred from Langfuse evaluation scores. The canonical likelihood source names
Langfuse as the replay exporter. The complete original likelihood object,
including its original evaluator source, remains in
`item.metadata.invarlock_original_likelihood`. All numerical and other identity
bindings remain unchanged. Original string case metadata is
preserved; replay-only metadata uses the reserved `invarlock_` prefix.

These files contain no judge measurements. The Langfuse judge journey tests
use separately identified synthetic full measurements to verify signed import
and independent replay. A scalar judge score cannot replace a complete judge
measurement document or establish that a model judge ran.

## Reproduce the exports

Install a candidate InvarLock wheel and Langfuse 4.14.1 in an isolated Python
environment. From the matching source checkout, run:

```bash
python examples/captured-results/references/langfuse/capture.py \
  --output /tmp/langfuse-replay
```

The destination must not exist. The script reads the two references already
in this checkout and writes new exports and source hashes. SDK-generated run
names and experiment IDs differ between executions. It never replaces the
retained exports or original evidence.

For installed-package qualification, install the candidate wheel in a separate
recipient environment, then run the tests with both interpreters specified:

```bash
INVARLOCK_REQUIRE_LANGFUSE_SDK=1 \
INVARLOCK_LANGFUSE_WHEEL_PYTHON=/path/to/recipient/bin/python \
python -m pytest tests/examples/test_langfuse_export.py \
  tests/integration/test_langfuse_sdk.py
```

The maintained `make langfuse-sdk-test` target builds the distributions and runs
both test files in hash-locked SDK and recipient environments. CI requires it
on Python 3.12 and 3.13; the examples coverage shard also requires the SDK tests.

The test interpreter requires Langfuse 4.14.1 and repository test dependencies.
The recipient interpreter uses the installed wheel with source imports disabled;
it does not require Langfuse. Tests evaluate, independently verify, and render
both JSON and HTML reports for exact-match, normalized NLL and synthetic judge
imports. An intentional likelihood policy regression remains a verified
regression, not a successful policy decision.
