# Independently replayable evaluator imports

This directory retains one real 102-record model evaluation and complete
per-record imports for the deterministic evaluator profiles in the retained
matrix. Use it to check that different evaluators' exact-match outputs can be normalized and
independently recomputed without rerunning the model. The directory name does
not make its files a recipient trust authority: these are retained examples,
and signed model transactions are a separate evidence set.

The corpus was produced by the immutable `Qwen/Qwen3.5-0.8B` revision and fixed
one-token, greedy CPU generation settings recorded in `cases.json`. The source
dataset has 102 fixed causal-completion records. The retained model outputs contain
61 exact matches and 41 mismatches.

For each deterministic evaluator,
`artifacts/<profile-id>/` contains:

- the digest-bound qualification profile;
- the real upstream evaluator output over all 102 model records;
- the normalized qualification export and result;
- complete runtime-import JSONL; and
- an import-replay result binding the qualification, runtime schedule, source
  model execution, and imported records.

## Choose what to run

Run commands from the matching repository root with Python 3.12 or newer and
the core development dependencies from [CONTRIBUTING](../../../CONTRIBUTING.md).
Offline replay needs no model weights, GPU, evaluator SDK or provider credentials.

Start with the network-free check of the retained profiles, exports and imported
records:

```bash
make evaluator-replayable-imports
```

A successful exit means the retained records reproduce their declared
exact-match scores and bindings, including the 61 matches and 41 mismatches.
A digest, score or schedule mismatch fails the check. This command verifies
the saved imports; it does not publish new signed evidence.

To execute the historical evaluator packages and CLIs again over the same saved
model outputs, use the following command. It requires `uv`, Node.js with `npx`,
and network access on a cold dependency cache. It refreshes retained
qualification artifacts in the checkout, so use a disposable checkout if you
only want to explore the behavior:

```bash
make evaluator-upstream-qualification
```

To regenerate the model outputs themselves, first make the pinned model snapshot
available locally. This command uses `uv`, the repository's locked Hugging Face
dependencies and CPU inference with model-network access disabled. It compares
the generated outputs with the retained corpus instead of silently replacing it:

```bash
make evaluator-replayable-corpus
```

For fresh evaluations under current literal-pair profiles, follow the
[qualification reference](../../../docs/reference/evaluator-qualification.md).
For your existing case data, use the [capture guide](../maintained/CAPTURE.md).

This layer demonstrates complete replayable exact-match imports. It does not
claim coverage of every evaluator capability or imply that replayable import
alone completes a signed InvarLock transaction. See the separate
[signed transactions](../signed-transactions/README.md) for retained native
execution and independently signed verification results.
