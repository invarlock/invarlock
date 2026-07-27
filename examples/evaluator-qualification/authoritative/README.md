# Authoritative evaluator imports

This directory retains one real 102-record model evaluation and seventeen complete
per-record evaluator imports. It demonstrates a deeper evidence level than the
nineteen-profile qualification matrix. LM Evaluation Harness currently
adds an end-to-end signed transaction over the same evaluator-neutral boundary.

The corpus was produced by the immutable `Qwen/Qwen3-0.6B` revision and fixed
one-token, greedy CPU generation settings recorded in `cases.json`. The source
dataset has 102 fixed causal-completion records. The retained model outputs contain
52 exact matches and 50 mismatches.

For each deterministic evaluator,
`artifacts/<profile-id>/` contains:

- the digest-bound qualification profile;
- the real upstream evaluator output over all 102 model records;
- the normalized qualification export and result;
- complete runtime-import JSONL; and
- an import-replay result binding the qualification, runtime schedule, source
  model execution, and imported records.

Run the network-free verification:

```bash
make evaluator-authoritative-imports
```

Re-execute the evaluator packages and CLIs:

```bash
make evaluator-upstream-qualification
```

Regenerate the model outputs with the repository's locked Hugging Face
dependencies and compare them with the retained corpus:

```bash
make evaluator-authoritative-corpus
```

This layer demonstrates complete replayable exact-match imports. It does not
claim coverage of every evaluator capability, and it does not claim that all
seventeen profiles currently complete a signed InvarLock transaction. LM Evaluation
Harness is the maintained example at that level in this revision; this is a
maturity status, not an architectural limitation on the other profiles.
