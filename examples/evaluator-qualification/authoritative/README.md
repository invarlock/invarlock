# Authoritative evaluator imports

This directory retains one real 102-record model evaluation and complete
per-record imports for every maintained deterministic evaluator profile. It
demonstrates a deeper evidence level than the small qualification corpus. The
demonstration-level manifest identifies profiles that also maintain an
end-to-end signed transaction over the same evaluator-neutral boundary.

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
claim coverage of every evaluator capability or imply that authoritative import
alone completes a signed InvarLock transaction. End-to-end status is recorded
separately as evidence maturity, not an architectural limitation on other
profiles.
