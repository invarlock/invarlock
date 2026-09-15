# K2 Horizon 32B judge pilot reference

The corrected retained pilot contains **480 complete judge trials out of 480**:
40 frozen cases per workflow, two answers per case and three ratings per answer.
Both grounded QA and slot extraction remain **`insufficient_evidence`**. Their
40-unit effect intervals have width `1.007489336443204`, exceeding the unchanged
maximum of `1`. Complete collection did not make either policy pass.

The original attempt is retained separately: all 480 recorded trials remain,
with 203 complete QA trials and 227 complete extraction trials, or 430 complete
in total. No original responses, incomplete attempts or old evidence bytes were
removed or rewritten. The corrected run is a new collection under explicit
reasoning effort `none`, not a repair of the historical measurements.

## What is retained

[reference.zip](reference.zip) is a deterministic archive with sorted members,
fixed ZIP timestamps and a physical SHA-256/byte-size inventory in
`reference.json`. [archive.json](archive.json) pins the transport and manifest.

- `corrected/{grounded_qa,extraction}/` contains the exact collected measurement
  file, original plan, runs, analysis policy and collection configuration,
  signed evidence, recipient policy, signed verification receipt and reports.
- `corrected/runtime-source.json` records source commit
  `351cbc8c242a477e6c6f733b6a293f88018238bd`, runtime source-file hashes and frozen
  input hashes. `execution.json` retains the observation timestamps and
  265.454-second combined collection duration. `usage.json` retains observed
  token usage and explicitly labelled cost estimates.
- `historical/{grounded_qa,extraction}/` preserves the prior collected
  measurements, protocol inputs and unsigned evidence. Its recorded installed
  runtime replay is retained in `historical/retention-status.json`.
- `attribution/` retains the original SQuAD 2.0 and Schema-Guided Dialogue source
  notices, publisher READMEs, licenses and dataset source pins.

The corrected plan requested `openai/gpt-5.6-sol`, with approved returned model
`gpt-5.6-sol`, no tools, one attempt, zero SDK retries, 128 output tokens per call
and explicit reasoning effort `none`. The pinned installed collector requested
standard processing. The 480 trials contain no collector errors, timeouts or
`max_tokens` finishes. Observed usage was 260,271 input and 3,868 output tokens,
with no reported cached input or reasoning tokens. The 265.454-second observation
is one run, not a service level or forecast for the larger final study.

## Replay without model calls

Install the core InvarLock implementation from the pinned source revision or a
compatible later revision. The reference uses development contracts beyond the
historical `v0.15.0` release; do not infer compatibility from the package version
alone. No Inspect package, provider SDK, model, GPU, credential or signing key
is needed for replay.

Obtain the archive pin from an independently trusted copy of this repository,
then run from the source checkout with its core package installed:

```bash
python examples/judge_measurements_pilot_reference.py \
  --bundle examples/judge-measurements/references/k2-32b-pilot/reference.zip \
  --expected-sha256 44966ad7be58b0f0ebe8ca3054cd92c9321ff366c4801e53032005eef60d2d58
```

The command checks every physical file pin, authenticates the corrected signed
receipts and replays both corrected packs through the existing core verifier.
Successful reference validation exits zero while returning `verified: true`,
`accepted: false` and `decision: insufficient_evidence` for each workflow.
The public `invarlock verify` command returns exit **7** for those retained
negative policy results. It must not be interpreted as failed signature replay.

The included recipient policies and verifier identities demonstrate reproducible
trust checks. They gain authority for this reference only through the
independently obtained archive pin; copying keys or expectations from an
arbitrary submitted package is not independent authentication. The policies
require a pass, while the retained analyses are advisory (`required: false`).
Accordingly the verifier also states that advisory evidence cannot satisfy a
required recipient decision.

The prior envelopes are unsigned and use older contracts. They are preserved as
historical observations, not reissued as current signed evidence. A separately
recorded comparison matched the installed historical runtime to source
`08430477203459b46f6f76d99ad3d9d08cd5dbe1`; this does not authenticate the original
live evaluator, whose launcher used a mutable checkout without a source pin.
Current replay does not rerun that historical environment or promote its result.

## Claim limits

This pilot measures the judge's ratings under the frozen rubric. It does not
establish agreement with independent reference labels, rubric validity, judge
accuracy, inter-rater reliability, broad model quality or deployment approval. No policy,
case membership, rubric or answer was relaxed to obtain a passing outcome.
Hosted judge identity does not reveal or authenticate its model weights.
Verification reconstructs signed recorded results; it does not repeat model
execution or independently attest the original provider execution.

This archive contains judge outcomes and answer roles. **Do not send it to a
blinded reviewer before their judgments are frozen.** Give that reviewer only
the separately prepared rubric-development sheets described in the
[frozen-answer reference](../k2-32b/README.md). No reference-rating sheets,
final-validation holdout, native role map, private keys or operator logs are included here.
Preserve the original dataset terms when redistributing the retained text;
SQuAD's software MIT license does not relicense all Wikipedia-derived dataset
material, and SGD-derived data retains CC BY-SA 4.0.

## Subsequent comparison

The [completed Luna held-out reference](../k2-32b-luna-xhigh-heldout/README.md)
contains separate final measurements and a reference-label comparison. The pilot
archive retains its original study metadata and insufficient-evidence outcome.
