# K2 Horizon 32B Luna xHigh judge pilot reference

This retained pilot contains 480 complete judge trials: 40 frozen cases per
workflow, two answers per case and three ratings per answer. Both grounded QA
and slot extraction are `insufficient_evidence` because their 40-unit effect
intervals have width `1.007489336443204`, above the frozen maximum of `1`.
These are advisory policy results for the pilot schedule.

Use this reference to compare two judge configurations on identical frozen
answers and to replay an authentic but inconclusive result. Grounded QA concerns
answers supported by supplied context; extraction concerns requested slot
values. The 480 ratings come from only 80 cases across both workflows, so the
rating count should not be read as the independent sample size.

## What is retained

[reference.zip](reference.zip) is a deterministic archive with sorted members,
fixed ZIP timestamps and a physical SHA-256/byte-size inventory in
`reference.json`. [archive.json](archive.json) pins the transport and manifest.

- `pilot/{grounded_qa,extraction}/` contains the collected measurements,
  protocol inputs, signed evidence, recipient policy, signed verification
  receipt, replayed analysis and reports.
- `summaries/` contains observed token use, signed-replay status and a
  descriptive comparison with the separately retained Sol pilot.
- `provenance/` records source commit
  `adb3c77d0af3e351b6dfe325959660deb5543e80`, the 390.745-second observation,
  163 runtime source-file pins, post-collection integrity checks and the
  analysis-policy binding correction.
- `attribution/` retains the original SQuAD 2.0 and Schema-Guided Dialogue
  notices, publisher READMEs, licenses and source pins.

The pilot requested `openai/gpt-5.6-luna`, with returned model `gpt-5.6-luna`,
xHigh reasoning effort, no tools, one attempt, zero SDK retries, 25,000 maximum
completion tokens and the standard service tier. Every call ended normally with
a parseable rating; all 480 provider response identifiers are distinct. Observed
usage was 260,271 input tokens and 72,467 output tokens, including 65,641
reasoning tokens. Token-based cost estimates are $0.139015 to $0.152028; the
provider invoice was not checked. The observed duration and cost are one
campaign, not a service level, price guarantee or forecast for the final study.

The initially copied analysis policy still named the Sol plan digest. Before
analysis or signing, a Luna-specific policy was derived by changing only
`plan_sha256`; all thresholds and statistical methods remained identical.
The active policy, the original Sol-bound policy and a pin-level correction
record are all retained. This binding correction made no model calls and did not
change measurements, cases, answers, rubrics or decision thresholds.

## Descriptive Sol comparison

The Luna and retained Sol configurations rated the same 80 cases and 160
answers with the same rubrics and repetition schedule. Luna used xHigh reasoning;
Sol used none, so this comparison covers the complete configurations rather than
model names alone.

| Measure | Grounded QA | Extraction |
| --- | ---: | ---: |
| Identical repetition-paired labels | 184/240 (76.7%) | 240/240 (100%) |
| Incorrect-versus-correct disagreements | 16/240 | 0/240 |
| Answers with identical mean score | 47/80 | 80/80 |
| Answers with varying Luna repetitions | 28/80 | 0/80 |
| Answers with varying Sol repetitions | 2/80 | 0/80 |

Repetitions are clustered within answers, and answer pairs are clustered within
cases; the 240 ratings per workflow are not independent tasks. Sol ratings are
not an independent correctness standard. Matching Sol does not establish
accuracy, while disagreement does not establish that Luna is wrong. The
configurations ran in separate
observation windows under a rate-limited collector, so their durations are not
a general latency comparison.

## Replay without model calls

You need Python 3.12 or newer, the core InvarLock package and the checked-in
archive. No optional collector, API key, model or GPU is needed. The helper
uses temporary extraction and prints JSON without changing the archive.

Use the helper from this repository revision with the core InvarLock
implementation installed from collector source
`adb3c77d0af3e351b6dfe325959660deb5543e80` or a compatible later revision.
Obtain the archive pin from an independently trusted copy of this repository,
then run from the checkout root:

```bash
python examples/judge_measurements_pilot_reference.py \
  --bundle examples/judge-measurements/references/k2-32b-luna-xhigh-pilot/reference.zip \
  --expected-sha256 4d8f50e1cba0056d2118695a4dab73cce4a5ab10829320e2f8ea4b0c48d0e766
```

The command checks the archive and every physical member pin, authenticates both
signed receipts and replays the evidence through the core verifier. Successful
reference validation exits zero with `verified: true`, `authenticated: true`,
`replayed: true`, `accepted: false` and `decision: insufficient_evidence` for
both workflows. A required recipient cannot accept the retained advisory
evidence. The independently obtained archive pin gives authority to the included
public-key anchors; keys copied from an untrusted submitted package would not.

Both workflow entries should show successful verification and rejected
acceptance. The helper exits zero because those are the retained expected
outcomes. A nonzero helper exit means an archive, signature, binding or replay
check failed. Preserve that error and recover the approved inputs rather than
changing the retained measurements or their policy.

## Claim limits

This pilot measures the judge's ratings under the frozen rubric. It does not
establish agreement with independent reference labels, judge accuracy, rubric
validity, inter-rater reliability, broad model quality, model equivalence or
deployment approval.
Hosted model identity does not reveal or authenticate model weights. Replay
checks the recorded signed evidence and does not repeat or independently attest
the provider execution.

The archive contains judge outcomes and answer roles. Keep it from blinded
reviewers until their labels are frozen; provide only the separate blinded review
sheets in the [frozen-answer reference](../k2-32b/README.md). No reference-rating
sheets, final-validation holdout, private keys, credential material, operator
authorization, collector console logs or checkpoint files are included.
Preserve the original dataset terms when redistributing retained text; SQuAD's
software MIT license does not relicense all Wikipedia-derived dataset material,
and SGD-derived data remains under CC BY-SA 4.0.

## Subsequent comparison

The [completed Luna held-out reference](../k2-32b-luna-xhigh-heldout/README.md)
contains separate final measurements and a reference-label comparison. The pilot
archive retains its original study metadata and insufficient-evidence outcome.
