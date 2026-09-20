# K2 Horizon 32B Luna xHigh held-out reference

This reference retains all 10,260 judge ratings from 1,710 frozen cases: 422
grounded-QA cases and 1,288 extraction cases, with two answers per case and
three ratings per answer. Both comparisons meet their policies declared in advance.
Neither establishes that the subject is better or that the judge is generally
accurate.

Use this reference to replay a complete measured judge campaign and inspect why
its two comparison policies pass. The answers were already frozen before
judging. Grounded QA rates answers against supplied context; extraction rates
the requested slot values. The source cases were kept separate from the pilot
used to develop the rubric, which is what “held-out” means here.

For the commands, go to [retained files and replay](#retained-files-and-replay).
For interpretation, read each effect interval together with its allowed
degradation and the absolute subject-score requirement below.

## Task and proposed change

The practical change is a system instruction for stricter output-format
compliance on the same `IFM/K2-Horizon-32B` checkpoint, revision
`466db5f23c8a7c96b0b320b688612ee6f4446a35`. Baseline receives the user task alone.
Subject adds a system instruction to follow the task and its format exactly,
without extra commentary, Markdown fences or additional fields. The user text,
checkpoint and native generation settings stay the same.

Grounded QA asks for a supported source span or `NO_ANSWER` from supplied context.
Extraction asks for directly mentioned slot/value pairs from the current user
utterance, returned as a raw JSON array or `[]`. These are useful components of
context-based answering and structured-data pipelines. The
[frozen-answer reference](../k2-32b/README.md) retains the original task text,
answers and change mappings.

The result supports accepting this prompt change under the declared fixed
benchmark policies. It does not establish that the change improves quality or
meets a different application's requirements. Start with extraction for a
concrete example of schema-based extraction; use QA to explore context support and
answerability.

## Results and interpretation

| Measure | Grounded QA | Extraction |
| --- | ---: | ---: |
| Complete ratings | 2,532/2,532 | 7,728/7,728 |
| Independent source units | 422 | 1,288 |
| Mean baseline score | 0.79581 | 0.89454 |
| Mean subject score | 0.82267 | 0.90386 |
| Mean subject-minus-baseline score | +0.02686 | +0.00932 |
| Paired effect interval | −0.12823 to +0.18195 | −0.07946 to +0.09809 |
| Allowed degradation | 0.15 | 0.10 |
| Subject score lower bound | 0.74512 | 0.85947 |
| Required subject lower bound | 0.60 | 0.60 |
| Policy result | Pass | Pass |

The rubric assigns 0 for incorrect, 0.5 for partially correct and 1 for correct.
Its mean is not exact-match accuracy. The allowed degradation is an absolute
change on that scale; the 0.60 requirement applies to the lower bound on mean
subject score. Maximum interval widths are 0.32 for QA and 0.18 for extraction.
These thresholds were chosen as fixed benchmark requirements before collection;
they are not application-specific error budgets. Choose requirements suited to
your intended use before collecting a new comparison.

The frozen Hoeffding analysis uses alpha 0.05 and comparison-family size four.
Repetitions within answers do not increase the independent unit count. The paired intervals
include zero; passing the allowed-degradation gate does not prove improvement.
The result applies to the fixed benchmark under the declared independence and
judging assumptions, not representative production traffic.

The judge was `openai/gpt-5.6-luna`, returned as `gpt-5.6-luna`, with xHigh
reasoning, standard processing, no tools, no SDK retries and one attempt per
rating. Its output allowance was 10,000 tokens, compared with 25,000 in the
separate Luna pilot; its input reservation was 4,352 tokens. All response IDs
are distinct and all ratings completed. The source commit is
`adb3c77d0af3e351b6dfe325959660deb5543e80`.

Collection took 10,973.169 seconds. Observed token use implies a standard-rate
estimate of $2.986760 and conservative cache-write accounting of $3.286154.
The invoice was not checked. These are observed campaign estimates, not future
price or latency guarantees.

## Reference-label comparison

Frozen review labels cover 160 held-out answers per workflow under the declared
rubrics. The archive retains those labels, notes, label-source provenance and
confusion counts. Exact agreement was 334/480 (69.6%) for
QA and 463/480 (96.5%) for extraction. Each answer has three Luna ratings;
these denominators are not 480 independent tasks.

These counts measure agreement with the frozen reference labels, not general
judge accuracy. Disagreements and label-source provenance remain available in
the archive, including `review/manifest.json`.
QA review disregarded surrounding whitespace for judgment only; retained answer
bytes did not change. Pilot review and the held-out comparison remain separate.

## Retained files and replay

You need Python 3.12 or newer, the core InvarLock package, the matching helper and
the checked-in archive. No collector package, API key, model or GPU is needed.
Run from the repository root. Replay uses temporary extraction and reads a larger
archive than the pilot examples, so allow space for the expanded files below.

The archive contains the original signed evidence, recipient policies and signed
receipts for both workflows; reports; collection settings; complete usage and
agreement summaries; frozen runtime source pins; split checks; and dataset
attribution. Measurements are retained once inside each evidence directory.
Pilot and held-out cases and source clusters do not overlap.

The ZIP has 47 members and expands to 216,183,325 bytes. The replay helper's
legacy 64 MiB expansion default is intentionally unchanged. Supply the exact
expanded inventory size for this larger reference:

```bash
python examples/judge_measurements_pilot_reference.py \
  --bundle examples/judge-measurements/references/k2-32b-luna-xhigh-heldout/reference.zip \
  --expected-sha256 19574e2f68f00e06f13432369d99818e625b58d10bf53156bebf86b538409242 \
  --max-expanded-bytes 216183325
```

Use the core implementation from the source above or a compatible later build.
The command authenticates the independently obtained archive pin, physical
member inventory, signed receipts and replayed decisions without model calls.
It returns `verified: true`, `authenticated: true`, `replayed: true`,
`accepted: true` and `decision: pass` for both workflows. Included keys have
reference authority only through the independently trusted archive pin.

Successful reference validation exits zero and prints those results as JSON.
A nonzero exit indicates a validation failure, not a request to recollect ratings.
Preserve the error and check the archive pin, installed implementation and size
allowance. Do not regenerate or edit historical evidence to make replay pass.

Replay checks retained evidence, not fresh provider execution or hidden weights.
The archive includes no private keys, credentials, execution authorization,
checkpoint directories or local filesystem paths. Source provenance is a selected
metadata record. Original selection mappings let readers reproduce the agreement
counts. Completed rating sheets and signed evidence are byte-for-byte unchanged. The original incomplete pilot and both corrected pilot archives remain
separate and unchanged, including their insufficient-evidence outcomes.

Preserve the attribution and source terms when redistributing the task text.
SQuAD's software license does not relicense Wikipedia-derived data; SGD-derived
material remains under CC BY-SA 4.0. See `attribution/` in the archive and the
[frozen-answer reference](../k2-32b/README.md).
