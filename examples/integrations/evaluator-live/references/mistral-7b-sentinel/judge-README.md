# Retained judge measurements

This companion preserves 992 completed Luna ratings and 32 original blocked attempts
from the eight-case evaluator integration campaign. All 52 signed judge evidence
packs replay to `insufficient_evidence`. These results demonstrate the integration
and verification paths; eight cases do not establish a powered model-quality
comparison.

The primary campaign covers 19 evaluator ecosystems through both envelope and native
JSON imports. Supplemental runs cover Inspect, Harness and Promptfoo with references
omitted and with three repetitions. Two original Inspect runs contain 16 blocked
attempts each. Their later corrections are separate packs containing 16 completed
ratings each; both original failures remain unchanged. The complete primary matrix
comprises 38 judge journeys alongside the [76 exact-match and normalized-NLL
journeys](README.md).

[judge-reference.json](judge-reference.json) fixes the archive hash, historical
signer and recipient identities, public-key fingerprints, original outcomes and
relative replay paths. [judge-reference.zip](judge-reference.zip) contains the
original frozen plans, analysis policies, complete baseline and subject runs,
measurements, signed evidence envelopes, recipient trust policies, original signed
receipts and reports. It also preserves the exact normalized Inspect event-source
bytes embedded in each measurement artifact and the original completed attempt
checkpoints. The model inputs and outputs came from the fresh model executions
recorded in the linked `captures.zip`; replay makes no new judge or model calls.

Private keys, credentials, caches, model weights and execution authorization
documents are excluded. Public keys remain in the original signatures. Some original
reports contain historical absolute paths; replay uses the catalog's relative paths
and does not require those original locations. No authenticated historical bytes were
edited for publication.

The archive's `model-captures.json` binds all 38 baseline/subject SDK captures to the
independently pinned [capture archive](captures.zip), including each capture
manifest, native result and protocol. It preserves model revisions, artifact and
tokenizer hashes and dataset attribution. SQuAD 2.0 context is Wikipedia-derived and
carries [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); LAMBADA
follows the retained corpus attribution and selection described in the capture
reference. The repository's Apache-2.0 license is retained separately and does not
replace dataset licenses.

Run with an independently installed core-only recipient from this checkout:

```sh
/path/to/recipient/bin/python -I \
  examples/integrations/evaluator-live/references/mistral-7b-sentinel/judge_replay.py \
  --output /tmp/judge-reference-replay
```

The output directory must not already exist. The helper disables network access,
validates both physical archive hashes and bounded ZIP inventories, checks every
judge member and linked capture file, then authenticates every original signed
receipt and requires its exact evidence analysis to replay. An authentic
`insufficient_evidence` decision completes replay with exit zero; it is not converted
into policy acceptance. The helper writes `replay-results.json` beside the extracted
evidence. No SDKs, API credentials, private signing keys or fresh measurements are
needed.
