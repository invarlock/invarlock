# Live evaluator qualification

This campaign checks whether an evaluator can run a real model task, retain its
original results, and deliver them to a separately installed InvarLock recipient.
It covers the 19 maintained profiles and the three native scorers.
The [retained Mistral 7B sentinel](references/mistral-7b-sentinel/README.md)
contains 304 fresh model executions and all 114 scorer/import journeys. It retains
992 usable judge ratings, the original 32 blocked attempts, and independently
verified outcomes. This establishes the declared integration profile, not model
quality or support for every configuration of an evaluator.

SDK tests execute real framework callbacks with controlled task results. They
test integration code without charging a provider or loading a model. The
[export parity example](../evaluator-parity/README.md) separately checks retained
measurements. Neither substitutes for fresh model and judge execution.

## Components

| File | Responsibility |
| --- | --- |
| `prepare.py` | Freeze cases, attribution, model files, SDK versions and example policies |
| `model_worker.py` | Keep one model loaded; generate fresh answers and measure exact reference continuations |
| `scalar.py`, `batch.py`, `harness.py` | Invoke the supported evaluator APIs and retain actual exports |
| `capture.py` | Journal requests and responses between the evaluator and model worker |
| `bindings.py` | Bind runtime measurements to the SDK capture without changing original measurements |
| `recipient.py` | Check original task ledgers against exports; evaluate, independently verify and report |
| `supervise.py` | Apply an external process-group deadline |
| `recovery.py`, `recover_harness.py` | Recover explicitly supported interrupted captures without changing original results |
| `judge.py` | Freeze a bounded judging schedule, collect authorized measurements and verify them offline |

The model pair is Mistral 7B base and Mistral 7B Instruct, using the exact revisions
and file inventories in the maintained
[likelihood reference](../../captured-results/references/mistral-7b-likelihood/README.md).
The worker accepts only that profile, authenticates every local model file,
downloads nothing, and disables remote model code. Each evaluator receives fresh
executions. Warming a model once avoids loading 19 copies; it does not return
cached answers to another evaluator. Run the two models sequentially on one GPU,
or in parallel on two GPUs with a separate worker, private socket and output
directory per role. For parallel workers, set `CUDA_VISIBLE_DEVICES=0` for one
and `CUDA_VISIBLE_DEVICES=1` for the other.

## Data and example policies

The eight-case sentinel contains two narrative continuations, three answerable
passage questions and three unanswerable questions. The complete engineering
corpus contains 64 cases, split 16/24/24. Selection uses source IDs and fixed hashes,
never measured outputs. Contexts remain complete and source grouping is retained.

Sources are the existing LAMBADA selection and the public
[SQuAD 2.0 development set](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json).
SQuAD context and questions retain their CC BY-SA 4.0 attribution. The upstream
software license is not a license for the dataset text. Generated protocols
retain source hashes, attribution and transformation details.

These counts exercise capture boundaries, not production model quality or a
particular statistical power. Exact match compares the entire completion with
the selected reference, including its continuation separator. It is stricter
than semantic judging and is not the official SQuAD scoring procedure. NLL
measures that reference continuation, not the probability of a generated answer.

Example policies require no more than a two-percentage-point accuracy loss or a
1.05 subject-to-baseline NLL ratio, with an interval-width limit of 0.1. These are
declared example requirements, not universal production recommendations. A small
sample can fail the precision requirement. An authenticated rejection or an
insufficient-evidence outcome can still establish a working integration.

## Prepare and preflight

Save the public SQuAD source locally. Use new output directories:

```bash
python examples/integrations/evaluator-live/prepare.py \
  --squad /path/to/dev-v2.0.json --stage sentinel --device cuda \
  --output /path/to/protocol.json
```

Review the protocol and printed digest independently. Use `mps` for a suitable
Mac or `cpu` for an admitted CPU run. Changing configuration, corpus or policy
changes the protocol identity.

The CUDA runtime uses `lm-eval[hf]==0.4.12`, `transformers==5.14.1`,
`torch==2.13.0+cu130`, `accelerate==1.14.0+invarlock.1`, and
`sentencepiece==0.2.2`. Build the hardened Accelerate wheel using the repository's
[development setup](../../../CONTRIBUTING.md#development-setup). Retain the full
resolved package inventory and implementation hashes. Tokenizer-only preflight
does not qualify model execution.

```bash
python examples/integrations/evaluator-live/model_worker.py \
  --protocol /path/to/protocol.json --protocol-sha256 sha256:REVIEWED_DIGEST \
  --role baseline --model-dir /path/to/pinned-baseline-files \
  --output /path/to/new-baseline-preflight --socket /private/path/model.sock \
  --preflight
```

Repeat for the subject. Preflight hashes weights without loading them and checks
the tokenizer's exact context/continuation boundary. It refuses truncation and
extra files outside the declared model inventory.

## Execute an admitted campaign

Obtain any required compute and provider budget approval. Keep the Unix socket
inside a private directory. Start `model_worker.py --execute` under
`supervise.py --seconds LIMIT --output NEW_DIRECTORY -- COMMAND...`, using a cap
no greater than the admitted limit. The external supervisor is required: a Python
timer alone cannot interrupt a stalled native GPU call.

Use separate, pinned SDK environments. The normal locks are in
`examples/evaluator-qualification/locks/`. LightEval also needs this example's
`locks/lighteval.txt` for a compatible `xxhash` version. Promptfoo uses the verified npm archive;
LightEval requires checksum-verified NLTK resources staged before execution.
`scripts/evaluator_sdk_gate.sh` demonstrates setup and tests it without model calls.

```bash
python examples/integrations/evaluator-live/capture.py \
  --protocol /path/to/protocol.json --protocol-sha256 sha256:REVIEWED_DIGEST \
  --role baseline --evaluator inspect-ai --socket /private/path/model.sock \
  --output /path/to/new-inspect-baseline
```

Repeat for the admitted evaluators and both model roles, following the selected
sequential or separate-GPU schedule. Metric libraries
run inside an application that invokes the worker; they are not model servers.
This local profile does not qualify hosted experiment storage, remote dataset
APIs or arbitrary framework configurations.

Requests are recorded in the execution journal before inference. Lost responses and interrupted captures
remain incomplete and cannot silently rerun. Preserve failed attempt directories.
A changed protocol requires new captures.

Harness's Arrow dataset can add null metadata fields that occur on other cases.
The capture driver checks this exact expansion against the frozen case schedule
and binds likelihood metadata to the actual exported document. Original worker
measurements remain unchanged. To recover an earlier affected export, run the
offline helper in the independently installed recipient environment:

```bash
/path/to/recipient/bin/python -I examples/integrations/evaluator-live/recover_harness.py \
  --protocol /path/to/protocol.json --protocol-sha256 sha256:REVIEWED_DIGEST \
  --capture /path/to/original-harness-capture --role baseline \
  --output /path/to/new-derived-harness-capture
```

Repeat for the subject, then use the derived directories as recipient inputs.
The helper preserves every original SDK export and task ledger under `original/`,
records the derivation in `recovery.json`, and never calls the model. It refuses
changes beyond the schedule-derived null fields and never overwrites a capture.

## Independent recipient and judge measurements

Build the candidate wheel and install it in a clean core-only environment. The
recipient refuses source imports, evaluator SDKs and provider credentials. Use
physical paths rather than symlinked directory aliases.

```bash
/path/to/recipient/bin/python -I examples/integrations/evaluator-live/recipient.py \
  --protocol /path/to/protocol.json --protocol-sha256 sha256:REVIEWED_DIGEST \
  --baseline-capture /path/to/inspect-baseline \
  --subject-capture /path/to/inspect-subject \
  --evaluator inspect-ai --route native-json --output /path/to/new-recipient
```

Repeat with `--route envelope`. The helper retains raw captures and task ledgers,
checks planned membership, task text, references, metadata, identity, failures
and measurements, then creates signed comparisons and verification receipts.
Keep generated private signing keys out of published example bundles.

Supply a reviewed `--judge-recipe` to freeze a judge plan against actual answers.
`prepare.judge_recipe` prepares a bounded Luna xHigh proposal using the prior
corrected pilot's 25,000-token output allowance. It does not authorize calls.
Its reference-aware profile passes references separately to the judge, never to
the evaluated model, and groups related questions by source.

The proposed per-call reservation is $0.0312, covering 4,096 input tokens,
25,000 output tokens and conservative cache-write accounting at the documented
[Luna standard rates](https://developers.openai.com/api/docs/pricing).
This is a maximum reservation, not an expected bill. Recheck provider prices
and approve the aggregate schedule before collecting measurements.

The admitted `judge.py collect --execute-collection` helper sets
`INVARLOCK_ALLOW_JUDGE_NETWORK=1` only for its collection subprocess. Preflight,
verification and reporting do not inherit that permission. When invoking the
collection CLI directly, scope the setting to that command:
`INVARLOCK_ALLOW_JUDGE_NETWORK=1 invarlock evaluate request.json --signing-key signer.pem`.
This permits provider calls only inside the configured judge phase; it does not
enable network access for native model capture.

Collect fresh measurements for that exact plan, retain all attempts, and finish
the normal judge evaluation and verification workflow. Old ratings cannot be
rebound to new runs. Import formats can share calls only if their complete plan
identities are identical. Record reference-free judging, repetitions, interruption
and budget-boundary checks separately from the primary reference-aware campaign.

## Completion record

Keep a result for each evaluator/version/scorer/import-route combination, linking
the original SDK export, execution ledger, recipient outcome, reports and
rejection tests. Record setup and execution time separately. Preserve adverse
outcomes. A row is qualified only after its real capture, scorer measurements
and independent recipient checks exist. Controlled callbacks alone cannot close it.

## Controlled HTTP task service

The priority Inspect, Harness, Promptfoo and Langfuse callbacks can use a real
loopback HTTP boundary before reaching the private model worker. This example's
`/v1/tasks` endpoint returns generation and reference-continuation likelihood.
It is a controlled HTTP task service, not an OpenAI-compatible API or qualification
of a third-party cloud provider. The separate
[HTTP completion example](../../hosted-service/README.md) covers the
OpenAI-compatible completion profile and its stated scoring limits.

Before admitting this profile, add `http_services.baseline` and
`http_services.subject` to the frozen protocol. Each declaration must contain
`provider`, `service`, `deployment`, `requested_model`, `endpoint` and
`helper_sha256`. Use distinct endpoints such as
`http://127.0.0.1:18081/v1/tasks` and `http://127.0.0.1:18082/v1/tasks`; the helper
rejects DNS names, non-loopback addresses, redirects and alternate paths. Pin the
physical SHA-256 of `http_service.py` after reviewing the source. Recompute and
independently approve the complete protocol digest before starting workers.

Start the ordinary model worker with that protocol, then start the HTTP bridge:

```bash
python examples/integrations/evaluator-live/http_service.py \
  --protocol /path/to/http-protocol.json \
  --protocol-sha256 sha256:REVIEWED_DIGEST --role baseline \
  --socket /private/path/baseline.sock --output /path/to/new-http-baseline
```

Run both the worker and bridge under the external `supervise.py` process-group
deadline described above. The bridge additionally applies a whole-process alarm
that interrupts slow HTTP reads and worker waits. Repeat for the subject, then
run the ordinary capture command in each admitted SDK environment. Presence of
`http_services` selects the declared HTTP endpoint. Omit `--socket` from the
HTTP capture command; local captures still require their private socket. Python capture network permission is
limited to the exact endpoint, alongside Promptfoo's existing local callback
bridge permission.

The request carries the exact frozen task text, reference and configuration.
References are used for continuation likelihood; they are not appended to the
model's generation prompt. The server checks these values before invoking the
worker, then returns model identity and configuration from the actual worker
result. HTTP bodies and observed timing remain separate from the unchanged
artifact-bound task results. Interrupted or lost responses cannot silently repeat
an admitted task.

The complete observation window becomes known only after capture. Therefore the
helper retains the original SDK JSON as `native-original.json`, records its byte
digest, and explicitly derives `native.json` with hosted NLL identity bindings.
The derivation preserves numerical likelihoods and original artifact facts. Its
canonical run has `artifact_digest: null` and a `service_identity` descriptor;
the descriptor's digest identifies the observed service, never hidden weights.
The independent recipient rechecks HTTP requests and responses against every
original task, recomputes the observed window and exact export derivation, and
then runs the existing scorer and judge-plan workflows offline. No historical
capture or judge measurement can be relabeled as a fresh HTTP execution.

## Extended four-evaluator judge campaign

`closure_judge.py` freezes a separate campaign for Inspect, Harness, Promptfoo
and Langfuse. It preserves the original sentinel's plans and limits. Build and
install the current core wheel in both the provider and recipient environments
before using the new batch stop/resume control.

Prepare a JSON specification with `format: invarlock/live-judge-campaign-v2`,
`maximum_calls`, `maximum_cost_microusd`, `cost_microusd_per_call: 31200`, and a
`groups` list. The helper caps this specific campaign at 1,288 admitted calls and
40,185,600 microdollars ($40.1856); these are campaign reservations, not core
InvarLock limits or estimates of the bill. Lower caps are allowed if the schedule
fits. Each group has exactly these fields:

| Field | Required value |
| --- | --- |
| `id` | Unique lowercase letters, digits and hyphens, at most 48 characters |
| `protocol`, `protocol_sha256` | Protocol path and its independently checked canonical digest |
| `captures` | JSON index mapping evaluator names to original `baseline` and `subject` capture directories |
| `evaluators` | Selected names from `inspect-ai`, `lm-evaluation-harness`, `promptfoo`, `langfuse` |
| `routes` | `envelope`, `native-json`, or both |
| `case_count` | Exact complete case count in this group's protocol |
| `profile` | `primary`, `reference-free`, `repeat-control`, or `budget-control` |
| `admitted_calls` | `null`, except an explicit smaller positive call cap for `budget-control` |

Primary judging uses per-case references and one rating per side/case.
Reference-free judging uses one rating without references; repeat-control uses
three ratings with references. Budget-control uses a separate reference-free
three-rating plan with a smaller admission limit, deliberately retaining an
incomplete outcome. Repetitions never increase the independent case count.

The proposed extension consists of four 64-case local primary comparisons
(1,024 calls across both routes), four eight-case HTTP primary comparisons
(128), Langfuse's original eight-case reference-free and repetition controls
(128), and four two-call budget controls (8). The budget controls deliberately
leave 184 trials without call admission. Those incomplete controls are not model-quality
references. Freezing this proposal does not perform or authorize collection:

```bash
/path/to/recipient/bin/python -I examples/integrations/evaluator-live/closure_judge.py freeze \
  --specification /path/to/specification.json --output /path/to/new-campaign
```

Review the resulting plans, original inputs and aggregate reservation. After
collection is authorized, pass the printed admission digest and an entry ID from
its manifest to the provider environment:

```bash
/path/to/provider/bin/python -I examples/integrations/evaluator-live/closure_judge.py collect \
  --root /path/to/new-campaign --admission-sha256 sha256:REVIEWED_DIGEST \
  --entry local-primary-inspect-ai-envelope \
  --recipient-python /path/to/recipient/bin/python \
  --execute-collection --stop-after-batches 1
```

Use the same command with `--resume`, omitting `--stop-after-batches`, to finish
that entry. The pause occurs after a batch is durably retained and adds no calls
to its original schedule. Earlier attempts and reservations remain charged.
Completed collections and budget-exhausted controls produce signed evidence;
verification and reporting run in the separate offline recipient. If only that
recipient step fails after publication, `--resume` retries the recipient step
without collecting again. An already verified entry refuses further collection.

Keep fresh execution status separate from these implementation instructions.
Only retained results and their independent replay close the corresponding
qualification rows.
