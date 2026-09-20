# Live evaluator qualification

This campaign checks whether an evaluator can run a real model task, retain its
original results, and deliver them to a separately installed InvarLock recipient.
It covers the 19 maintained profiles and the three native scorers.
**Fresh qualification is incomplete until the actual model captures, judge
measurements and recipient results have been collected and reviewed.**

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
`locks/lighteval.txt` for compatible xxhash. Promptfoo uses the verified npm archive;
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

Requests are journaled before inference. Lost responses and interrupted captures
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
`INVARLOCK_ALLOW_NETWORK=1` only for its collection subprocess. Preflight,
verification and reporting do not inherit that permission. When invoking the
collection CLI directly, scope the setting to that command:
`INVARLOCK_ALLOW_NETWORK=1 invarlock evaluate request.json --signing-key signer.pem`.

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
