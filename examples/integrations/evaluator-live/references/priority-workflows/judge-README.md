# Retained judge evidence for priority workflows

This companion retains 1,288 completed Luna ratings across 24 signed evidence
packs. Independent offline replay reproduces `insufficient_evidence` for every
pack. These results demonstrate the exercised integration paths and preserve
their statistical limits; they do not establish model quality or policy
acceptance.

| Profile | Packs | Completed ratings | Scheduled trials |
| --- | ---: | ---: | ---: |
| Four evaluators, 64 local cases, both routes | 8 | 1,024 | 1,024 |
| Four evaluators, 8 controlled HTTP cases, both routes | 8 | 128 | 128 |
| Langfuse, 8 cases, reference-free, both routes | 2 | 32 | 32 |
| Langfuse, 8 cases, three repetitions, both routes | 2 | 96 | 96 |
| Four evaluators, separate bounded budget controls | 4 | 8 | 192 |

The four evaluators are Inspect, LM Evaluation Harness, Promptfoo, and Langfuse.
Routes are the evaluator export envelope and native JSON import. The budget
controls deliberately leave 184 scheduled trials without a call; their
incomplete observations and decisions are retained. The reference-free and
repetition controls use the original eight-case model captures from the
[sentinel reference](../mistral-7b-sentinel/README.md), with newly collected
judge ratings for this campaign. The local and HTTP primary profiles use the
fresh captures in [this reference](README.md).

Each local native-JSON primary entry paused after its first complete two-call
batch, then resumed the same frozen plan. The archives include four original
partial measurement sets, stop/resume observations, the unchanged checkpoint
header and first result bytes, and all final result shards. Replay checks the
first attempts remain identical and binds the final result events to the
authenticated normalized SDK sources. The stop/resume observation files are
unsigned retained execution observations; they are not independent attestations
of wall-clock timing. Execution admission bodies are excluded, while their
opaque hashes remain inside the unchanged observations.

`judge-reference.json` names and pins the two judge archives and their capture
companions. `judge_replay.py` independently pins that catalog, validates bounded
ZIP inventories, re-imports the complete original SDK/task/HTTP captures, and
compares every reconstructed run with the frozen judge inputs. It then verifies
the original public verifier signatures and replays their evidence and policies,
including the HTTP service subject identities. Original absolute paths remain
historical data; replay uses the catalog's relative paths.

From the repository root, use an independently installed core-only environment
without evaluator SDKs or API credentials:

```sh
/path/to/recipient/bin/python -I \
  examples/integrations/evaluator-live/references/priority-workflows/judge_replay.py \
  --output /tmp/priority-judge-replay
```

The output directory must be new. Replay blocks network access and makes no
model or judge calls. Temporary reconstruction keys are newly generated locally,
removed with the temporary directory, and do not replace any retained signature.
The replay summary records all 24 original receipt verification results. It also
renders current HTML, Markdown and JUnit reports under `current-reports/` from the
unchanged evidence. These derived reports use the installed renderer, so display
improvements do not rewrite the original signed evidence or archived historical
reports.

The archives preserve original plans, runs, policies, normalized SDK events,
measurements, signed evidence, public trust inputs, receipts and reports.
Private keys, credentials, model weights, caches, execution admission bodies and
controller files are excluded. Original data licensing and model attribution
remain in the linked capture references; the judge archives also include the
repository license. Each archive is smaller than 10 MiB.
