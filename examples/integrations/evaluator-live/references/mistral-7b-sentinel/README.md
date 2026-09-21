# Real evaluator execution sentinel

Nineteen evaluator integrations executed eight fixed cases against
**Mistral-7B-v0.1** and **Mistral-7B-Instruct-v0.1**. Each integration drove its
actual framework callback and retained its native SDK output. This reference
contains those original captures, model-worker records, recoveries and signed
exact-match/NLL comparisons through both supported JSON import routes.

All **76 signed comparison packs** authenticated and replayed successfully.
Their original policy result is **regression / fail**, with verification status
**7**. Replay success means the recorded result was reproduced; it does not mean
the subject model was accepted. These eight engineering cases do not establish
representative model quality or a general performance difference between the
models.

The [judge companion](judge-README.md) retains the separately collected real Luna
ratings, original failed attempts and their outcomes. Earlier synthetic adapter
fixtures remain separate contract tests.

## Read the evidence

[reference.json](reference.json) lists all 19 evaluators and pins three archives:

| Archive | Contents |
| --- | --- |
| [captures.zip](captures.zip) | Original SDK captures and worker records; public runtime/source inventories; failed captures and explicit recoveries; shared recipient inputs and the relative replay map |
| [exact-match.zip](exact-match.zip) | 38 signed exact-match packs, original trust inputs and receipts, preflight results and reports |
| [normalized-nll.zip](normalized-nll.zip) | 38 signed NLL packs with the same recipient artifacts |

The split follows artifact roles and keeps each file below 10 MiB. Extracted
paths remain consistent across the three archives. Original `report.html` and
`report.md` files accompany every evaluator/route/scorer combination.
[provenance.json](provenance.json) describes the retained byte identities and
validation scope.

## Replay without models or SDKs

Use a clean environment with the core InvarLock wheel installed, without the
`judge` extra or evaluator SDKs:

```bash
/path/to/recipient/bin/python -I replay.py
```

The command verifies the installed package location, blocks outbound network,
authenticates the pinned archives and every member, authenticates the original
receipts, and independently recomputes all 76 signed comparisons. It generates
an ephemeral verifier key in memory and temporary new receipts; it does not
need any original signing keys. Temporary artifacts are removed afterward.

Exit **0** and `ok: true` mean this reference replay completed. Each result keeps
`decision: regression`, `policy_verdict: fail` and `verification_status: 7`.
The original collection and verification outcomes are never replaced with the
replay command's success status. Historical trust anchors come from this pinned
reference; they are not fresh deployment approvals.

## Scope and preserved failures

The fixed corpus contains three answerable SQuAD 2.0 questions, three
unanswerable questions and two LAMBADA narrative completions. The original
protocol retains the exact selected text, references, metadata, selection seed,
source hashes, model revisions and model-file digests. SQuAD 2.0 is attributed to
Pranav Rajpurkar, Robin Jia and Percy Liang, with Wikipedia-derived context under
CC BY-SA 4.0. LAMBADA source attribution is retained as
`EleutherAI/lambada_openai`, together with the original corpus selection and
[retained source](../../../evaluator_transaction/lambada_qwen35_deployment_400.jsonl).

Generation and reference likelihood came from the shared, fixed model workers;
this does not claim every framework has a native NLL scorer. There were 152
admitted case requests per model across the 19 integrations. The archives bind
observed outputs and numerical likelihood facts to each capture. Offline replay
can check those bindings and recompute scoring; it does not rerun inference or
independently attest that the original GPU computation happened.

The original failed LightEval captures, Promptfoo setup failure, Azure delivery
failure and Harness recipient refusals remain separate from their successful
setup or transport/serialization recoveries. Recovery records retain old/new
hashes and original inputs; no model calls were repeated to repair those
artifacts. Earlier and later helper source versions are retained separately.

No model weights, private keys, credentials, execution authorizations, SDK
installation trees or NLTK caches are included. Source files and SDK databases
inside the archives are retained data; the replay never executes or deserializes
them as code. The outbound controls apply to this trusted replay process and do
not claim containment of malicious SDK code.
