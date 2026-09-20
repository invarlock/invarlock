# Priority evaluator workflows

These retained runs exercise Inspect, LM Evaluation Harness, Promptfoo and
Langfuse with two real Mistral 7B models. All 32 exact-match and normalized-NLL
comparisons have valid evidence and signed verification receipts. Their recorded
policy decision is **regression**, not acceptance. A successful archive replay
means those original results and their source bindings reproduced correctly.

| Source profile | Cases per model and evaluator | Actual boundary |
| --- | ---: | --- |
| Local artifact | 64 | Each SDK drives its callback into the fixed model worker. |
| Controlled HTTP service | 8 | Each SDK callback crosses the loopback `/v1/tasks` service for generation and reference likelihood. |

The four evaluators made 576 fresh case requests across both models and profiles.
Each case captured generation and reference likelihood. Both the native JSON and
export-envelope import routes were checked with both scorers: four evaluators ×
two profiles × two routes × two scorers = 32 signed packs. This does not claim
that each SDK implements its own NLL scorer; the fixed model worker supplies the
likelihood measurements and the recipient checks their bindings.

The [earlier sentinel](../mistral-7b-sentinel/README.md) retains the separate
8-case qualification across all 19 evaluators. Its original archives are
unchanged. The deeper 64-case runs here apply to these four evaluators. Judge
measurements are a separate companion; these three archives contain EM/NLL
results only.

## Replay offline

Build and install InvarLock from this checkout. From the repository root:

```bash
QUALIFICATION_TMP=$(mktemp -d)
uv build --wheel --out-dir "$QUALIFICATION_TMP/dist"
uv venv --python 3.12 "$QUALIFICATION_TMP/recipient"
uv pip install --python "$QUALIFICATION_TMP/recipient/bin/python" \
  --require-hashes -r requirements/workflows/core-py312.txt
uv pip install --python "$QUALIFICATION_TMP/recipient/bin/python" \
  --no-deps "$QUALIFICATION_TMP"/dist/invarlock-*.whl
"$QUALIFICATION_TMP/recipient/bin/python" -I \
  examples/integrations/evaluator-live/references/priority-workflows/replay.py
```

The helper requires an installed package outside the checkout and blocks outbound
network. It makes no model or judge calls and requires no evaluator SDKs, model
weights, API credentials or historical private signing keys. It authenticates the
three archive snapshots and complete member manifest, then checks:

- Original SDK, task and HTTP request/response bindings against each frozen
  protocol, including actual service observation windows and model identities.
- Native inputs and both import routes against the exact normalized run digests
  in the signed evidence.
- Every original signed receipt and a fresh independent evidence verification
  against the retained pair, policy and signer anchors.

Exit 0 and `ok: true` indicate successful replay. The returned policy verdicts
remain `fail`, with original verification status 7. The historical public trust
anchors authenticate these retained results; a new recipient still needs its
own independently selected models, policy and trusted signers.

## Models, tasks and limits

The baseline is `mistralai/Mistral-7B-v0.1` at revision
`27d67f1b5f57dc0953326b2601d68371d40ea8da`. The subject is
`mistralai/Mistral-7B-Instruct-v0.1` at revision
`ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca`. Both used the declared CUDA float16
configuration, 1,024-token context limit, a 32-token generation allowance and
complete reference-likelihood capture. Exact model-file inventories, token
facts and runtime/source identities remain in the archives. Repeating inference
requires those original model bytes and the declared runtime and SDK versions.

The 64-case profile contains 16 LAMBADA narrative completions, 24 answerable
SQuAD 2.0 questions and 24 unanswerable questions. The HTTP profile uses an
eight-case subset containing two, three and three respectively. The protocols preserve selected text, references,
source records, grouping, selection seed and task-family counts. SQuAD 2.0 is
attributed to Pranav Rajpurkar, Robin Jia and Percy Liang, with Wikipedia-derived
context under CC BY-SA 4.0. LAMBADA attribution is retained as
`EleutherAI/lambada_openai`, together with the original corpus selection and
[retained source](../../../evaluator_transaction/lambada_qwen35_deployment_400.jsonl).
The SQuAD reference projection uses the first original answer alias and
`NO_ANSWER` for impossible questions, as declared in the frozen case metadata.

This is a selected integration engineering corpus, not a representative
production workload or a powered general comparison of base and instruction
models. The HTTP profile is a locally controlled task service, not a claim about
OpenAI-compatible endpoints, arbitrary cloud providers or the hosted Langfuse
product. Offline replay verifies retained observations and recomputes scoring;
it does not independently attest that the original GPU computation occurred.

## Retained files

[reference.json](reference.json) maps source profiles, captures, signed packs and
archive pins. [provenance.json](provenance.json) records the original collection
archive and the unchanged-file mapping. `captures.zip` includes original model
and task journals, HTTP bodies, SDK exports, setup failures and their recovery
logs, preflight results, source inventories and resource observations.
`exact-match.zip` and `normalized-nll.zip` contain the original signed evidence,
public trust inputs, verification receipts and reports.

The public archives omit private process-launch admissions, credentials, private
keys, model weights and environment/cache trees. Original model/task admissions
are preserved. Member hashes distinguish the unchanged original evidence from
new packaging metadata; no model identities, observations or signed decisions
were rewritten to make this reference portable. Every archive is below the
repository's 10 MiB file limit.
