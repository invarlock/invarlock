# Judge through an OpenAI-compatible endpoint

Use `openai-compatible-judge` to rate frozen answers through a vLLM, Ollama,
LM Studio, or other compatible Chat Completions server. The collector sends the
same bounded request profile to `/v1/chat/completions`. It retains requests,
responses and service identity for later offline verification. This route suits
an already running service; direct local judging separately authenticates the
local artifact and executes it inside the declared offline runtime.

The endpoint is a service boundary. A model name, localhost URL, server product
label or reported fingerprint does not authenticate its weight files or runtime.
For authenticated local artifacts executed directly by InvarLock, use the
[native local judge example](../native-local-judge/README.md).

This example is a setup demonstration. Compatibility depends on the server and
model accepting the transmitted parameters and returning the declared JSON
rating. A successful transport check does not qualify every version or model.
For a retained, powered campaign whose two declared policies pass, see the
[K2 Luna held-out reference](../judge-measurements/references/k2-32b-luna-xhigh-heldout/README.md).

## Prepare a server and frozen answers

Install the matching InvarLock core wheel and obtain its matching examples
checkout, plus `httpx==0.28.1` for live HTTP collection. The optional
`invarlock[judge]` extra also includes that client, but the hosted-provider SDKs
are not needed for this route. Choose and record the server version,
endpoint configuration, served model name and any deployment provenance you
need. Keep model downloads and server installation separate from collection.

| Server | Typical local base URL | Setup reference |
| --- | --- | --- |
| vLLM | `http://127.0.0.1:8000/v1` | [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/) |
| Ollama | `http://127.0.0.1:11434/v1` | [Ollama compatibility](https://docs.ollama.com/api/openai-compatibility) and [local model import](https://docs.ollama.com/import) |
| LM Studio | `http://127.0.0.1:1234/v1` | [Headless service](https://lmstudio.ai/docs/developer/core/headless) and [local model import](https://lmstudio.ai/docs/cli/local-models/import) |

Use the exact model identifier exposed by your deployment. The service applies
its own chat template and runtime settings. vLLM can also apply the model's
`generation_config.json`; its documented `--generation-config vllm` option
selects vLLM defaults instead. Record whichever configuration you choose.
Local Ollama ignores API keys, so a placeholder key is not server authentication.
For LM Studio imports, use `--copy` if you need to preserve the original GGUF;
the default import moves it.

Start with existing canonical `baseline_run.json` and `subject_run.json` files.
For a small synthetic setup demonstration, use the pair in
`examples/native-local-judge`. These are illustrative answers, not measured
qualification results. From the repository root:

```bash
mkdir endpoint-judge-work
python examples/openai-compatible-judge/prepare.py \
  --workspace endpoint-judge-work \
  --baseline examples/native-local-judge/baseline_run.json \
  --subject examples/native-local-judge/subject_run.json \
  --service vllm --base-url http://127.0.0.1:8000/v1 --model judge-model
```

Select `--service ollama` or `--service lm_studio` with that deployment's URL and
model identifier. `openai_compatible` is available for another explicitly chosen
compatible service. Use a new workspace for each configuration. The helper
refuses existing generated files and makes no server request. For LM Studio it
also authors `response_format: json_schema`, because that service requires the
strict schema form rather than the default `json_object` request.

The helper freezes the exact input/output pair, rubric, sampling units and
service model identity. The plan binds the selected service family and the
SHA-256 digest of the canonical normalized base URL, including its trailing
slash. By default, each rendered request includes its case's reference.
`--reference-mode none` omits the reference from judge messages while preserving
the original frozen runs. It creates a different measurement plan.
For custom runs, edit the rubric and sampling units in `recipe-template.json`
before preparation; do not assign independent units merely to increase a count.

## Preflight, evaluate and report

Preparation writes the closed collection configuration:

```json
{
  "profile": "openai-compatible-text-frozen-answer-v1",
  "service": "vllm",
  "base_url": "http://127.0.0.1:8000/v1",
  "model": "judge-model",
  "authentication": "none",
  "request_timeout_seconds": 300,
  "max_calls": 2,
  "max_input_bytes": 131072,
  "max_output_tokens": 256
}
```

The example reserves two calls, 64 KiB per call for the aggregate transmitted
JSON request bodies, and 128 output tokens per call. `--max-input-bytes` authors
a different aggregate input-byte limit. Both calls and output reservations cover
the whole plan. Byte limits do not claim a tokenizer-specific input-token count.
The collector currently retains one source per call, with at most 1,000 calls
per plan. That permits 500 paired cases with one rating per side; repetitions
reduce that case count. Large responses can reach the 384 MiB evidence limit
earlier. These are storage limits, not recommended statistical sample sizes.

For an authenticated server, prepare with `--authentication bearer_env` and set
`INVARLOCK_OPENAI_COMPATIBLE_API_KEY` through your credential mechanism. The
configuration contains no secret. The collector uses this dedicated variable;
ambient OpenAI base URLs or keys do not select a different endpoint. Bearer
authentication requires HTTPS unless the endpoint is explicit loopback HTTP.

```bash
cd endpoint-judge-work
invarlock evaluate request.json --unsigned --preflight --json
INVARLOCK_ALLOW_JUDGE_NETWORK=1 invarlock evaluate request.json --unsigned --json
invarlock report evidence --html report.html --json
```

Preflight validates configuration without calling the server. The network opt-in
is scoped to judge collection, including loopback HTTP; offline verification and
reporting need no network permission. Use `--signing-key` in place of `--unsigned`
when publishing with your trusted evidence signer. A signing key is available to
that collector process; this workflow does not isolate it from the process.

Collection uses one attempt per trial, no automatic retry, and no response cache.
Completed checkpoints can be reused for the same frozen request. An admitted
request without a retained result is ambiguous and cannot be silently repeated.
Failures and malformed ratings remain visible. The bundled one-unit fixture
should produce insufficient evidence under its minimum-unit policy.

## Verify independently

Transfer signed evidence to a core-only recipient environment, then use a
recipient-owned trust policy and verifier key. See the complete
[recipient policy and trust guide](../../docs/reference/judge-measurements.md#replay-authentication-and-acceptance)
before authoring that policy:

```bash
invarlock verify evidence --trust-profile recipient-policy.json \
  --receipt verification.receipt.json --verifier-signing-key verifier.pem \
  --verifier-identity endpoint-judge-recipient --json
invarlock report evidence --html verified-report.html --json
```

If collection was unsigned, first publish the unchanged frozen runs, plan,
measurements and analysis policy through `judge_import` on the trusted signing
host. Do not call unsigned publication a successful recipient authentication.

Offline replay of `retained-openai-compatible-judge-v1` checks the frozen request,
response normalization, endpoint binding, planned schedule and analysis. The
service label is declared configuration; returned model identifiers and
fingerprints are service reports. Verification does not recover or authenticate
hidden model weights, prove that a server honored every generation parameter,
or attest the remote machine. Ordinary recipient authorization, sample coverage
and precision requirements still apply.
