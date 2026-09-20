# ModelKit content and recipient handoff

Use this example at the receiving end of a model handoff. It checks that the
delivered ModelKit blobs contain the actual baseline and subject directories,
replays their signed technical evidence, and applies the recipient's current
policy. The point-of-use checker consumes an existing evaluation. The separate real
journey launcher also packs and transfers actual models and runs a bounded
inference smoke after acceptance, or after an explicit
`--smoke-on-policy-rejection` request that preserves the rejected decision.

Prepare Python 3.12 or newer, an installed InvarLock wheel and matching example
checkout, the local package blobs and model directories for both sides, native
pack-v1 evidence, its acceptance attestation, and independently selected trust
inputs. Create `recipient.json` from the guide and schema below; its paths resolve
relative to that file. Captured packs and judge envelopes are not supported by
this handoff.

Run from the repository root:

```bash
python examples/integrations/modelkit_handoff.py --request recipient.json
```

Read the JSON result and exit code together: `0` means all package, content,
technical and current acceptance checks passed; `1` is an authenticated policy
rejection; `2` means invalid, unsupported, altered or unauthenticated input. Keep
the checked model directories unchanged before loading them. A pass is a check
of this handoff at invocation time, not a general model-quality claim.

Follow the [complete guide](../../../docs/user-guide/modelkit-handoff.md) and the
[recipient request schema](recipient.schema.json). This source example supports
a bounded KitOps 1.15.0 model-directory format and runs without an external service.
For GGUF, the optional `artifact_file` selects the evaluated file within each
model directory. Its file SHA-256 remains separate from the package digest and
complete directory inventory.

The [real journey](../../../docs/user-guide/modelkit-handoff.md#transfer-actual-evaluated-models-and-run-an-inference-smoke)
uses `python -m examples.integrations.modelkit_real_journey` with actual model
files, independently selected native evidence and trust inputs, a pinned KitOps
binary, a separate installed-wheel Python, and an immutable local inference
image in Docker or Podman. Select `--container-engine docker` (the default) or
`--container-engine podman`; the launcher uses only that engine and requires a
full `sha256:` local image ID. It retains original/repacked package digests,
independent recipient checks, intentional rejections and generated text. It needs local model storage
and CPU memory; it does not use a paid API or external registry.

The fast tests and the original real-CLI serialization test retain synthetic
package contents and make no inference claim.

The [retained Mistral-7B reference](references/mistral-7b/README.md) includes a
real BF16/Q4 comparison on 128 fixed cases, actual KitOps delivery checks, and
bounded Docker and Podman generation. Its policy rejection remains visible; the
compact replay verifies historical signed evidence without model execution.
