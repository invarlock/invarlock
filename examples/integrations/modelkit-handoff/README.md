# ModelKit content and recipient handoff

Use this example at the receiving end of a model handoff. It checks that the
delivered ModelKit blobs contain the actual baseline and subject directories,
replays their signed technical evidence, and applies the recipient's current
policy. It consumes an existing evaluation; it does not run inference or deploy
the model.

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
The real CLI test uses synthetic package contents and makes no inference claim.
