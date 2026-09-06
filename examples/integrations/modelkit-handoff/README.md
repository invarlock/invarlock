# ModelKit content and recipient handoff

Run `python examples/integrations/modelkit_handoff.py --request recipient.json`
to check raw ModelKit blobs against both actual model directories, replay signed
technical evidence, and apply independently supplied current recipient policy.

Follow the [complete guide](../../../docs/user-guide/modelkit-handoff.md) and the
[recipient request schema](recipient.schema.json). This source example supports
a bounded KitOps 1.15.0 model-directory format and runs without an external service.
The real CLI test uses synthetic package contents and makes no inference claim.
