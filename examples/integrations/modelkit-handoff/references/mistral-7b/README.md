# Mistral ModelKit handoff reference

> **Surface:** A retained Mistral-7B-v0.1 BF16/Q4 comparison, real KitOps package
> handoff, and historical recipient replay.
>
> **Stability:** Historical evidence for the exact models, sample, policy,
> runtime and tools recorded in [provenance.json](provenance.json).
>
> **Use this page when:** Reviewing a concrete package handoff that preserved a
> rejected model comparison, or checking its signed evidence without model files.

The BF16 baseline answered **94/128** cases correctly; the published Q4_K_M
subject answered **93/128**. The fixed policy rejected this comparison. Both
actual ModelKit packages nevertheless passed their content and evidence binding
checks. A successful transfer did not turn the rejected comparison into an
accepted model.

Start with the [HTML report](report.html) or [Markdown report](report.md),
then follow the package identities and replay instructions below. The report's
model IDs are digest-based: baseline means Mistral-7B-v0.1 BF16 GGUF and subject
means the published Mistral-7B-v0.1 Q4_K_M GGUF throughout this reference.

## Read the comparison outcome

| Measurement | Observed result |
| --- | --- |
| Baseline exact match | 94/128, or 73.4375% |
| Subject exact match | 93/128, or 72.65625% |
| Subject minus baseline | −0.78125 percentage points (pp) |
| 95% paired interval | −4.10336 to +2.49680 pp |
| Required lower bound | At least −2 pp; **failed** |
| Required sample | At least 128 records; passed |
| Required interval width | At most 20 pp; observed 6.60016 pp, passed |

The interval crosses zero. These observations do not establish degradation;
they fail to rule out a decline larger than the declared 2 pp allowance. There
were two baseline-only successes and one subject-only success. The sample and
policy were fixed before capture and were not changed after this result. No
absolute accuracy floor was declared.

These are 128 short, selected LAMBADA cases with one generated token per case,
not a representative production workload or the full LAMBADA benchmark. The
input comes from the repository's existing [400-passage subset](../../../evaluator_transaction/lambada_qwen35_deployment_400.jsonl)
of `EleutherAI/lambada_openai`, revision
`900124bf3b8235c6daf21033af9948b3f07346c4`. The selection retained prompts of at
most 96 Mistral tokens and lossless one-token targets with the declared BOS
prefix, then selected the first 128 of 194 eligible records by the frozen
record-ID hash order. `capture/selection.json` in the archive retains every ID,
the selection rule, tokenizer revision and source hashes. The final records
have SHA-256 `5b8a38c5f60b52df4f6562261f3c1b01bf497ed2d92f9244ffa0d63a59324731`.

The BF16 file was reproduced from `mistralai/Mistral-7B-v0.1` revision
`27d67f1b5f57dc0953326b2601d68371d40ea8da` using the pinned llama.cpp converter,
and exactly matched its previously declared file hash. The Q4 file came from
`TheBloke/Mistral-7B-v0.1-GGUF` revision
`d4ae605152c8de0d6570cf624c083fa57dd0d551`. Its publisher identifies the source
model; the external quantization process is not attested by this reference.

## Separate model contents from package identity

| Role | GGUF file SHA-256 | Original ModelKit manifest SHA-256 |
| --- | --- | --- |
| BF16 baseline | `9c35a363920c201c1eeffa8354b6bbd527131ebe9af98603b0836ba1d6b46e54` | `6b335b27fb77f2c71e43a0d9afcfec5512508489598d546465b76b1f14f0877d` |
| Q4 subject | `ce6253d2e91adea0c35924b38411b0434fa18fcb90c52980ce68187dbcbbe40c` | `3dc03aced5137c61d12b8e09763fd2e954ac2a64fa27de17d34ebc3bc362eca7` |

The file hash identifies the actual GGUF bytes. The ModelKit OCI manifest hash
identifies their packaging, including its configuration and layer descriptors.
Changing package metadata produced different package identities while preserving
both model file hashes. Technical evidence also binds the expected GGUF artifact
identities, runtime images, request, schedule and policy; these independent
anchors are retained in `reference.json` inside the pinned archive and in
[provenance.json](provenance.json).

KitOps 1.15.0 packed both real models, repacked their metadata, copied the local
content store to a separate recipient store, and unpacked the independently
selected original manifest digests after their tags had changed. The recipient
checked the manifest/configuration/layer graph and all model bytes against its
expected pair before replaying the signed comparison and acceptance envelope.
The model files were 14,484,732,000 and 4,368,438,912 bytes respectively.

The ordered checks rejected a wrong runtime anchor, revoked envelope signer,
altered evidence, missing selected package and changed candidate bytes. The
revocation result explicitly recorded the revoked-signer error in addition to
the existing policy rejection. The runner restored intentionally changed inputs
and verified the recipient again before loading the delivered model.

An explicit `--smoke-on-policy-rejection` allowed isolated usability checks.
Docker and Podman generated the same eight-token continuation for
`The capital of France is`: `a city of many faces. It is` with one leading
space and two trailing newlines. That is a nonempty load/generation observation, not a correct-answer
claim or an acceptance decision. Its exact UTF-8 SHA-256 is
`84d4aa3a07e5b68ffa748e47d605ac83393067655ab862f8804717b69418a557`.
Both recipient decisions remained rejected. No second 128-case comparison was
run under Podman.

The runtime image's original manifest, configuration and twelve filesystem
layers were authenticated during Docker-to-Podman transport. Docker's manifest
identity and Podman's configuration identity differ; their relationship and
fresh pre-load inspection results are recorded separately. Each generation used
CPU execution without network access, a non-root user, a read-only model mount
and root filesystem, dropped capabilities and explicit resource limits.

## Replay without models or containers

Use Python 3.12 or newer with an installed InvarLock wheel and core dependencies.
From the repository root:

```bash
python -I examples/integrations/modelkit-handoff/references/mistral-7b/replay.py
```

Expected output includes `"ok": true`, `"integrity_ok": true`,
`"receipt_ok": true`, `"verification_status": 7`, `"policy_verdict": "fail"`
and `"historical_recipient_accepted": false`. Exit zero means the retained
outcome was verified; it does not mean the model was accepted.

Replay authenticates the complete archive against its fixed SHA-256 before
extraction. It freshly checks the comparison, original signed receipt and DSSE
against retained independent anchors. It checks the historical recipient policy
at the recorded original recipient-check time, not the current clock. It issues
no new receipt and performs no model inference. The demonstration trust inputs
are historical reference anchors, not a current recipient's approval or trust
configuration.

The archive contains 68 regular files: 137,128 bytes compressed and 350,827 bytes
expanded. It includes unchanged signed evidence, public trust inputs, exact
sample and source provenance, compact delivery observations, eight small package
manifest/configuration JSON files, and selected original failure records. It
contains no model files, model or image layer payloads, blob-store trees, private
keys, wheels or broad execution logs. The complete local collection was retained
before notifying that the remote host could be terminated. Its digest and the one diagnostic path
normalization in the public delivery observation are recorded in provenance.

Compact replay independently rechecks signed evidence and envelope bindings.
It does **not** repeat the recorded package-content or inference checks. To repeat
those checks, supply both exact GGUF files and the original selected ModelKit
manifest, configuration and complete model-layer blobs, together with the
recipient's independently chosen pair, policy, trusted public keys and runtime
anchors. A fresh recipient must apply its current clock and policy. Repeating
inference also requires the exact qualified runtime image and selected container
engine. Follow the [complete handoff guide](../../../../../docs/user-guide/modelkit-handoff.md)
for the maintained point-of-use checker and real journey launcher.

## Timing and retained attempts

The two-model capture took 117 minutes 52 seconds on a Linux x86-64 guest with
32 virtual CPUs, using ten compute threads and a ten-CPU, 24-GiB limit per side.
The final runtime settings were context 128, prompt batch 128, prompt micro-batch
64, seed zero, one generated token and a 300-second per-record timeout. The
maintained portable CPU build launches a fresh llama-completion process per
record. A separate four-thread diagnostic measured about 82.4 seconds evaluating
an 81-token prompt; process/thread-pool setup reached about 0.295 seconds. The
measurement does not support attributing the dominant cost to process startup.

Measured canaries selected the declared ten-thread configuration before capture;
changing four to ten threads reduced the first BF16 canary from 120.36 to 49.30
seconds and the first Q4 canary from 84.74 to 34.43 seconds. These timings describe
this model, sample, settings and portable backend build, not general InvarLock
or production latency. Real Kit transfer, ordered checks and Docker generation
took another 19 minutes 32 seconds. Podman's recipient recheck and generation
took 166.30 seconds. Preparing its authenticated image in parallel took 4.34
seconds; the six-minute preparation limit was not its measured duration.

The original local CPU timeout, an early non-canonical observation preflight
failure, and the receipt postprocessing error remain preserved. The latter
omitted an expected trust-profile digest; resolving it from the original trust
inputs verified the same receipt. Only report and DSSE preparation resumed.
Original evidence and receipt bytes were not rewritten, no scored records were
rerun, and the failed policy remained unchanged.

Execution used source `59d9bbc048d1fe10083e912b32912445cb7bb974` and the exact
wheel recorded in provenance; its `0.16.1` filename does not identify the
published release wheel. The HTML report was rendered later from the same
signed pack using source `7f6eb7f6b7ae0a7d4c2940e3721f1778c4d421c6`. This is an
execution and handoff reference for those declared artifacts, not a broad
quantization-quality, production-workload or release qualification.
