# Harness likelihood reference

> **Outcome:** Replay a real CPU likelihood capture through signed evaluation,
> independent verification and reporting.
>
> **Audience:** Users integrating an existing likelihood evaluator with InvarLock.
>
> **Prerequisites:** An installed InvarLock package that supports captured NLL,
> its matching example checkout and Python. Offline replay needs no model,
> Harness, Torch, GPU or provider account.

This is a small integration check: can an existing evaluator measure the
likelihood of reference text, and can InvarLock independently verify the resulting
comparison? The same model runs on both sides, so it should reproduce matching
measurements. For a comparison between two different 7B models, see the
[Mistral likelihood reference](../mistral-7b-likelihood/README.md).

To inspect the existing result, go to [offline replay](#offline-recipient-replay).
Only [reproducing the measurement](#reproduce-the-model-measurement) needs the
model and evaluator dependencies.

## Result and scope

The unmodified LM Evaluation Harness `HFLM.loglikelihood` measured six fixed
context/reference pairs, twice, using separately loaded instances of the same
`sshleifer/tiny-gpt2` snapshot. Both runs used CPU float32 with caches disabled.
The original twelve measurements are retained; no model run was repeated to
improve its outcome.

| Observation | Retained result |
| --- | --- |
| Model revision | `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be` |
| Harness | Unmodified `lm-eval==0.4.12` |
| Cases | Six authored compatibility cases, including a non-ASCII reference and a two-token continuation |
| Mean normalized NLL, each side | 1.9267758581373426 nats per UTF-8 byte |
| Subject/baseline ratio | 1.0 |
| Paired bootstrap interval | [1.0, 1.0], 95% mass, 2,048 replicates |
| Declared policy decision | Pass |
| Recipient verification | Signed evidence and receipt authenticated; arithmetic replay completed |

The policy checks same-model conformance: at least six pairs, ratio at most
1.000001 and interval width at most 0.000001. These are integration tolerances,
not recommended production regression thresholds. The interval collapses
because every measured pair is identical. It does not establish model quality,
representative task coverage or a useful production sample size.

Each likelihood is the log probability of the **specified reference continuation**
given its exact context. The score divides its negative log-probability sum by
the reference's UTF-8 byte count, then averages the six per-case values. It does
not pool all bytes or measure the likelihood of a generated answer. The model
did not generate answers in this rehearsal.

The installed package ran outside the source checkout in a core-only environment.
The recipient independently regenerated the inputs before publication, verified
the signed pack, signed a separate receipt and generated HTML and Markdown
reports with the model identity. Verification and reporting preserved every
evidence file. This establishes one real external likelihood integration profile;
it does not qualify all evaluator/scorer combinations or confer native runtime
execution assurance on supplied measurements.

## Retained files

- `capture/manifest.json` binds the cases, exact model/tokenizer files,
  configuration, package versions and capture implementation before measurement.
- `capture/raw-results.json` retains Harness return values, token boundaries,
  byte counts, identities and measured timings for both runs.
- `evidence/` contains the signed captured comparison, its policy and normalized
  records. These derived records also preserve the original Harness facts.
- `verification.receipt.json` and the two public keys retain recipient replay.
- `reference.json` independently records the expected file hashes, run/request
  anchors, signer and recipient fingerprints, policy outcome and wheel identity.

Only public keys are retained. Model weights, tokenizer assets and private keys
are excluded. The recorded wheel identifies a development build from the stated
source commit; its `0.15.0` package metadata is not a claim that the released
v0.15.0 package supports this workflow.

Review and trust the reference's repository revision separately from the evidence
you receive. A pack or receipt cannot choose its own trusted anchors. File hashes
and signatures authenticate supplied facts; they cannot establish that an
untrusted capture operator actually executed the model.

## Offline recipient replay

Run these commands from the matching example checkout with the installed core
package. Choose a new `RECIPIENT` directory; existing destinations are refused.
The fixed hashes below pin the reviewed original capture. No inference occurs.

```bash
REFERENCE="$PWD/examples/captured-results/references/harness-likelihood"
REPLAY_ROOT="$(mktemp -d)"
RECIPIENT="$(cd "$REPLAY_ROOT" && pwd -P)/recipient"
python examples/captured-results/harness_likelihood_handoff.py \
  --capture "$REFERENCE/capture" --output "$RECIPIENT" \
  --manifest-sha256 sha256:b63c9a8b15da8261369be4ade43dc0fdd108c0ba5bd5e40484d4a8610a862395 \
  --results-sha256 sha256:b680ed88e66f26db180596204e6f817616e9c0d8501838af2c6ba060c0cb382f
invarlock evaluate --keygen "$RECIPIENT/keys"
invarlock verify "$REFERENCE/evidence" \
  --policy "$RECIPIENT/policy.json" \
  --expected-baseline-run sha256:2557cb25141521b847edd839aa51aab39f2e5bf3054b60ffa0d70138dcf66897 \
  --expected-subject-run sha256:2eb7a8bd4a30ce00a7b11a6e215f7d6d94673b35a3caf7d3ec48c1b34d087444 \
  --expected-request-digest sha256:6bad9bc08b5ddfe0af552aafe94678d41582352e02412b9ec8c14c6dc8c91f77 \
  --expected-signer sha256:f153155f4476daa4777ec0bc9977a7477b242ca62f7519c0daeff73ba3ff7a86 \
  --verifier-signing-key "$RECIPIENT/keys/private.pem" \
  --verifier-identity harness-recipient \
  --receipt "$RECIPIENT/verification.receipt.json"
invarlock report "$REFERENCE/evidence" \
  --html "$RECIPIENT/report.html" --markdown "$RECIPIENT/report.md"
```

Expected result: completed replay, authenticated evidence, a passing conformance
decision and a new signed recipient receipt. The retained receipt remains
unchanged. Report generation displays the recorded result; the independent
verification above supplies recipient acceptance.

To exercise a fresh publication too, use a separate operator key:

```bash
invarlock evaluate --keygen "$RECIPIENT/operator-keys"
invarlock evaluate "$RECIPIENT/request.yaml" \
  --signing-key "$RECIPIENT/operator-keys/private.pem" --fail-on-policy
```

This writes a new pack under `$RECIPIENT/evidence`. Verify it with its new
operator fingerprint and independently regenerated anchors, following the
[signed handoff guide](../../../../docs/user-guide/captured-results.md#signed-handoff).
The original reference signer is not authorized for that new publication.

## Reproduce the model measurement

Use a separate Python 3.12 environment containing the versions retained in the
manifest. The capture helper does not install packages or download model files.

| Package | Version |
| --- | --- |
| lm-eval | 0.4.12 |
| torch | 2.13.0 |
| transformers | 5.14.1 |
| accelerate | 1.14.0 |
| tokenizers | 0.22.2 |
| safetensors | 0.8.0 |
| huggingface-hub | 1.27.0 |

Prepare a local copy of the [pinned model snapshot](https://huggingface.co/sshleifer/tiny-gpt2/tree/5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be)
containing exactly `config.json`, `merges.txt`, `pytorch_model.bin`,
`special_tokens_map.json`, `tokenizer_config.json` and `vocab.json`.
The script verifies every file hash and three installed Harness implementation
hashes before measuring. It rejects changed context/reference token boundaries,
truncation and whitespace moved from context into continuation. The model's
repository provides the source assets; none are redistributed here.

```bash
CAPTURE_PYTHON=/absolute/path/to/evaluator-env/bin/python
MODEL_DIR=/absolute/path/to/pinned-model
"$CAPTURE_PYTHON" examples/captured-results/harness_likelihood_rehearsal.py \
  --model "$MODEL_DIR" --output /tmp/harness-likelihood-new-capture
```

The helper uses one CPU thread, two separate model instances and a 300-second
timeout. It disables result caches, cached logits and remote model code, selects local
files only and rejects Python socket connections. This is a local execution
guard, not an isolation boundary for untrusted model packages. Use trusted,
reviewed dependencies. Actual model loading and measurement took less than one
second combined in the retained run; package imports and environment preparation
are excluded from those timings. No provider calls or GPU were used.

The pinned Transformers loader reported obsolete `attn.masked_bias` buffers.
Checkpoint inspection found all current learned parameters present; the unused
entries were attention-mask constants. This reviewed warning applies to this
specific checkpoint/runtime combination.

Keep new capture bytes separate from this reference, review their new hashes,
then run the handoff helper and signed journey with those hashes. New timings,
runtime versions or file identities must not overwrite the retained evidence.
The [Harness source](https://github.com/EleutherAI/lm-evaluation-harness) defines
the measured API; the helper records its exact installed implementation hashes.
