# Mistral 7B likelihood comparison

This reference compares two distinct full 7B checkpoints through unmodified
LM Evaluation Harness 0.4.12. It retains all 400 paired measurements, a frozen
policy, signed captured evidence and an independently verified rejection receipt.

| Role | Model | Immutable revision |
| --- | --- | --- |
| Baseline | `mistralai/Mistral-7B-v0.1` | `27d67f1b5f57dc0953326b2601d68371d40ea8da` |
| Subject | `mistralai/Mistral-7B-Instruct-v0.1` | `ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca` |

The subject is the official instruction-tuned version of the baseline. Both see
identical plain text, without a chat template. The comparison measures narrative
continuation likelihood; it does not test instruction-following quality.

The practical scenario is a checkpoint migration for a prose-completion
workload: does the replacement preserve likelihood on the existing reference
text? This result rejects the change under the declared illustrative policy.
It does not answer whether the instruction-tuned checkpoint is a better assistant.
Use the retained [policy](evidence/inputs/policy.json) and
[measurement summary](reference.json) with the replay commands below to explain
that decision.

To check this result yourself, use [offline replay](#replay-with-the-core-package)
with Python, an installed InvarLock package and matching example files. You do
not need model weights, a GPU or Harness. The expected result is a verified
**rejection**, followed by a report explaining why.

## Result and policy

| Measurement | Result |
| --- | ---: |
| Complete pairs | 400 |
| Baseline mean normalized NLL | 0.544636752 nats/UTF-8 byte |
| Subject mean normalized NLL | 0.594286819 nats/UTF-8 byte |
| Subject/baseline ratio | 1.091161802 |
| Paired 95% ratio interval | [1.084563591, 1.098064743] |
| Decision | **regression** |

Before inference, the illustrative policy required all 400 pairs, an upper ratio
interval bound at most 1.05 and interval width at most 0.10. It uses 2,048 paired
bootstrap replicates. Lower NLL means the model assigned greater probability to the specified
reference text. The subject's mean NLL is about 9.1% higher, exceeding the
5% tolerance. This policy illustrates a release gate; it is not a universal
business threshold. No threshold or case was changed after measurement.

Verification reproduced the rejection and authenticated its receipt. Reporting
preserved every signed pack file. A valid receipt proves the recorded rejection
was replayed; it does not mean the subject passed the policy.

## Data and measurement

The source is the existing
[400-passage LAMBADA subset](../../../integrations/evaluator_transaction/lambada_qwen35_deployment_400.jsonl).
The protocol retains its upstream revision, source hashes, license metadata and
original selection criteria. It inherits that subset's Qwen tokenizer eligibility
and length stratification; it was not selected using either Mistral model's
measurements.

For each original passage, concatenate its retained prompt and expected word
without editing text. Split at the whitespace boundary nearest two-thirds of its
UTF-8 bytes, requiring at least 64 bytes on each side; ties choose the earlier
boundary. The original whitespace belongs to the reference continuation. All
400 passages qualify, preserving their IDs, order and exact text. References span
66–212 bytes. This derived continuation task is different from the official
LAMBADA final-word benchmark.

Harness receives the exact context and reference continuation. It returns the
sum of reference-token log probabilities and a greedy-match flag. No answer is
generated or invented. Each retained row includes the native return, token IDs,
context/reference digests, byte and token counts, model artifact identity,
tokenizer identity and configuration digest. InvarLock recomputes per-case
normalized NLL, the ratio, interval and policy decision from those facts.

SentencePiece can remove leading whitespace when decoding a continuation alone.
The capture therefore checks exact context and complete-text round trips, the
unchanged token prefix and the resulting contextual continuation. It never
strips the reference to make a check pass. The largest complete input is 168
tokens, below the declared 512-token context cap; no record is truncated.

The record-level interval describes this fixed subset. It does not establish
independence across source books, population-wide effects or broad model quality.

## Replay with the core package

Obtain the matching source checkout and install the candidate core wheel as in
[getting started](../../../../docs/user-guide/getting-started.md). Take
`reference.json` from an independently trusted copy of this repository; do not
derive approved pins from an untrusted submitted pack.

From the checkout, reconstruct recipient inputs without models, evaluator
packages, provider credentials or inference:

```bash
python -I examples/captured-results/harness_model_handoff.py \
  --capture examples/captured-results/references/mistral-7b-likelihood/capture \
  --protocol-sha256 sha256:245ac509709ed5741c97078a7e6a660cf3dcb7d429de0501a2082202f332d324 \
  --baseline-sha256 sha256:6f84774a5181cfafd6c2ac1eca9ec113f8b0e3ab8d9a7af4b34a38f27f7dae02 \
  --subject-sha256 sha256:82df7a2a4c59d673f9cb2a271398dacc46edf65718d6251776ac6946870ffba5 \
  --output mistral-recipient
invarlock evaluate --keygen mistral-recipient/keys
```

The handoff checks the original observations and independently reconstructs the
policy, paired runs and request. Build recipient-owned trust from those outputs
and the separately approved signer:

```python
import json
from pathlib import Path

root = Path("examples/captured-results/references/mistral-7b-likelihood")
reference = json.loads((root / "reference.json").read_text())
recipient = Path("mistral-recipient")
derived = json.loads((recipient / "anchors.json").read_text())
anchors = {key: derived[key] for key in reference["anchors"]}
assert anchors == reference["anchors"]
trust = {
    "format": "invarlock/trust-inputs-v2",
    "kind": "captured",
    "policy": {"path": "policy.json"},
    "anchors": {
        **anchors,
        "evidence_signer_fingerprint": reference["signer_fingerprint"],
    },
    "verifier": {
        "identity": "local-reviewer",
        "signing_key_path": "keys/private.pem",
    },
}
(recipient / "trust.json").write_text(json.dumps(trust))
```

```bash
invarlock verify examples/captured-results/references/mistral-7b-likelihood/evidence \
  --trust-profile mistral-recipient/trust.json \
  --receipt mistral-recipient/verification.receipt.json
invarlock report examples/captured-results/references/mistral-7b-likelihood/evidence \
  --html mistral-recipient/report.html
```

Verification must exit **7**, with intact evidence, completed replay and the
expected regression. The report command succeeds and shows both model identities.
Keep the recipient private key private. The repository contains only public keys.
The automated reference tests additionally authenticate the retained receipt,
reject substituted trust pins and confirm the pack remains unchanged.

## Reproduce model capture

Use a separate evaluator environment and the capture script from this source
revision. The exact package versions and implementation hashes are retained in
each capture manifest. Stage only the model files enumerated in the protocol;
symlinks to complete local snapshot files are accepted and their target bytes
are hashed. The script downloads nothing and refuses remote model code.

```bash
python examples/captured-results/harness_model_comparison.py \
  --protocol examples/captured-results/references/mistral-7b-likelihood/capture/baseline/protocol.json \
  --protocol-sha256 sha256:245ac509709ed5741c97078a7e6a660cf3dcb7d429de0501a2082202f332d324 \
  --baseline-model /absolute/path/to/staged-baseline \
  --subject-model /absolute/path/to/staged-subject \
  --output fresh-capture --preflight
```

Preflight validates all text/token boundaries without loading model weights or
making inference calls. Use a different output directory for actual capture and
run each `--role baseline` and `--role subject` in a separate process, omitting
`--preflight`. The profile fixes Apple MPS, float16 weights, float32 log probability
calculations,
batch size one, seed zero, deterministic algorithms and four CPU threads. It
disables caches and asynchronous weight loading. Each role has a one-hour cap
and an MPS memory fraction of 0.42, or 20.16 GiB on the measured host. These are
this rehearsal's resource settings, not InvarLock-wide evaluation limits.

The completed baseline and subject captures took about 171 and 163 seconds,
including inventory checks and loading. These local observations are not a
production latency guarantee. Two earlier processes stopped during loading and
produced zero measurements: an asynchronous MPS copy crash and a subject memory
cap failure. The former required the supported
[serial-loading setting](https://github.com/huggingface/transformers/issues/48029)
in a new protocol before measurement. The latter succeeded in a fresh subject
process under the identical frozen protocol and cap. The baseline's completed
results were reused. [Failure details](initial-load-failure.json) preserve their
identities and log hashes; no measured result was rerun or excluded.

## Evidence scope

The installed-wheel journey used a core-only environment outside the checkout.
The recipient independently reconstructed approved runs and policy in a separate
directory and generated its own verification key. The operator and recipient
were controlled by the same project; this is not validation by an independent
organization.

Signature verification and replay bind the retained observations, model labels
and derived decision. They do not attest that an untrusted source actually ran
those checkpoints, or recompute model inference without weights. Runtime versions
and implementation inventories remain captured source assertions. This is an
external likelihood integration reference, not native runtime qualification,
judge-quality qualification or support for every evaluator/scorer combination.
