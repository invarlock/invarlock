# Troubleshooting

InvarLock fails closed: a missing binding, unverifiable input, or ambiguous
destination stops publication or produces a rejected verification result.

!!! tip "User guide"

    **Outcome:** Identify the earliest failing transaction boundary and recover
    without mutating signed evidence or weakening independent verification.

    **Audience:** Evaluation operators, verifier operators, runtime integrators,
    and decision owners investigating an evaluation, verification, or report error.

    **Prerequisites:** The original command and structured error, unchanged
    submitted artifacts, independently sourced anchors, and knowledge of which
    output paths existed before the attempt.

## Triage by transaction boundary

Start with the first boundary that failed. Later artifacts may not exist and
should not be synthesized.

| Boundary | Typical signal | Expected output state |
| --- | --- | --- |
| Request loading | Schema, YAML, path, provider, or metric error | No provider execution and no evidence directory |
| Provider execution | Artifact identity, runtime, backend, device, or record error | No published evidence directory |
| Import authentication | Sidecar, schedule, runtime-manifest, or pair mismatch | No published evidence directory |
| Publication | Signing, staging, parent identity, or destination error | No partial published destination |
| Verification | Nonzero result with integrity, policy, or anchor errors | Bundle unchanged; signed rejection receipt when the transaction completed |
| Reporting | Invalid bundle or existing HTML destination | Bundle unchanged; no new HTML |

Use machine-readable output for the first two core transactions:

```bash
invarlock evaluate request.yaml --signing-key evidence-signer.pem --json
invarlock verify evidence/ \
  --policy trusted/acceptance.json \
  --expected-baseline-artifact sha256:PINNED_BASELINE_ARTIFACT_DIGEST \
  --expected-subject-artifact sha256:PINNED_SUBJECT_ARTIFACT_DIGEST \
  --expected-schedule sha256:PINNED_CANONICAL_SCHEDULE_DIGEST \
  --expected-baseline-runtime sha256:PINNED_BASELINE_DIGEST \
  --expected-subject-runtime sha256:PINNED_SUBJECT_DIGEST \
  --expected-signer sha256:AUTHORIZED_EVIDENCE_SIGNER_FINGERPRINT \
  --receipt verification.receipt.json \
  --verifier-signing-key verifier.pem \
  --verifier-identity release-verifier \
  --json
```

Read the earliest error first. A downstream checksum, report, or policy error
may be a consequence of an earlier identity or binding failure.

## Safe recovery pattern

1. Preserve the submitted evidence and any signed rejection receipt.
2. Classify the failure as request, runtime, import, publication,
   verification, or presentation.
3. Correct the source input or environment, not the signed artifact.
4. Use a new evidence, receipt, or HTML destination.
5. Repeat independent verification from the original independently maintained anchors.

Do not turn a mismatch into a pass by copying the observed digest or signer
from the failed bundle into verifier configuration.

## `an Ed25519 evidence signing key is required`

Pass a PKCS8 Ed25519 private key with `--signing-key` or
`INVARLOCK_SIGNING_KEY`. Other key algorithms are rejected.

For the offline example:

```bash
cd examples
python generate_keys.py --output-dir .keys
invarlock evaluate request.yaml --signing-key .keys/evidence-signer.pem
```

## Output or receipt destination already exists

Evidence publication and receipt creation are no-clobber operations. Choose a
new destination in the request or verify command. For a disposable tutorial,
start from a fresh copy of `examples/`. Do not remove or mutate an artifact that
has already been distributed.

HTML rendering is also no-clobber. Choose a new `--html` path.

## Request path is rejected

Request file references must be nonempty, relative, root-confined paths. They
cannot contain absolute prefixes, URLs, backslashes, `.` or `..` segments,
duplicate separators, control characters, leading/trailing whitespace, or a
trailing slash. Symbolic links and identity changes during the transaction fail
closed. Put all request inputs beneath one real directory and use forward-slash
relative paths.

## Run-mode JSONL preparation fails

Run mode authenticates the exact JSONL source before deriving the canonical
schedule. Check that:

- `comparison.dataset.sha256` is the bare lowercase SHA-256 of the source bytes;
- the file contains one nonempty JSON object per line and no blank lines;
- `input_field`, `expected_output_field`, and optional `id_field` name fields
  present on every selected record;
- IDs are unique when `id_field` is supplied;
- `limit`, when present, selects an available exact prefix; and
- the source file and its parent path are regular, root-confined, and unchanged
  during the read.

Without `id_field`, InvarLock derives stable IDs as
`record/{source_index:08d}` while preserving source order.

## Import schedule bytes are not canonical

Import mode requires the exact compact, key-sorted UTF-8 serialization that
InvarLock authenticates. Rebuild it with
`build_runtime_behavioral_schedule_from_material` and
`canonical_runtime_behavioral_schedule_json`; do not pretty-print it afterward.

Also confirm:

- every text-part digest hashes its UTF-8 text and every `input_sha256` hashes
  the canonical ordered `input_parts` array;
- record IDs are unique;
- record order has not changed; and
- `comparison.dataset` and import-mode `execution.schedule` identify the same
  exact bytes.

## Observation or pairing mismatch

Baseline and subject observations must bind the same schedule digest and repeat
the exact ordered `(record_id, input_sha256)` sequence. The imported paired
records must equal InvarLock's derivation byte-for-byte in canonical JSON.
Regenerate provider output from the frozen schedule; do not edit paired scores
to make them agree.

If import mode fails, check the sidecars as one atomic set:

- `model-artifact.identity.json` must equal a fresh identity derived by the
  installed provider from request settings;
- `runtime-provider.receipt.json` must bind that identity, provider,
  capabilities, settings, device, outer image, and observation;
- `runtime-scoring.observation.json` must bind the same schedule and contain
  one successful record per scheduled ID in order;
- `report.json` must bind the provider, identity, observation, schedule, and
  record count;
- `runtime.manifest.json` must name the exact sibling bytes; and
- imported paired records must equal InvarLock's fresh derivation.

Replacing only the file named by the first error usually creates the next
cross-binding error. Re-export the complete side from the original provider
execution.

## Runtime image digest is missing or mismatched

Run mode requires independently known lowercase `sha256:` image digests and
references embedding the corresponding digest. A shared image can be supplied
once; cross-runtime comparisons supply the side-specific image bindings
documented by `invarlock evaluate --help`.

```bash
export INVARLOCK_RUNTIME_IMAGE=registry.example/runtime@sha256:PINNED_DIGEST
export INVARLOCK_RUNTIME_IMAGE_DIGEST=sha256:PINNED_DIGEST
```

The executing container must report the same identity. Do not replace the
expected digest with the value copied from a failed evidence bundle.

During verification, pass the baseline and subject digests from the decision
authority. They may be equal when both sides intentionally used one runtime.

## Provider is unavailable

The base package includes the HF text provider and can authenticate its local
snapshot identity for import mode without Torch or Transformers. Install
`invarlock[hf]` before run-mode execution. GGUF, TensorRT-LLM, and Hugging Face
vision-text providers are separate first-party add-ins and must be installed
and qualified in their required runtimes. A
request naming an undiscoverable provider fails before evidence publication.

Run the add-in's conformance entry point in the same Python environment:

```bash
invarlock-gguf-conformance
invarlock-tensorrt-llm-conformance
invarlock-hf-vision-text-conformance
```

If the command is missing, compare `python -m pip --version`, `which
invarlock`, and the active virtual environment. If conformance passes but
evaluation fails, troubleshoot the concrete artifact, backend resources,
runtime image, and device; conformance does not execute the model.

For GGUF, confirm the artifact, pinned `llama.cpp` executable, and source
archive are the same inputs used to derive request settings. For TensorRT-LLM,
also confirm the engine was inspected on the target compute capability with the
same tokenizer contract and runner. For vision-text, confirm the processor
contract and every content ID, media type, byte length, and digest against the
caller-authorized content store.

## Strict runtime refuses the environment

Strict run mode expects a real container boundary, a digest-bearing image
identity, offline execution, remote code disabled, third-party plugins
disabled, and `batch_size=1`.

Environment switches do not simulate those controls. In particular:

- `INVARLOCK_CONTAINER_EXECUTION=true` does not replace kernel-visible
  container evidence;
- setting `INVARLOCK_ALLOW_NETWORK` makes the environment unsuitable for the
  strict evidence path;
- enabling remote code or arbitrary third-party plugins can make strict
  evaluation reject; and
- a mutable image tag is not a runtime digest.

Invoke the host CLI with the pinned image bindings. It establishes the
network-disabled Docker or Podman workers and the strict child context. If the
host launch fails, inspect local engine availability, exact image presence,
digest agreement, signing-key placement, and device syntax. If a child starts
but rejects the boundary, inspect its image entrypoint, container markers,
offline flags, and runtime-integration resources. See
[Runtime providers](runtime-providers.md) for provider-specific setup.

## Worker output cleanup fails

A failed or interrupted worker can leave temporary files owned by its numeric
user. InvarLock first attempts bounded host cleanup. If those permissions block
removal, it runs a cleanup helper with the same non-root user and pinned image,
disabled networking, and only that side's temporary output mounted writable.
The helper has a 30-second deadline, 128 MiB of memory, and limits of 4,096
entries and 64 nested directory levels. It does not follow symbolic links or
change regular-file permissions.

If cleanup still fails, the diagnostic names the retained private temporary
directory. An existing evaluation error remains the primary error; cleanup
failure is also reported. Check the local engine and exact image availability,
then have the machine operator inspect and remove only the reported temporary
directory using its file ownership. A cleanup failure does not publish a
completed evidence bundle.

## Policy digest mismatch

Verification uses the exact bytes at `--policy`; it never trusts a policy found
only inside submitted evidence. Obtain the approved policy file from the
acceptance authority. Reformatting equivalent JSON changes its digest. If the
policy truly changed, produce a new comparison rather than relabeling the old
bundle.

## Signer fingerprint is invalid or unexpected

Fingerprints must be `sha256:` plus 64 lowercase hexadecimal characters and
must hash raw Ed25519 public-key bytes. Do not hash PEM text. Read the expected
evidence-signer value from a trusted registry or the example's independently generated
`evidence-signer.fingerprint`, not from `manifest.signature.json`.

## Verification fails but a receipt exists

This is expected when enough inputs are available to sign a rejection. Preserve
the receipt: it records the evidence manifest digest, external anchors, verifier
identity, and failed verdict. Inspect the CLI error or `--json` result, correct
the external or evaluation-side cause, and use a new receipt destination for any
new verification attempt.

Never treat the existence of a receipt as a passing result. Check its signed
verdict and independently pin the verifier fingerprint.

## A received receipt does not validate

The stable receipt reader checks six independent relationships: verifier
signature, verifier identity and fingerprint, pack manifest digest, policy
digest, both runtime digests, and evidence-signer fingerprint.

Classify the error before retrying:

| Error family | Likely cause |
| --- | --- |
| Signature or embedded public-key mismatch | Receipt was changed, malformed, or signed by another key. |
| Verifier identity or fingerprint mismatch | Caller trust configuration does not authorize the signer. |
| Manifest digest mismatch | Receipt belongs to another pack or pack bytes changed. |
| Policy/runtime/evidence-signer anchor mismatch | Caller and verifier did not use the same independently maintained trust inputs. |
| Receipt is inside the evidence pack | The closed bundle was modified or packaged incorrectly. |

Do not read expected values from the failing receipt to silence these errors.
Locate the correct evidence/receipt pair and compare both with the independent
authorization record.

## Exact match is unexpectedly zero

Exact match is literal Unicode string equality. Whitespace, case, punctuation,
and line endings are significant. If normalization is part of the desired
metric, define and authenticate it before provider observation; do not normalize
only one side after execution.

## Normalized NLL fails closed

Each successful record needs finite `logprob_sum`, positive `token_count`, and
positive `utf8_byte_count`. InvarLock derives `-logprob_sum / utf8_byte_count`.
The baseline mean must be greater than zero. Confirm both providers use the same
scoring target and byte-count definition before comparing them.

## Perplexity interpretation is unavailable

The verifier derives a token-weighted perplexity ratio only when normalized-NLL
evidence binds matching authenticated tokenizer digests and equal positive
target-token counts for every pair. If those facts differ or are unavailable,
the report records a closed unavailable reason. The byte-normalized NLL decision
remains authoritative; do not copy one side's tokenizer facts into the other.

## Point value appears within policy but the verdict fails

The point comparison is descriptive. Exact match passes only when the paired
Newcombe interval's lower bound meets `delta_min_pp`; normalized NLL passes only
when the schedule-resampling interval's upper bound meets `ratio_max`.
If the policy enables sample qualification, the observed record count must also
meet `minimum_record_count` and the observed width must not exceed the matching
maximum-width field. Inspect `sample_qualification` before changing the metric
threshold. If exact-match policy sets `minimum_side_accuracy`, both observed
side means must also meet that floor; inspect `side_accuracy` before changing
the policy.
Review the authenticated schedule, interval method, and record-level variation.
Changing the threshold or schedule requires a new evidence transaction.

## Report rejects an otherwise visible directory

`report` authenticates the closed inventory before rendering. Extra files,
missing files, changed bytes, non-canonical manifest/report JSON, symbolic links,
or an invalid evidence signature are fatal. Keep receipts and HTML outside the
evidence directory. Recover from an immutable trusted copy; do not repair files
inside a signed bundle.

## Machine-readable diagnostics

Use `--json` for evaluation and verification automation:

```bash
invarlock evaluate request.yaml --signing-key evidence-signer.pem --json
invarlock verify evidence/ ... --receipt receipt.json --json
```

The report command intentionally renders human-facing console or HTML output.
The canonical JSON report remains inside the authenticated bundle.

## Before escalating a bug

Collect the smallest non-sensitive reproduction that preserves the failing
contract:

- InvarLock and first-party add-in versions;
- the command name and structured error text;
- request shape with private locators and content redacted consistently;
- which boundary failed and whether any output was published;
- provider name, ABI, artifact kind, and high-level device class;
- whether the offline example passes in the same installation; and
- a synthetic schedule or sidecar fixture when the failure can be reproduced
  without private model or dataset material.

Do not attach private keys, local absolute paths, access tokens, environment
dumps, proprietary prompts, raw checkpoints, or evidence bundles not approved
for publication to a public issue.

For exact contract shapes, see [Public contracts](../reference/contracts.md).
For trust implications, see the [Threat model](../security/threat-model.md).
