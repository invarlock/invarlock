# Verify a ModelKit at the point of use

> **User guide**
>
> **Outcome:** Check the exact delivered package, both actual model directories,
> signed evaluation evidence, and the recipient's current acceptance policy.
>
> **Audience:** Model publishers and recipients integrating an artifact handoff.
>
> **Prerequisites:** Python 3.12 or newer, the installed
> InvarLock wheel, a separately reviewed copy of the example script, complete
> local package blobs, and independently selected trust inputs.

Use the [matching wheel and examples](getting-started.md#matching-wheels-and-examples).
The example uses the public InvarLock verifier and needs no service account. Its result is a check at the time
of invocation. A deployment consumer must use the checked directories and prevent
later writes or substitutions before loading them.

This handoff consumes native pack-v1 evidence and its native acceptance
attestation. Captured packs and judge envelopes are different assurance
contracts and cannot be substituted, even when their signed decisions pass.

## Select the package and the contents separately

A ModelKit digest identifies the exact OCI manifest bytes. It does not identify
the contained model in InvarLock's checkpoint format. Changing a package description
or switching from tar to gzip changes the package digest while preserving the
model contents. Changing a weight, tokenizer, or configuration file changes the
model contents and requires matching evaluation evidence.

The example verifies the manifest, config, stored layer digest, decompressed
archive digest, and every extracted file. It compares the complete file inventory
with the actual candidate directory, including operational files that the core
checkpoint identity intentionally excludes. For a Hugging Face directory, it also recomputes the normal
`hf_snapshot_tree_sha256` identity. For a GGUF model, the recipient supplies an
`artifact_file` relative to that directory, and the content identity is the exact
file's SHA-256. Additional files still belong to the complete package inventory;
a GGUF file digest does not authenticate those files by itself. Neither a package label nor a mapping supplied
by a package author becomes an acceptance authority.

The supported packaging subset is KitOps **1.15.0**, one embedded model directory,
one tar or gzip-compressed tar model layer, manifest schema version 2, and `Kitfile`
manifest version `1.0.0`. Only regular files and directories in ordinary tar
headers are accepted. Model parts, references, additional code or dataset layers,
raw single-file packages, `zstd`, sparse files, links, and PAX/GNU extension headers
fail explicitly. Extended metadata is rejected before the tar parser reads its
payload. Keep all model assets inside the selected model directory.

## Produce a portable package

Use a reviewed KitOps 1.15.0 executable and check its binary checksum against the
release asset. The source revision is
`6b8162ae5da4d46f1d2af2beb43e7fb077f052f4`. Prepare this `Kitfile` beside `model/`:

```yaml
manifestVersion: 1.0.0
package:
  name: release-candidate
model:
  path: model
```

```bash
kit --config publisher-store pack package-context \
  --tag registry.example/models/candidate:review
kit --config publisher-store inspect registry.example/models/candidate:review
```

Freeze the returned SHA-256 manifest digest. Retain its exact raw blob, the config
blob, and the model-layer blob. The CLI's local content store holds them under
`publisher-store/storage/blobs/sha256/`, named by the hexadecimal digest. A portable
recipient directory can contain just these blobs; the verifier does not consult
tags, registry credentials, a publisher cache, or the CLI's rendered manifest.

Perform this for both baseline and subject. Repackaging is a separate operation
from model transformation: retain the actual transformation's command, source and
output identities, runtime, and configuration with the evaluation provenance.
This example checks package-to-content binding; it does not independently
establish that the declared transformation was executed.

## Prepare independent recipient inputs

The recipient selects the package digests, expected model-content digests,
technical artifact-identity and runtime digests, schedule, evidence signer, trusted
envelope keys, evaluated policy, and current recipient policy through its own
review process. Copying those choices from the incoming package provides no
independent trust decision.

Create `recipient.json` using the closed
[recipient schema](https://github.com/invarlock/invarlock/blob/main/examples/integrations/modelkit-handoff/recipient.schema.json).
Paths are relative to that file. Its fields are:

| Field | Recipient input |
| --- | --- |
| `format` | `invarlock/example-modelkit-recipient-v1` |
| `sides.baseline`, `sides.subject` | Each contains `blobs`, `package_digest`, `candidate`, and `content_digest`; optional `artifact_file` selects a GGUF member relative to `candidate` |
| `evidence` | The complete signed evidence directory |
| `technical_policy` | The independently selected evaluated policy JSON |
| `technical_anchors.artifact_digests` | Typed artifact-identity digests for `baseline` and `subject`; these differ from model-content digests |
| `technical_anchors.runtime_digests` | Runtime digests for `baseline` and `subject` |
| `technical_anchors.schedule_digest` | Independently selected evaluation schedule digest |
| `technical_anchors.evidence_signer_fingerprint` | Trusted evidence signer fingerprint |
| `technical_anchors.request_digest` | Optional independently selected SHA-256 digest of the complete canonical normalized evaluation request |
| `envelope` | The signed acceptance DSSE envelope |
| `recipient_policy` | Current recipient policy, including signer status and freshness limits |
| `trusted_public_keys` | A nonempty mapping from independently trusted fingerprints to public-key paths |
| `limits` | Optional positive resource ceilings described below |

Supply `request_digest` when acceptance requires the exact evaluated context,
including the request's generation settings, scorer configuration, security
settings, and observation payload references. The example passes this expectation
to the public verifier and rejects a mismatch even when both package mappings
and signatures are valid. It does not infer the expected digest from the incoming
evidence. Omitting the field preserves the existing artifact, runtime, schedule,
policy and signer checks; omission does not establish an independently selected
complete-request match.

The current replay checks this expectation against the evidence manifest also
bound by the acceptance envelope's receipt. It does not retroactively claim that
an older receipt recorded a request anchor. Requiring a particular verifier
trust profile remains a separate recipient-policy choice.

The actual `candidate` paths must be directories with no symlink components.
For GGUF, set `artifact_file` to a safe relative filename such as `model.gguf` and
set `content_digest` to that file's SHA-256. The selected path must be a regular
file ending in `.gguf`. Traversal, links, missing files and other formats fail
closed. Both authenticated acceptance-predicate digest kinds must match the
selected directory or file identity; a package author cannot switch that meaning.
For a delivered ModelKit, unpack its frozen digest into private staging, then
verify those actual paths. Keep staging inaccessible to untrusted writers during
verification and model loading. The verifier extracts a separate temporary copy
for comparison and removes it after checking; it does not publish a deployment.

```bash
python examples/integrations/modelkit_handoff.py --request recipient.json
```

To run outside the checkout, copy the reviewed `modelkit_handoff.py` from the
same source artifact into a recipient directory and run it against the installed
public wheel. The script imports only the standard library and InvarLock.
Retain its source checksum with your integration. It is example code, not a
new installed CLI or a compatibility promise for arbitrary ModelKit layouts.

## Interpret the result

The JSON result keeps package/content mappings, technical integrity, technical
policy verdict, envelope authentication, envelope-to-evidence binding, and current
acceptance separate. It cross-checks the envelope's embedded receipt against the
exact evidence manifest that the verifier replayed, then checks both actual
candidate directories again before returning.

| Exit | Meaning |
| --- | --- |
| `0` | Package and content checks, technical replay, and current recipient acceptance all passed |
| `1` | Authentic bound material was rejected by technical or current recipient policy |
| `2` | Inputs were invalid, unsupported, altered, incomplete, unauthenticated, or inconsistently bound |

An expired or revoked acceptance may fail while the historical technical result
still passes. Preserve both results. A successful check does not assert production
quality, general model safety, delivery receipt, or permission to replace the
recipient's deployment controls.

Default limits are 2 MiB per JSON document, 160 GiB per stored blob, decompressed
archive and model contents, and 200,000 archive members or candidate entries.
The optional keys are `max_json_bytes`, `max_blob_bytes`, `max_archive_bytes`,
`max_model_bytes`, and `max_members`. Choose smaller limits for smaller models.
Archive and model bytes are streamed, but verification needs temporary disk space
for the stored layer, decompressed archive, and extracted model. Large models also
require repeated content reads. Apply an outer execution deadline and disk quota
appropriate to the expected artifact sizes.

## Transfer actual evaluated models and run an inference smoke

The real journey is a separate launcher from the synthetic serialization tests.
It requires actual model files, native signed evidence for those exact files,
an acceptance envelope, and independently selected technical and recipient
policies. The subject must be an executable GGUF model. A published quantized
sibling and its baseline at original precision must each have a content identity;
do not substitute evidence from a different model revision or conversion.
Prepare the comparison through the [GGUF runtime example](https://github.com/invarlock/invarlock/blob/main/examples/integrations/gguf-llama-cpp/README.md)
and retain its transformation provenance. A passing small sample establishes only
the policy and precision stated in that sample, not a broader model qualification.

Prepare `source-recipient.json` with the fields described above. Its `candidate`
paths point to the actual source model directories. The source-side `blobs` and
`package_digest` fields may be omitted: the launcher generates and freezes those
package identities before transfer. All technical anchors, content digests,
policies and public keys must already be selected independently. Keep private
signing keys outside the package and recipient directories.
The operator-owned request may select absolute source paths or relative paths
that stay beside the request file. The launcher rejects selected source
symlinks, scans copied directories for links, and requires each trusted public
key to have a SHA-256 fingerprint before transferring files. Do not run a
recipient request supplied by an untrusted party.

Use a separate Python environment containing the installed InvarLock wheel for
recipient verification. Build or select a reviewed local GGUF runtime image and
freeze its immutable image ID in the selected Docker or Podman engine. The
image used for evaluation is authenticated by
the evidence's runtime anchors; the separate smoke image establishes only the
bounded inference recorded by this launcher. Reserve disk for the source models,
publisher blobs, recipient blobs and unpacked models, plus temporary verifier
archives. A pair of 7B models at original and quantized precision can need over
100 GiB of free space.

Run from the matching example checkout, with an output directory outside it:

```bash
python -m examples.integrations.modelkit_real_journey \
  --request source-recipient.json \
  --kit /tools/kit --kit-sha256 REVIEWED_BINARY_SHA256 \
  --python /environments/recipient/bin/python \
  --container-engine docker \
  --smoke-image sha256:REVIEWED_LOCAL_IMAGE_ID \
  --container-llama /opt/llama.cpp/llama-completion \
  --output /artifacts/modelkit-handoff
```

Use `--container-engine podman` for Podman. The default is `docker`; the
launcher uses only the selected engine and does not fall back to another engine.
Supply the full `sha256:` image ID from that engine's local image store; mutable
tags and shortened IDs are rejected. Podman inspection may omit the `sha256:`
prefix, which is normalized before checking the exact selected image identity.

The launcher packs both actual models with KitOps 1.15.0, freezes their original
manifest digests, changes each tag through a metadata repack, and copies the
content store and evidence to a separate recipient directory. The recipient
unpacks by the original digest, verifies the complete package/content/evidence
and acceptance binding, and confirms that the repack has different package
identities but the same model identities. It rejects a changed candidate,
missing package blob, altered evidence, wrong runtime anchor and revoked signer.
It then checks acceptance again and loads the exact unpacked subject for up to
eight generated tokens by default. The Docker or Podman smoke uses CPU execution,
no network, a non-root user, a read-only model mount, and explicit resource limits.

`result.json` records the package digests, individual acceptance decisions and
actual generated output, selected container engine and immutable smoke image ID.
`logs/` retains commands, exit codes, runtime identity and diagnostics. `recipient/recipient.json` is a portable input to the verifier.
A failed check stops the journey; inspect its saved diagnostics, restore the
expected input, and rerun with a new output directory. The launcher does not
contact an external registry or establish interoperability with a hosted service.

## Review a retained real handoff

The [Mistral-7B ModelKit reference](https://github.com/invarlock/invarlock/blob/main/examples/integrations/modelkit-handoff/references/mistral-7b/README.md)
records an actual BF16 baseline and published Q4_K_M subject, 128 fixed short
LAMBADA cases, and real KitOps 1.15.0 packaging and independent recipient checks.
The scores were 94/128 and 93/128. The paired 95% interval extended from −4.10336
to +2.49680 percentage points, so it failed the declared −2 pp lower-bound
requirement. This does not establish degradation; it fails to rule out a decline
larger than the allowance. No absolute accuracy floor was declared.

The reference separates GGUF file hashes from original and repacked ModelKit
manifest identities, retains explicit caller policy and pair anchors, and
preserves both valid package bindings and the rejected comparison. Its short
Docker and Podman generation checks used the delivered subject bytes through
`--smoke-on-policy-rejection`; they do not imply acceptance or a second 128-case
Podman evaluation.

The compact replay independently verifies historical evidence, the original
receipt and DSSE. Exit zero means that the recorded rejection reproduced
correctly. Repeating the physical package-content checks requires the original
model files and complete selected package blobs listed in the reference; a fresh
recipient must supply its own current policy, trust inputs and expected pair.

## Exercise the supported boundary

The real CLI test packs synthetic fixture bytes with the pinned executable,
repackages them using gzip, copies the content store to an independent recipient,
unpacks the original digest after the tag changes, and rejects later candidate
replacement. It runs no model inference.

```bash
INVARLOCK_KIT_BINARY=/tools/kit \
INVARLOCK_KIT_BINARY_SHA256=REVIEWED_BINARY_SHA256 \
  python -m pytest -q tests/integration/test_modelkit_cli.py
```

The fast example tests also exercise signed evidence and acceptance, malformed
metadata, path traversal, links, duplicate members, unreadable directories,
resource limits, and mutations during verification. Retained historical evidence
is not modified by these tests.

See the [KitOps 1.15.0 release](https://github.com/kitops-ml/kitops/releases/tag/v1.15.0),
the [pinned model format implementation](https://github.com/kitops-ml/kitops/blob/6b8162ae5da4d46f1d2af2beb43e7fb077f052f4/pkg/artifact/kitfile.go),
and [recipient acceptance attestations](../reference/acceptance-attestations.md).
