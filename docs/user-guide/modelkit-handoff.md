# Verify a ModelKit at the point of use

!!! tip "User guide"
    **Outcome:** Check the exact delivered package, both actual model directories,
    signed evaluation evidence, and the recipient's current acceptance policy.
    **Audience:** Model publishers and recipients integrating an artifact handoff.
    **Prerequisites:** Python 3.12 or newer, the current source candidate's installed
    InvarLock wheel, a separately reviewed copy of the example script, complete
    local package blobs, and independently selected trust inputs.

The example is available in the unreleased source checkout. It uses the public
InvarLock verifier and needs no service account. Its result is a check at the time
of invocation. A deployment consumer must use the checked directories and prevent
later writes or substitutions before loading them.

## Select the package and the contents separately

A ModelKit digest identifies the exact OCI manifest bytes. It does not identify
the contained model in InvarLock's checkpoint format. Changing a package description
or switching from tar to gzip changes the package digest while preserving the
model contents. Changing a weight, tokenizer, or configuration file changes the
model contents and requires matching evaluation evidence.

The example verifies the manifest, config, stored layer digest, decompressed
archive digest, and every extracted file. It compares the complete file inventory
with the actual candidate directory, including operational files that the core
checkpoint identity intentionally excludes. It also recomputes the normal
`hf_snapshot_tree_sha256` identity. Neither a package label nor a mapping supplied
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
This example checks package-to-content binding; it does not certify a package author's
claim that a transformation was executed honestly.

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
| `sides.baseline`, `sides.subject` | Each contains `blobs`, `package_digest`, `candidate`, and `content_digest` |
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
