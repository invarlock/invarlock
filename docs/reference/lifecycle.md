# Evaluation lifecycle and failure boundaries

The evaluate, verify, and report transactions have deliberately separate read
and write boundaries.

!!! info "Reference"

    - **Surface:** Transaction stages, write boundaries, failure classification, and outputs
    - **Stability:** Behavioral contract for public evaluate, verify, and report transactions
    - **Use this page when:** Diagnosing where a transaction stopped, designing retries, or determining which output may exist after failure

```text
request parsed
  -> inputs authenticated
  -> JSONL prepared or canonical schedule imported
  -> runtime integrations executed or sidecars imported
  -> pairs, interval, and report re-derived
  -> candidate pack validated
  -> manifest signed
  -> destination published once
  -> independently verified
  -> optionally rendered
```

Only the publication transition creates the evidence directory. A failure at
an earlier transition leaves no partial public destination.

## 1. Load the request

The request loader:

- accepts one JSON-compatible YAML object no larger than 1 MiB;
- rejects aliases, tags, duplicate keys, merge keys, include-like directives,
  non-canonical scalars, excessive depth, and excessive node count;
- anchors every file reference beneath the real request directory;
- rejects absolute, URL, traversal, backslash, symbolic-link, and unsafe path
  components; and
- checks provider availability and metric capability.

The transaction repeats no-follow reads later and detects inputs that change
while being read. Request validation is therefore not the only filesystem
check.

## 2. Execute or import

### Run mode

Run mode requires an artifact `path` for each side and a digest-pinned local
JSONL dataset object. InvarLock preserves source order, resolves or derives
record IDs, authenticates each input, and prepares the canonical schedule. The
caller supplies runtime resources separately from the request. The host CLI
launches one independently pinned Docker or Podman worker for each side. Each
worker receives only its read-only job, canonical schedule, artifact, and
closed support resources, plus one isolated writable output directory. It emits
a report, runtime manifest, artifact identity, provider receipt, scoring
observation, and canonical runtime configuration. The host authenticates those
six files before deriving or signing the comparison.

If either side cannot produce complete strict evidence, the overall evidence
destination is not published.

For each side, provider execution follows one lifecycle:

1. validate the closed runtime spec;
2. authenticate the artifact and caller-owned resources;
3. open a session inside the strict runtime context;
4. score the canonical schedule in order;
5. obtain a provider receipt bound to the completed observation;
6. write and authenticate the side report and runtime manifest; and
7. close the session on success or failure.

Two CPU workers may run in parallel. Two workers assigned to different explicit
CUDA indexes may also run in parallel. Generic CUDA selection, the same explicit
CUDA index, and a CPU/CUDA pair run sequentially to avoid GPU contention.
Pairing depends on the authenticated schedule and record IDs, not on worker
concurrency.

### Import mode

Import mode requires, for both sides, an artifact identity, provider receipt,
scoring observation, runtime-side report, runtime manifest, and runtime config.
It also requires the canonical schedule and paired-record file.

The engine does not trust imported aggregates. It reproduces artifact identity
from the request/provider, authenticates receipt and runtime bindings, derives
paired records from observations and the schedule, and requires the imported
paired file to be canonical and exactly equal to that derivation.

Run and import therefore converge on the same bundle and verifier semantics.

| Required import material | Fresh verifier/evaluator action |
| --- | --- |
| Artifact identity | Reproduce through the selected provider and request spec |
| Provider receipt | Validate capabilities, artifact, settings, device, image, and observation binding |
| Scoring observation | Validate canonical bytes, schedule binding, record order, and facts |
| Side report | Reconstruct exactly from authenticated sidecars |
| Runtime manifest/config | Recheck strict runtime and sibling digests |
| Paired records | Re-derive from schedule and both observations, then require exact equality |

## 3. Derive the comparison

The engine requires both observations to have the same ordered record IDs and
input digests as the schedule. It derives built-in record scores or replays the
authorized scorer, then derives means, the comparison value, paired interval,
optional count/interval-width qualification, and policy verdict from the
relevant lower or upper bound and any configured sample controls. New
transactions write `invarlock/comparison-report-v2`; verification preserves
the report-version-specific arithmetic of signed v1 evidence. The comparison ID is
deterministically derived from normalized intent, both artifact identities,
schedule, policy, runtime digests, and paired records.

## 4. Publish once

Publication validates the complete candidate in a private staging directory,
writes canonical payloads and checksums, signs the manifest, and atomically
renames the directory into place. The destination and a sibling publication
lock are no-clobber. A failed publication removes staging and leaves no partial
evidence directory.

Published permissions are read-only, and all later outputs remain external.
See [Evidence artifacts](artifacts.md) for the exact inventory.

The atomic rename protects visibility, while checksums, the signed manifest,
and later snapshot verification protect integrity. Read-only permissions are
defense in depth; they are not accepted as cryptographic proof.

## 5. Verify independently

Verification requires independent baseline and subject artifact digests,
canonical schedule digest, policy, both runtime digests, evidence signer,
verifier identity, and verifier-key inputs. It replays the entire comparison
rather than trusting stored scores or verdicts.

A completed verification writes a no-clobber, read-only signed receipt outside
the pack. This includes rejection receipts: integrity or policy failure remains
a nonzero command result, but the verifier can still authenticate what it
rejected and under which anchors.

Structural failure that prevents the verification transaction from reaching a
completed result may prevent receipt creation. Automation must check both exit
status and expected receipt presence.

Verification snapshots the submitted pack before replay so subsequent reads
refer to the same regular-file inventory. It validates syntax and signatures
before depending on semantic content, then reconstructs identities, pairings,
scores, interval, report, and policy decision under the independent anchors.

## 6. Render the authenticated report

Reporting authenticates signed evidence bundle integrity and renders its closed
canonical report. It can write one new HTML file outside the pack. Independent
verification and its signed receipt remain the acceptance record.

## Failure classification

| Boundary | Representative failure | Result |
| --- | --- | --- |
| Request | Unsafe path, unknown field/provider, unsupported metric | No execution, no bundle |
| Runtime | Missing worker image, unpinned image, network enabled, incomplete side result | No bundle |
| Import | Sidecar mismatch, records not equal to derivation | No bundle |
| Publication | Destination exists, signing failure, parent changes | No partial destination |
| Verification integrity | Signature, checksum, inventory, runtime, pairing, or report mismatch | Nonzero; rejection receipt when verification completed |
| Verification policy | Replayed report verdict is `fail` | Integrity may be true, but command is nonzero and receipt records failure |
| Reporting | Unauthenticated or malformed bundle, existing HTML destination | No rendered output and no bundle mutation |

Retries should use a new output or receipt path after correcting the underlying
input. InvarLock never treats overwriting an earlier result as a retry.

## Transaction outputs

| Transaction result | Evidence directory | Receipt | HTML | Exit |
| --- | --- | --- | --- | --- |
| Evaluation succeeds | New immutable directory | None | None | `0` |
| Evaluation fails | Absent | None | None | Nonzero |
| Verification accepts | Unchanged | Signed success receipt | None | `0` |
| Verification completes rejection | Unchanged | Signed rejection receipt | None | Nonzero |
| Verification cannot complete safely | Unchanged | May be absent | None | Nonzero |
| Reporting succeeds | Unchanged | None | Optional new file | `0` |
| Reporting fails | Unchanged | None | Absent or no replacement | Nonzero |

For forensic retention, preserve the immutable pack and any external signed
receipt together. Do not modify either to add annotations; store operator notes
as separate records that identify their target by manifest or receipt digest.

## Related documentation

- [Architecture](architecture.md) defines the component and trust boundaries
  behind each lifecycle stage.
- [Command-line interface](cli.md) maps transaction outcomes to public exit and
  write behavior.
- [Evidence artifacts](artifacts.md) describes the directory published at the
  evaluation boundary.
- [Reports and receipts](reports.md) interprets success and rejection outputs.
