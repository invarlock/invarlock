# Command-line interface

The public command line is intentionally limited to one ordered journey:

!!! info "Reference"

    - **Surface:** `invarlock evaluate`, `invarlock verify`, and `invarlock report`
    - **Stability:** Stable public CLI; command help is authoritative for installed options
    - **Use this page when:** Automating a transaction, selecting flags or environment fallbacks, or interpreting outputs and exit status

```text
invarlock evaluate request.yaml
invarlock verify evidence/
invarlock report evidence/
```

Use `invarlock --version` for the installed version and `invarlock --help` for
the authoritative option list. The commands have the same transaction
boundaries as the [Python facade](api-guide.md).

## Root command

| Form | Result |
| --- | --- |
| `invarlock` | Show help; no transaction runs |
| `invarlock --help` | Show the three-command surface and exit |
| `invarlock --version` | Print `InvarLock <installed-version>` and exit |
| `invarlock COMMAND --help` | Show the exact arguments and options for one transaction |

Shell completion is deliberately not installed by the CLI. The root command
also applies the process security defaults used by every transaction before a
command implementation imports a runtime backend.

## `evaluate`

```text
invarlock evaluate REQUEST \
  --signing-key PATH \
  [--allow-installed-scorers] \
  [--runtime-image IMAGE] \
  [--runtime-image-digest sha256:...] \
  [--baseline-runtime-image IMAGE] \
  [--baseline-runtime-image-digest sha256:...] \
  [--subject-runtime-image IMAGE] \
  [--subject-runtime-image-digest sha256:...] \
  [--container-engine docker|podman] \
  [--runtime-device cpu|cuda|cuda:<index>] \
  [--baseline-runtime-device DEVICE] \
  [--subject-runtime-device DEVICE] \
  [--runtime-entrypoint auto|python|nvidia] \
  [--baseline-runtime-entrypoint PROFILE] \
  [--subject-runtime-entrypoint PROFILE] \
  [--json]
```

`evaluate` loads one closed YAML request, prepares or validates its canonical
schedule, executes or imports paired provider records, derives the selected
metric and its paired interval, applies the policy to the conservative bound,
and atomically publishes the evidence directory named by the request.

For a valid run request, the host prepares the canonical schedule and launches
one independently configured Docker or Podman worker for each comparison side.
Each worker loads only its side's artifact and support resources, scores the
same authenticated schedule, and returns a closed six-file side result. The
host validates both results, derives the comparison, signs the manifest, and
publishes the evidence directory. Import requests do not launch workers.

| Input | Required | Environment alternative | Purpose |
| --- | --- | --- | --- |
| `REQUEST` | Yes | None | Existing readable YAML governed by `evaluation_request.schema.json`; its parent is the request root |
| `--signing-key PATH` | Yes | `INVARLOCK_SIGNING_KEY` | Ed25519 evidence-signing private-key file |
| `--allow-installed-scorers` | Only for a scorer-bound request | `INVARLOCK_ALLOW_INSTALLED_SCORERS` | Authorize loading and executing the exact installed scorer bound by the request and policy |
| `--runtime-image IMAGE` | Run mode from host | `INVARLOCK_RUNTIME_IMAGE` | Local OCI image reference; must contain a digest or be paired with the digest option |
| `--runtime-image-digest DIGEST` | When not embedded in image; recommended explicitly | `INVARLOCK_RUNTIME_IMAGE_DIGEST` | Pinned lowercase OCI `sha256:...` identity |
| `--baseline-runtime-image IMAGE` | No | `INVARLOCK_BASELINE_RUNTIME_IMAGE` | Baseline image override; otherwise the common image is used |
| `--baseline-runtime-image-digest DIGEST` | No | `INVARLOCK_BASELINE_RUNTIME_IMAGE_DIGEST` | Baseline image-digest override; otherwise the embedded or common digest is used |
| `--subject-runtime-image IMAGE` | No | `INVARLOCK_SUBJECT_RUNTIME_IMAGE` | Subject image override; otherwise the common image is used |
| `--subject-runtime-image-digest DIGEST` | No | `INVARLOCK_SUBJECT_RUNTIME_IMAGE_DIGEST` | Subject image-digest override; otherwise the embedded or common digest is used |
| `--container-engine ENGINE` | No | `INVARLOCK_CONTAINER_ENGINE` | `docker` or `podman`; defaults to `docker` |
| `--runtime-device DEVICE` | No | `INVARLOCK_RUNTIME_DEVICE` | Shared device: `cpu`, `cuda`, or `cuda:<index>`; defaults to `cpu` |
| `--baseline-runtime-device DEVICE` | No | `INVARLOCK_BASELINE_RUNTIME_DEVICE` | Baseline override |
| `--subject-runtime-device DEVICE` | No | `INVARLOCK_SUBJECT_RUNTIME_DEVICE` | Subject override |
| `--runtime-entrypoint PROFILE` | No | `INVARLOCK_RUNTIME_ENTRYPOINT` | Shared worker entrypoint: `auto`, `python`, or `nvidia`; defaults to `auto` |
| `--baseline-runtime-entrypoint PROFILE` | No | `INVARLOCK_BASELINE_RUNTIME_ENTRYPOINT` | Baseline entrypoint override |
| `--subject-runtime-entrypoint PROFILE` | No | `INVARLOCK_SUBJECT_RUNTIME_ENTRYPOINT` | Subject entrypoint override |
| `--json` | No | None | Emit one compact `invarlock/evaluation-result-v1` object |

The signing key must be a real regular file. The request and every referenced
input must remain beneath the request root. Keep the signing key in a separate,
caller-controlled location. It remains in the host process and is never mounted
into either worker. The output destination must not already exist.

Run-mode `comparison.dataset` is a digest-pinned local JSONL object. The
transaction verifies the source bytes and deterministically prepares the
canonical ordered schedule. Import-mode `comparison.dataset` is the canonical
schedule path and must match `execution.schedule`.

Each OCI worker uses its selected local image only (`--pull=never`), disables
networking, uses a read-only container root, drops capabilities, enables
`no-new-privileges`, bounds process count, and receives a temporary filesystem.
The job description, canonical schedule, side artifact, and closed support
resources are mounted read-only. Only an isolated side-output directory is
writable. The worker never receives the other side's artifact or the signing
key.

CPU workers add no GPU mapping. `--runtime-device cuda` or a per-side CUDA
override exposes the selected GPU to that worker; it does not add CUDA support
to a CPU-only image. Use an image built from `runtime/Dockerfile.cuda` for the
canonical x86_64 CUDA Hugging Face runtime. Two CPU workers may run in parallel.
Two workers on distinct explicit indexes such as `cuda:0` and `cuda:1` may also
run in parallel. Generic CUDA selection, a shared explicit index, or a CPU/CUDA
pair runs sequentially so two workers do not contend for one GPU.

The `auto` entrypoint profile selects `nvidia` for TensorRT-LLM and `python` for
the other first-party providers. Explicit per-side profiles are available when
an authenticated image requires one of those known launch forms.

Text success output identifies the published directory. JSON success output
contains `format_version`, `ok`, `comparison_id`, and `evidence`. JSON failures
use the same format version with `ok: false` and an `errors` array.

```json
{
  "comparison_id": "cmp-...",
  "evidence": "artifacts/evidence",
  "format_version": "invarlock/evaluation-result-v1",
  "ok": true
}
```

The displayed evidence path is the host-side destination from the request.
Treat it as an opaque location; the signed manifest, rather than the path name,
carries the comparison identity.

## `verify`

```text
invarlock verify EVIDENCE \
  --policy PATH \
  --expected-baseline-artifact sha256:... \
  --expected-subject-artifact sha256:... \
  --expected-schedule sha256:... \
  --expected-baseline-runtime sha256:... \
  --expected-subject-runtime sha256:... \
  --expected-signer sha256:... \
  --receipt verification.receipt.json \
  --verifier-signing-key verifier.pem \
  --verifier-identity release-verifier \
  [--allow-installed-scorers] \
  [--json]
```

`verify` treats the bundle as untrusted. It requires all acceptance anchors
from the caller and writes a signed receipt outside the bundle.

| Input | Required | Environment alternative | Meaning |
| --- | --- | --- | --- |
| `EVIDENCE` | Yes | None | Existing readable evidence-pack directory |
| `--policy PATH` | Yes | `INVARLOCK_POLICY` | Independently sourced policy bytes |
| `--expected-baseline-artifact DIGEST` | Yes | `INVARLOCK_EXPECTED_BASELINE_ARTIFACT` | Approved baseline artifact-identity digest |
| `--expected-subject-artifact DIGEST` | Yes | `INVARLOCK_EXPECTED_SUBJECT_ARTIFACT` | Approved subject artifact-identity digest |
| `--expected-schedule DIGEST` | Yes | `INVARLOCK_EXPECTED_SCHEDULE` | Approved canonical schedule digest |
| `--expected-baseline-runtime DIGEST` | Yes | `INVARLOCK_EXPECTED_BASELINE_RUNTIME` | Expected baseline outer-image digest |
| `--expected-subject-runtime DIGEST` | Yes | `INVARLOCK_EXPECTED_SUBJECT_RUNTIME` | Expected subject outer-image digest |
| `--expected-signer FINGERPRINT` | Yes | `INVARLOCK_EXPECTED_SIGNER` | Expected Ed25519 evidence-signer fingerprint |
| `--receipt PATH` | Yes | None | New receipt path outside the pack |
| `--verifier-signing-key PATH` | Yes | `INVARLOCK_VERIFIER_SIGNING_KEY` | Independent verifier Ed25519 private key |
| `--verifier-identity TEXT` | Yes | `INVARLOCK_VERIFIER_IDENTITY` | Stable verifier name placed in the receipt |
| `--allow-installed-scorers` | Only for scorer-bound evidence | `INVARLOCK_ALLOW_INSTALLED_SCORERS` | Independently authorize loading and replaying the exact installed scorer pinned by the request and policy |
| `--json` | No | None | Emit one compact verification result |

`--receipt` has no environment alternative. The destination must be outside the
evidence directory and must not already exist.

Verification checks the closed manifest, evidence signature, checksums and
inventory; caller-approved artifact and schedule identities; all input and
evidence references; the normalized request and schedule; runtime manifests
and provider sidecars; record ordering and input digests; derived paired
scores; the canonical comparison report; and the policy verdict. A policy
verdict of `fail` is a valid, integrity-checked result but is not command
success.

An installed scorer is executable code. The flag does not authorize an
arbitrary scorer: discovery still requires the exact scorer ID, ABI, version,
descriptor digest, configuration digest, task, and policy pins bound by the
transaction. Evaluation and independent verification must authorize and load
the same scorer identity separately.

`--json` emits `evidence-pack-verify-v1` plus the signed-receipt path and
verifier identity. Exit status `0` means the evidence passed both integrity and
policy verification. Any nonzero status must be treated as rejection.

The result distinguishes integrity from acceptance. For example, a correctly
signed and internally consistent pack can have `integrity_ok: true`,
`policy_verdict: "fail"`, and `ok: false`. See [Reports and
receipts](reports.md#verification-result) for the complete field matrix.

## `report`

```text
invarlock report EVIDENCE [--html report.html] [--explain]
```

`report` verifies the bundle's closed inventory, checksums, reference digests,
canonical JSON, and embedded evidence signature before rendering
`reports/evaluation.report.json`.

| Input | Required | Meaning |
| --- | --- | --- |
| `EVIDENCE` | Yes | Existing readable evidence-pack directory |
| `--html PATH` | No | Write a new self-contained HTML report outside the pack |
| `--explain` | No | Add a concise explanation of the decision and evidence bindings |

`--html` refuses to overwrite an existing file. `report` always emits the text
view to standard output; when HTML is requested it also prints the written
path.

Reporting does not accept independent artifact, schedule, policy, runtime, or
signer anchors and does not issue a verification receipt. It is therefore a
safe renderer, not a substitute for `invarlock verify`.

## Exit and write behavior

All three commands fail closed. A nonzero exit indicates that the requested
transaction did not succeed. Evaluation does not publish a partial destination;
verification does not write inside the pack; and reporting does not mutate pack
bytes. Output files and directories are no-clobber by design.

| Status | Public meaning |
| --- | --- |
| `0` | Requested transaction completed successfully |
| `1` | Operational write failure where the command reports one explicitly, such as an HTML output error |
| `2` | Invalid invocation, missing trust input, request/evaluation failure, or high-level verification/report rejection |
| Other nonzero | A lower-level evidence-pack status surfaced by a transaction; reject and inspect machine output |

Do not build automation that accepts a particular nonzero value. The
lower-level `EvidencePackStatus` enum has finer internal categories, while the
public transaction may normalize a precondition failure to `2`. Automation
should require status `0`, the expected output, and `ok: true` where JSON is
available.

## Automation pattern

Use a fresh output name for every immutable transaction and capture JSON on a
dedicated stream:

```bash
invarlock evaluate request.yaml \
  --signing-key evidence-signer.pem \
  --runtime-image registry.example/invarlock-runtime-cuda@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --runtime-image-digest sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --container-engine docker \
  --runtime-device cuda \
  --json
invarlock verify artifacts/evidence \
  --policy policy/acceptance.json \
  --expected-baseline-artifact sha256:... \
  --expected-subject-artifact sha256:... \
  --expected-schedule sha256:... \
  --expected-baseline-runtime sha256:... \
  --expected-subject-runtime sha256:... \
  --expected-signer sha256:... \
  --receipt receipts/verification.json \
  --verifier-signing-key verifier.pem \
  --verifier-identity release-verifier \
  --json
```

The `aaaaaaaa...` value above is illustrative and must be replaced. Ellipses in
the verification command are placeholders, not valid digests. Fingerprints and
all verifier anchors use `sha256:` plus 64 lowercase hexadecimal characters.
Artifact anchors identify the canonical provider artifact-identity bytes, not
a path or mutable model name. The schedule anchor identifies the exact
canonical schedule bytes. Obtain both through a verifier-controlled handoff,
not from the submitted evidence pack. Provider settings and local JSONL source
hashes use bare lowercase 64-character values where their schemas require
them. Keep the receipt and optional HTML report outside the immutable evidence
directory.

See [Environment variables](environment.md) for the complete small CLI and
runtime inventory.

## Related documentation

- [Environment variables](environment.md) lists supported secret, anchor, and
  runtime inputs.
- [Python API](api-guide.md) exposes the same transactions to embedding
  applications.
- [Evaluation lifecycle](lifecycle.md) explains write boundaries and retry
  behavior.
- [Reports and receipts](reports.md) defines the machine and human outputs.
