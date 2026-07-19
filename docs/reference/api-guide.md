# Python API

Applications embed InvarLock through `invarlock.engine`. This module is the
stable facade for the paired evaluation engine. Modules that encode or decode
individual bundle files remain internal unless explicitly documented.

!!! info "Reference"

    - **Surface:** Python facade exported by `invarlock.engine`
    - **Stability:** Stable public API; undocumented `invarlock.*` imports remain internal
    - **Use this page when:** Embedding evaluate, verify, report, request loading, receipt verification, or provider-ABI types in Python

## Stable exports

The facade deliberately groups these stable surfaces:

| Surface | Exports |
| --- | --- |
| Transactions | `evaluate_request_file`, `verify_evidence`, `render_evidence`, `verify_signed_verification_receipt` |
| Request | `load_evaluation_request`, `EvaluationRequest`, `EvaluationRequestError` |
| Results and errors | Evaluation, verification, reporting, receipt, and evidence-pack result types |
| OCI host execution | Per-side launch values, host executor, and environment-backed launch resolution |
| Provider ABI | ABI constant, provider/session protocols, capabilities, runtime specs/resources/context, resolver protocols, batches, identities, receipts, and observations |
| Runtime-import authoring | Strict JSONL record loading, typed observation and receipt builders, complete side publication/reload, and verifier-derived paired-record publication |
| Scorer extension | ABI constant, descriptor and binding types, replay protocol and request/result types, explicit registry, and binding/result builders |
| HF identity | `checkpoint_tree_sha256`, `hf_tokenizer_contract_sha256` |

Imports from other `invarlock.*` modules are not stable merely because they are
importable. Use the facade unless implementing the provider protocol documented
in [Runtime providers](runtime-providers.md).

## Transactions

```python
from pathlib import Path

from invarlock.engine import (
    OciRuntimeExecutor,
    evaluate_request_file,
    launch_from_environment,
    render_evidence,
    verify_evidence,
    verify_signed_verification_receipt,
)

# Load both fingerprints from an independently managed trust registry.
evidence_signer_fingerprint = "sha256:" + "3" * 64
verifier_fingerprint = "sha256:" + "4" * 64
artifact_digests = {
    "baseline": "sha256:" + "5" * 64,
    "subject": "sha256:" + "6" * 64,
}
schedule_digest = "sha256:" + "7" * 64

evaluation = evaluate_request_file(
    Path("request.yaml"),
    signing_key_path=Path("keys/evidence-signer.pem"),
    runtime_executor=OciRuntimeExecutor(launch_from_environment()),
)

verification = verify_evidence(
    evaluation.evidence_path,
    policy_path=Path("policy/acceptance.json"),
    expected_baseline_artifact=artifact_digests["baseline"],
    expected_subject_artifact=artifact_digests["subject"],
    expected_schedule=schedule_digest,
    expected_baseline_runtime="sha256:" + "1" * 64,
    expected_subject_runtime="sha256:" + "2" * 64,
    expected_signer=evidence_signer_fingerprint,
    receipt_path=Path("verification.receipt.json"),
    verifier_signing_key_path=Path("keys/evidence-verifier.pem"),
    verifier_identity="release-verifier",
)

report = render_evidence(
    evaluation.evidence_path,
    html_path=Path("evidence.html"),
    explain=True,
)

receipt_check = verify_signed_verification_receipt(
    verification.receipt_path,
    evaluation.evidence_path,
    policy_path=Path("policy/acceptance.json"),
    expected_artifact_digests=artifact_digests,
    expected_schedule_digest=schedule_digest,
    expected_runtime_digests={
        "baseline": "sha256:" + "1" * 64,
        "subject": "sha256:" + "2" * 64,
    },
    expected_pack_signer_fingerprint=evidence_signer_fingerprint,
    expected_verifier_identity="release-verifier",
    expected_verifier_fingerprint=verifier_fingerprint,
)
assert receipt_check.ok
```

The result types are:

| Type | Important fields |
| --- | --- |
| `EvaluationPreflightResult` | `execution_mode`, `output`, input digests, providers, checks, `as_json()` |
| `EvaluationTransactionResult` | `evidence_path`, `comparison_id`, `pack_manifest_digest`, `as_json()` |
| `EvidenceVerification` | `evidence_path`, `payload`, `receipt_path`, `summary`, `as_json()` |
| `EvidenceReport` | `text`, `html_path`, `evidence_signer` |
| `ReceiptVerification` | `ok`, `signed`, `statement`, `verifier_fingerprint`, `errors` |

`EvaluationPreflightError`, `EvaluationTransactionError`,
`EvidenceVerificationError`, and `EvidenceReportError` carry CLI-compatible
`exit_code` values. The first three also expose machine-readable JSON behavior.
Applications should treat every exception or verification payload with
`ok != true` as rejection.

`EvidenceReceiptError` is also exported for failures that prevent safe receipt
processing. Ordinary receipt authenticity or anchor mismatches are reported in
`ReceiptVerification.errors` with `ok` set to false.

`render_evidence` authenticates the bundle's internal evidence signature but
does not take independent artifact, schedule, policy, runtime, or signer
anchors. Call `verify_evidence` when an independent acceptance decision is
required.

## Function signatures

```python
evaluate_request_file(
    request_path: Path,
    *,
    signing_key_path: Path | None,
    resource_resolver: RuntimeResourceResolver | None = None,
    runtime_executor: OciRuntimeExecutor | None = None,
    runtime_image_digests: Mapping[str, str] | None = None,
    scorer_registry: ScorerExtensionRegistry | None = None,
) -> EvaluationTransactionResult

preflight_evaluation_request(
    request_path: Path,
    *,
    signing_key_path: Path | None,
    scorer_registry: ScorerExtensionRegistry | None = None,
    runtime_image_digests: Mapping[str, str] | None = None,
    resource_resolver: RuntimeResourceResolver | None = None,
) -> EvaluationPreflightResult

load_evaluation_request(
    path: str | Path,
    *,
    provider_resolver: ProviderResolver | None = None,
) -> EvaluationRequest

verify_evidence(
    evidence_path: Path,
    *,
    policy_path: Path | None,
    expected_baseline_artifact: str | None,
    expected_subject_artifact: str | None,
    expected_schedule: str | None,
    expected_baseline_runtime: str | None,
    expected_subject_runtime: str | None,
    expected_signer: str | None,
    receipt_path: Path | None = None,
    verifier_signing_key_path: Path | None = None,
    verifier_identity: str | None = None,
    trust_profile_digest: str | None = None,
    scorer_registry: ScorerExtensionRegistry | None = None,
    policy_bytes: bytes | None = None,
    verifier_signing_key_bytes: bytes | None = None,
) -> EvidenceVerification

render_evidence(
    evidence_path: Path,
    *,
    html_path: Path | None = None,
    explain: bool = False,
) -> EvidenceReport
```

The optional annotations on `verify_evidence` support a uniform Python
signature; the public evidence-pack v1 transaction requires every artifact,
schedule, policy, runtime, signer, receipt, verifier-key, and verifier-identity
input. Missing values raise `EvidenceVerificationError` rather than reducing
assurance.

`evaluate_request_file` always completes the same request, input, schedule,
policy, runtime, key, and output preflight before it calls a run-mode executor.
Every programmatic run must supply both independently inspected
`runtime_image_digests`; neither an executor nor a resource resolver can
self-assert those trust inputs. `preflight_evaluation_request` exposes that gate separately when an
application wants to inspect the result without executing a runtime or mutating
the output tree. Embedders must independently confirm that run-mode images are
local before supplying their digests; the CLI performs that inspection through
Docker or Podman without starting a container. For a scorer-bound request,
preflight also loads the explicitly authorized scorer's deterministic
descriptor and schema and validates the exact binding and task/input/output
compatibility without calling scorer replay.

`EvaluationPreflightResult` serializes as
`invarlock/evaluation-preflight-v2`. When policy sample qualification is
enabled, `sample_qualification.record_count` contains the minimum, observed
count, and `status: pass`; `sample_qualification.interval_width` contains the
maximum, unit, and `status: pending_execution`. Applications must not convert
that pending state into a successful precision claim.

Applications evaluating a run request that selects a runtime input hook must
pass a side-aware `RuntimeResourceResolver` to the public transaction. The hook
authenticates schedule-bound external resources without loading the model.
Successful resolution of both sides against their inspected image digests is
recorded as `runtime_resources` in `EvaluationPreflightResult.checks`. The OCI executor implements both execution
and resource resolution so its frozen per-side image, device, root, and support
bindings cannot drift between baseline and subject validation.

When a request selects `comparison.scorer_extension`, the embedding must pass
an explicit `ScorerExtensionRegistry` to both `evaluate_request_file` and
`verify_evidence`. The registry may contain caller-injected authorized scorers
or may explicitly enable installed `invarlock.scorers` entry-point discovery.
Neither an installed package nor a binding inside the submitted evidence
authorizes scorer code by itself. Strict verification resolves the policy-
pinned scorer through this registry and replays it deterministically.

## Request loading

```python
from invarlock.engine import load_evaluation_request

request = load_evaluation_request(Path("request.yaml"))
```

The loader returns an immutable `EvaluationRequest` whose paths are resolved
beneath the request root. Its default provider resolver exposes only the
built-in Hugging Face provider. `evaluate_request_file` uses the core registry,
which also discovers installed first-party add-ins.

Loading validates structure and path confinement; the evaluation transaction
repeats no-follow reads at the point of use to detect later filesystem changes.

## Host-coordinated OCI evaluations

The stable facade exports `OciEvaluationLaunch`, `OciSideLaunch`,
`OciRuntimeExecutor`, and `launch_from_environment`. The public CLI constructs
this executor for every run-mode request. The host prepares the schedule,
launches one independently pinned worker per side, validates both closed side
results, and keeps the evidence-signing key outside the workers.

An embedding can build the same launch explicitly:

```python
from invarlock.engine import (
    OciEvaluationLaunch,
    OciRuntimeExecutor,
    OciSideLaunch,
    OciWorkerLimits,
)

baseline_digest = "sha256:" + "1" * 64
subject_digest = "sha256:" + "2" * 64
executor = OciRuntimeExecutor(
    OciEvaluationLaunch(
        engine="docker",
        worker_limits=OciWorkerLimits(
            cpus="8",
            memory_mib=16384,
            user="12001:12001",
        ),
        baseline=OciSideLaunch(
            image_ref=f"registry.example/invarlock-runtime@{baseline_digest}",
            image_digest=baseline_digest,
            device="cpu",
            entrypoint="auto",
        ),
        subject=OciSideLaunch(
            image_ref=f"registry.example/invarlock-runtime@{subject_digest}",
            image_digest=subject_digest,
            device="cpu",
            entrypoint="auto",
        ),
    )
)
```

Each worker receives read-only job, schedule, artifact, and support mounts plus
one isolated writable output mount. `OciWorkerLimits` applies validated CPU,
memory, and numeric non-root user controls independently to each worker; its
defaults are `4`, 65536 MiB, and `65532:65532`. The launcher may run two CPU
workers or two different explicit CUDA indexes concurrently. Generic CUDA
selection, a shared index, and CPU/CUDA pairs run sequentially. CUDA selection
exposes a device but does not make a CPU-only image CUDA-capable; use the x86_64
image built from `runtime/Dockerfile.cuda` for canonical CUDA Hugging Face
execution.

`launch_from_environment` resolves the common and per-side image, digest,
device, entrypoint, and common worker-limit variables documented in [Environment
variables](environment.md). Explicit function arguments take precedence over
environment values.

## Direct executed evaluations and caller resources

`evaluate_request_file` accepts an optional `resource_resolver` implementing
the runtime resource resolver protocol. This is a lower-level programmatic hook
for execution that is already inside an authenticated strict runtime context.
It supplies caller-owned artifact roots, support files, device selection, and
the selected side-image digest. It is not the host OCI launcher. Passing both a
`resource_resolver` and `runtime_executor` is rejected.

`RuntimeResourceResolver` is exported from `invarlock.engine` as a structural
protocol. Its exact method is:

```python
def resolve(
    self,
    *,
    request_root: Path,
    role: Literal["baseline", "subject"],
    side: ComparisonSideRequest,
    provider: RuntimeProvider,
) -> RuntimeArtifactResources: ...
```

`ComparisonSideRequest` is intentionally not a stable construction surface;
the loader supplies it to the resolver. A custom resolver reads only the
already validated side and returns caller-authorized resources. The following
HF-only resolver is complete for a transaction already running inside its
strict side context because HF needs no executable support files:

```python
from pathlib import Path
from typing import Any, Literal

from invarlock.engine import (
    RuntimeArtifactResources,
    RuntimeProvider,
)


class AuthorizedHFResources:
    def __init__(self, *, image_digest: str, device: str = "cpu") -> None:
        self.image_digest = image_digest
        self.device = device

    def resolve(
        self,
        *,
        request_root: Path,
        role: Literal["baseline", "subject"],
        side: Any,
        provider: RuntimeProvider,
    ) -> RuntimeArtifactResources:
        if provider.name != "hf_transformers":
            raise ValueError(f"unsupported authorized provider: {provider.name}")
        artifact = side.artifact.path
        if artifact is None:
            raise ValueError(f"{role} has no run-mode artifact")
        relative = artifact.relative_to(request_root).as_posix()
        return RuntimeArtifactResources(
            root=request_root,
            primary_artifact=relative,
            support_resources={},
            device_kind=self.device,
            container_image_digest=self.image_digest,
        )
```

`ProviderResolver` is the callable type
`Callable[[str], RuntimeProvider]`. It is for controlled request loading; it
must reject unknown provider names rather than fall back to a default.

## Provider ABI types

The facade exports runtime-provider ABI version `1` as
`INVARLOCK_RUNTIME_PROVIDER_ABI`, together with these value and protocol types:

- `RuntimeProvider`, `RuntimeProviderCapabilities`, and `RuntimeSession`;
- `ModelRuntimeSpec`, `ModelArtifactIdentity`, and
  `RuntimeArtifactResources`;
- `RuntimeExecutionContext`;
- `EvaluationBatch` and `EvaluationRecord`;
- `RuntimeProviderReceipt`, `RuntimeScoringRecord`, and
  `ScoringObservation`;
- `RuntimeBehavioralSchedule` and the three artifact-identity variants; and
- plugin, backend, execution-settings, and device provenance values.

The two resolver annotations used by the public function signatures are also
exported: `ProviderResolver` and `RuntimeResourceResolver`.

### Core value constructors

The most commonly embedded ABI values have these stable constructor fields:

| Type | Required constructor fields | Defaults and constraints |
| --- | --- | --- |
| `ModelRuntimeSpec` | `provider_name`, `model_id` | `settings={}`; values must be finite JSON scalars and keys canonical names |
| `RuntimeArtifactResources` | absolute `root`, relative `primary_artifact`, `support_resources`, `device_kind`, `container_image_digest` | Root and every resource are checked without following links; device is `cpu` or `cuda`; digest is required |
| `RuntimeExecutionContext` | `strict`, `allow_network`, `container_image_digest`, `device_kind` | `artifact_identity_sha256`, opaque `provider_state`, scorer, and close callback default to `None` |
| `EvaluationInputPart` | `kind`, canonical `role`, bare `sha256` | Text parts carry `text`; content parts carry `content_id`, `media_type`, and `byte_length` |
| `EvaluationRecord` | `record_id`, first `input_text`, bare `input_sha256` | `expected_output=None`; `input_parts=()` retains the provider-construction compatibility path |
| `EvaluationBatch` | bare `schedule_sha256`, nonempty tuple of `records` | `metric="exact_match"`; `task="text_causal"`; record IDs must be unique |
| `RuntimeProviderCapabilities` | provider, artifacts, tasks, metrics, modes, required extra/image | Format and ABI fields are fixed by the type |
| `ScoringObservation` | provider, artifact digest, schedule digest, nonempty records, aggregate-source digest | Record IDs must be unique; format is fixed |

Mappings are copied into immutable views during construction. Passing a mutable
dictionary does not make the resulting ABI object mutable. Digest fields in the
ABI are generally bare 64-character lowercase SHA-256 values; OCI image fields
use `sha256:<64 hex>`. The constructor validation distinguishes them.

## Hugging Face identity helpers

The stable facade exposes the exact helpers used by the canonical built-in
provider:

```python
from invarlock.engine import (
    checkpoint_tree_sha256,
    hf_tokenizer_contract_sha256,
)
```

`checkpoint_tree_sha256(path)` returns a `sha256:` digest after a no-follow,
change-detecting tree scan. `hf_tokenizer_contract_sha256(tokenizer)` returns a
bare digest of the live tokenizer behavior contract and does not import or load
Transformers itself. The caller must load the local tokenizer with network and
remote code disabled. Importing `invarlock.engine` remains backend-lazy.

Provider implementations may also use the complete torch-free ABI module,
`invarlock.core.runtime_provider`, for canonical schedule construction and
observation-verification helpers that are outside the embedding facade. Backend
imports must remain lazy so provider discovery does not import a heavyweight or
hardware-specific runtime.

## Author complete runtime-import evidence

The stable facade includes a typed authoring path for evaluation records emitted
outside the native provider transaction. It is deliberately narrower than a
generic results importer:

- `load_external_scoring_records_jsonl` accepts one backend observation per
  line and rejects unknown or aggregate-summary fields;
- `build_runtime_import_observation` binds those records to the exact schedule
  and artifact identity;
- `build_runtime_import_receipt` binds provider, backend, execution, device,
  image, artifact, and canonical observation bytes;
- `write_runtime_import_side` atomically writes all six files needed for one
  import side, then reloads them through strict runtime-manifest verification;
- `load_runtime_import_side` independently reloads an existing side; and
- `write_runtime_import_paired_records` derives canonical pairs from two
  verified sides. It never accepts caller-supplied scores.

The high-level signatures are:

```python
load_external_scoring_records_jsonl(
    path: str | Path,
    *,
    schedule: RuntimeBehavioralSchedule,
) -> tuple[RuntimeScoringRecord, ...]

write_runtime_import_side(
    directory: str | Path,
    *,
    role: Literal["baseline", "subject"],
    schedule: RuntimeBehavioralSchedule,
    policy_digest: str,
    artifact_identity: ModelArtifactIdentity,
    records: tuple[RuntimeScoringRecord, ...],
    plugin: RuntimeProviderPluginIdentity,
    backend: RuntimeBackendIdentity,
    capabilities: RuntimeProviderCapabilities,
    execution_settings: RuntimeExecutionSettings,
    device: RuntimeDeviceFacts,
    runtime_image_ref: str,
    runtime_image_digest: str,
    generated_at_utc: str,
) -> RuntimeImportSideEvidence

write_runtime_import_paired_records(
    path: str | Path,
    *,
    schedule: RuntimeBehavioralSchedule,
    metric: str,
    baseline: RuntimeImportSideEvidence,
    subject: RuntimeImportSideEvidence,
    scorer_binding: ScorerExtensionBinding | None = None,
    scorer_registry: ScorerExtensionRegistry | None = None,
) -> RuntimeImportPairedRecords
```

The timestamp is explicit so identical inputs produce identical files. The
writer records strict container execution, offline operation, disabled remote
code, and disabled third-party discovery. Those fields describe facts the
caller is responsible for measuring; serialization does not attest that an
untrusted host reported them honestly. Independent verification must still pin
the artifact, schedule, policy, runtime-image, and signer identities.

`RuntimeImportAuthoringError` is raised before publication when record pairing,
output digests, provider identities, provenance, side files, or manifest replay
do not agree. An import request also needs the matching authorized provider
package so the transaction can reproduce the artifact identity from request
settings.

See [Runtime providers](runtime-providers.md) for lifecycle and conformance
obligations.

## Verify a signed receipt

`load_trust_inputs` loads a closed `invarlock/trust-inputs-v1` profile for
applications that want the same verifier-owned input grouping as the CLI. It
returns `TrustInputs`, including the policy and verifier-key bytes captured
through descriptor-relative reads, their display paths, normalized anchors,
scorer authorization, and the formatting-independent profile digest. Unsafe
paths, symlinks, duplicate JSON members, unknown fields, and missing files
raise `TrustInputsError` before verification.

Pass the captured `policy_bytes` and `verifier_signing_key_bytes` to
`verify_evidence`; do not reopen the display paths after loading the profile.
This keeps verification and receipt signing bound to the exact bytes that were
accepted during the profile read:

```python
trust = load_trust_inputs(Path("trust/trust-inputs.json"))
verification = verify_evidence(
    evidence_path,
    policy_path=trust.policy_path,
    policy_bytes=trust.policy_bytes,
    expected_baseline_artifact=trust.expected_artifact_digests["baseline"],
    expected_subject_artifact=trust.expected_artifact_digests["subject"],
    expected_schedule=trust.expected_schedule_digest,
    expected_baseline_runtime=trust.expected_runtime_digests["baseline"],
    expected_subject_runtime=trust.expected_runtime_digests["subject"],
    expected_signer=trust.expected_signer_fingerprint,
    receipt_path=Path("verification.receipt.json"),
    verifier_signing_key_path=trust.verifier_signing_key_path,
    verifier_signing_key_bytes=trust.verifier_signing_key_bytes,
    verifier_identity=trust.verifier_identity,
    trust_profile_digest=trust.profile_digest,
    scorer_registry=(
        ScorerExtensionRegistry(allow_installed=True)
        if trust.allow_installed_scorers
        else None
    ),
)
```

`verify_signed_verification_receipt` is the stable downstream receipt reader.
It accepts the receipt and pack paths plus independently supplied policy,
baseline and subject artifact-identity digests, canonical schedule digest,
baseline and subject runtime digests, expected evidence signer, expected
verifier identity, and expected verifier fingerprint. It verifies the receipt
signature and embedded public-key fingerprint, then compares every statement
binding with those caller-owned values and the supplied pack manifest.

The function returns `ReceiptVerification`; check `ok` before using `statement`.
`signed` reports whether a signed envelope was present, while `errors` contains
closed failure details. `require_signed` defaults to true and should remain true
for evidence-pack v1 acceptance.

```python
verify_signed_verification_receipt(
    receipt_path: Path,
    pack_dir: Path,
    *,
    policy_path: Path,
    expected_artifact_digests: dict[str, str],
    expected_schedule_digest: str,
    expected_runtime_digests: dict[str, str],
    expected_pack_signer_fingerprint: str,
    expected_verifier_identity: str,
    expected_verifier_fingerprint: str,
    expected_trust_profile_digest: str | None = None,
    require_signed: bool = True,
) -> ReceiptVerification
```

There is no separate receipt-verification CLI command. Applications and
automation use this Python facade when consuming a receipt issued by another
verifier.

## Evidence pack result types

`EvidencePackResult`, `EvidencePackStatus`, and `EvidenceObservation` are
exported for applications that inspect verifier results or attach typed
observation-only JSON during evidence publication. `EvidenceObservation`
contains an ID, `baseline`/`subject`/`comparison` scope, kind, and canonical JSON
payload bytes. Publication adds the comparison, schedule, policy, and artifact
bindings. The high-level `verify_evidence` transaction is preferred because
evidence-pack v1 requires a separately signed receipt.

The package contains additional lower-level publication and format readers for
its own transactions and tests. They are not re-exported by the stable engine
facade. Do not depend on their import paths when the evaluate, verify, report,
and signed-receipt functions cover the required workflow.

`EvidencePackStatus` exposes these lower-level values:

| Member | Value | Category |
| --- | --- | --- |
| `OK` | `0` | Complete success |
| `USAGE` | `2` | Missing or invalid caller input |
| `MISSING` | `3` | Required pack material absent |
| `FORMAT` | `4` | Contract, canonical-byte, or structural format rejection |
| `SIGNATURE` | `5` | Signature-specific rejection |
| `INTEGRITY` | `6` | Digest, inventory, or semantic binding rejection |
| `REPORTS` | `7` | Report replay or policy-result rejection |
| `INTEGRITY_ONLY` | `8` | Internal category for an integrity-only result |

The high-level transaction may normalize precondition failures and does not
promise that every enum member is emitted by every public command. Accept only
`OK`, never a selected nonzero category.

## Exception handling

| Exception | When raised | Machine-readable behavior |
| --- | --- | --- |
| `EvaluationRequestError` | Request parsing, schema, path, provider, or capability failure | Message only at request-loader boundary |
| `EvaluationPreflightError` | Execution-free request, input, runtime-availability, key, or output qualification fails | `as_json()` uses `invarlock/evaluation-preflight-v2` |
| `EvaluationTransactionError` | Execute/import/publication cannot produce evidence | `exit_code`; `as_json()` uses `invarlock/evaluation-result-v1` |
| `EvidenceVerificationError` | Trust input, pack replay, receipt write, or acceptance failure | `exit_code`, `payload`, and `as_json()` |
| `EvidenceReportError` | Signature-authenticated report cannot be rendered safely | `exit_code` |
| `EvidenceReceiptError` | Receipt parsing or processing cannot complete safely | Exception; ordinary authenticity mismatches remain in `ReceiptVerification.errors` |
| `TrustInputsError` | Trust profile parsing, schema, path, or referenced-file validation fails | Message only; no verification is attempted |

Do not catch `ValueError` broadly and continue with partial output. Catch the
documented transaction exception, log its closed message or payload, and reject
the transaction. A rejected verification can still have a signed receipt when
replay completed; inspect the exception payload for `signed_receipt`.

## Related documentation

- [Command-line interface](cli.md) maps the same transactions to shell commands
  and exit behavior.
- [Public contracts](contracts.md) defines the versioned JSON objects consumed
  and produced by the facade.
- [Runtime providers](runtime-providers.md) specifies the provider protocol and
  conformance obligations.
- [Reports and receipts](reports.md) explains the trust meaning of result and
  receipt objects.
