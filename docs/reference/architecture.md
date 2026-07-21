# Architecture

InvarLock is an authenticated paired-evaluation engine. Its core path has three
transactions:

!!! info "Reference"

    - **Surface:** Core transaction, package, data-flow, and trust boundaries
    - **Stability:** Architectural contract for the paired evaluation engine; implementation internals may change behind documented public surfaces
    - **Use this page when:** Locating responsibilities, selecting an integration boundary, or inspecting where evidence-signing and verifier authority separate

```bash
invarlock evaluate request.yaml
invarlock verify evidence/
invarlock report evidence/
```

![InvarLock paired release-regression architecture](../assets/evaluation-verification-flow.svg)

`evaluate` executes or imports one baseline-versus-subject comparison and
publishes one signed evidence bundle. `verify` replays that bundle against
independent artifact, schedule, policy, runtime, and signer anchors and writes
a separately signed receipt. `report` authenticates the bundle's closed inventory and embedded
evidence signature, then renders the canonical comparison report without changing the
bundle.

## Transaction boundaries

| Transaction | Reads | Writes | Independent trust required | Acceptance authority |
| --- | --- | --- | --- | --- |
| `evaluate` | Closed request, referenced inputs, caller runtime resources or imported sidecars | One immutable evidence directory | Evidence-signing private key; authenticated runtime/material inputs | Creates paired measurements, a finite-schedule policy result, and signed evidence |
| `verify` | Untrusted evidence directory plus independent artifact/schedule/policy/runtime/signer anchors | One signed receipt outside the pack | Expected artifacts, schedule, policy, runtimes, evidence signer, verifier identity, and private key | Yes, for the exact bound comparison |
| `report` | Signature-authenticated evidence directory | Console and optional HTML outside the pack | None beyond embedded evidence signature | Presents the canonical report; the verification receipt carries acceptance authority |

The same pack can be rendered many times and verified by many independent
authorities without changing a byte in the evidence directory.

## Core layers

| Layer | Responsibility |
| --- | --- |
| Closed request | Select exactly two artifacts, one pinned dataset source or canonical schedule, one policy, exactly one built-in metric or scorer-extension binding, one execution mode, and one output directory |
| Runtime integration ABI | Identify artifacts and emit typed receipts and ordered scoring observations |
| Paired transaction | Prepare or authenticate the schedule, cross-bind both sides, derive built-in scores or replay an authorized scorer, replay the paired interval, and qualify optional count/width controls |
| Canonical bundle | Bind normalized intent, identities, provider material, paired records, report, checksums, and evidence signature |
| Independent verifier | Recompute integrity, identities, pairs, scores, report, and policy verdict under caller-owned trust anchors |
| Human renderer | Produce a console view and optional self-contained HTML from the authenticated canonical report |

## Trust boundaries

The evidence-signing key authenticates the bundle bytes and identifies the
signer. It does not make the submitted assertions true. Verification therefore requires inputs that
are not selected by the bundle:

- the exact policy file;
- the expected baseline and subject artifact-identity digests;
- the expected canonical schedule digest;
- expected baseline and subject runtime image digests; and
- the expected evidence-signer fingerprint.

The verifier signs those anchors, the pack-manifest digest, and its verdict
with a separate key. Keeping the verification receipt outside the bundle lets
multiple authorities assess the same immutable evidence using independently
sourced copies of the policy bytes bound into that bundle.

| Claim | Evidence source | Independent verification action |
| --- | --- | --- |
| Which artifacts were compared | Typed artifact identities and request bindings | Re-derive identity and compare material digests |
| Which inputs were scored | Canonical schedule and ordered observation records | Recompute schedule digest, IDs, order, and input digests |
| Which runtime executed each side | Runtime manifests and provider receipts | Compare both image digests to caller-owned expected values |
| What each backend returned | Scoring observations | Validate per-record facts and observation digests |
| What score and threshold apply | Paired records, policy, scorer binding when selected, and canonical report | Re-derive scores, means, comparison, paired interval, threshold, optional count/width qualification, and verdict; require independently authorized scorer code when selected |
| Who signed the pack | Manifest signature | Compare the public-key fingerprint to the caller anchor |
| Who accepted or rejected it | External receipt | Verify receipt signature, identity, fingerprint, anchors, and manifest digest |

Runtime execution has a second boundary. Each strict executed side runs in its
own worker with an independently pinned image, selected device, and entrypoint
profile. The worker has a read-only job, schedule, artifact, and closed support
resources plus one isolated writable output directory. It never receives the
other artifact or either signing key. An observed container boundary,
digest-bearing image identity, offline execution, disabled remote code, and
disabled third-party plugins are required for each side. These bindings describe
the observed execution envelope; an image digest alone is not proof of every
property of the host or accelerator.

## Package boundaries

The `invarlock` distribution contains:

- the evaluate, verify, and report transactions;
- the request and evidence contracts;
- the runtime-provider ABI;
- the Hugging Face Transformers provider; and
- the independent verifier and human renderer.

The GGUF, TensorRT-LLM, and Hugging Face vision-text providers are first-party
optional distributions. They implement the same ABI and register through the
`invarlock.runtime_providers` entry-point group. Numeric diagnostics are a
separate observation-only package and have no acceptance authority.

```text
invarlock
├── request + evidence contracts
├── evaluate / verify / report transactions
├── Hugging Face provider
├── provider ABI
└── canonical verifier + renderer

first-party optional distributions
├── invarlock-runtime-gguf
├── invarlock-runtime-tensorrt-llm
├── invarlock-runtime-hf-vision-text
└── invarlock-diagnostics (observation only)
```

See [Runtime providers](runtime-providers.md) for the extension contract.

## Data flow

1. The request loader resolves all file references beneath the request root,
   without following symbolic links, and authenticates the exact source bytes.
2. In run mode, pinned local JSONL is transformed deterministically into the
   canonical schedule. The host CLI launches one independently digest-bound
   Docker or Podman worker per side. Both workers score the same schedule, and
   the host validates their complete side results. Import mode authenticates a
   supplied canonical schedule and complete runtime sidecars.
3. The engine computes a deterministic comparison ID and either derives one of
   two built-in paired metrics or replays one explicitly authorized
   deterministic text scorer. Exact match replays paired outcome counts, an
   exact McNemar probability, and the versioned Newcombe 95% interval.
   Normalized NLL and
   scorer-extension deltas use the fixed 2,048-replicate schedule-resampling
   interval. The scorer owns only per-record values in `[0, 1]`; the core owns
   means, subject-minus-baseline percentage-point delta, interval, and policy
   arithmetic. Policy reads the conservative bound of the selected interval.
   When the policy includes coupled sample controls, core also checks the
   authenticated record count and observed interval width. All three
   conditions must pass.
4. Publication stages a closed inventory, signs the canonical manifest, and
   renames the directory into place without replacing an existing destination.
5. Verification treats the submitted bundle as untrusted, replays all semantic
   bindings, and signs a receipt outside it.
6. Reporting reads only the signature-authenticated bundle and writes only the
   optional presentation output.

Run mode and import mode differ only before bundle assembly. Run mode asks each
isolated worker to emit the sidecars. Import mode authenticates supplied
sidecars and re-derives their identities and pairs. Both reach the same
canonical pack and independent verifier. CPU workers and workers assigned to
distinct explicit CUDA indexes may run in parallel. Generic CUDA, a shared CUDA
index, and CPU/CUDA pairs run sequentially. In every case, the authenticated
schedule and record IDs establish the pairing invariant.

The exact inventory is documented in [Evidence artifacts](artifacts.md); the
decision and receipt shapes are documented in [Reports and receipts](reports.md).

## Stable and internal surfaces

Embedding applications should use `invarlock.engine`. Provider implementations
use the runtime-provider ABI in `invarlock.core.runtime_provider`. An embedding
that selects a scorer extension supplies an explicitly authorized
`ScorerExtensionRegistry` to evaluation and verification. Other modules
that encode, decode, or verify individual internal files are implementation
details unless a page explicitly calls them public.

The schemas under `contracts/` are public inspection and interchange contracts.
The Python implementation remains authoritative for semantic cross-file replay,
safe filesystem traversal, and transaction behavior that cannot be expressed by
JSON Schema alone.

## Related documentation

- [Evaluation lifecycle](lifecycle.md) follows the transaction boundaries from
  request loading through publication, verification, and rendering.
- [Evidence artifacts](artifacts.md) inventories the canonical bundle and its
  dependency chain.
- [Public contracts](contracts.md) defines the stable interchange formats.
- [Runtime providers](runtime-providers.md) describes the ABI boundary around
  backend-specific execution.
