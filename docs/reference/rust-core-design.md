# Rust core design

> **Reference**
>
> **Surface:** Proposed Rust and Python ownership, interfaces and migration stages
>
> **Stability:** Experimental proposal; released behavior remains authoritative
>
> **Use this page when:** Reviewing migration scope and compatibility obligations

This is the target architecture for an experimental migration. It is not a
description of the released implementation or a commitment to adopt every
component. Adoption depends on the [comparison protocol](rust-comparison-protocol.md).

The initial reference is `staging/next` commit
`ce04f20ed7ccefc1bc826098f1b0aa4102b63ef3`, fetched on 2026-10-26. Its tree is
`95553894421fc6d3de83f50360457fff763ccda2`. The earlier source audit used another
commit with this same tree. Earlier timings remain exploratory measurements;
new comparisons use retained installed artifacts and explicitly identified inputs.

## Implemented experimental candidate

The isolated candidate now contains eight Rust crates covering contracts,
analysis, filesystem snapshots, evidence transactions, execution, Python bindings,
an offline verifier and a separate control executable. Public evaluation,
verification and reporting commands accept an explicit Rust backend. Python
continues to supply model frameworks, provider SDKs, external format adapters,
YAML decoding and resource negotiation.

The candidate includes captured/native/judge publication and replay, independent
recipient trust and receipts, evidence sets, qualification and attestations,
report rendering, deterministic and judge analysis, and separate process,
OCI, admission and checkpoint controls. Retained judge replay covers generic
JSON, Inspect events, OpenAI-compatible records and runtime-provider records.

The latest routes add native pair derivation to judge-import preflight and
capture, native live-worker evidence publication that preserves original provider
observations and receipts, and GGUF identity inspection through the explicitly
selected provider backend. The final rebuilt extension and tools passed the
listed differential fixtures, including these routes; this does not qualify live
provider execution.
Provider hooks still run in Python, and `_validate_import_side` currently repeats
some portable checks there. Low-level Python APIs remain Python unless callers
select a supported native backend or transaction.

The implementation coverage ledger at `experiments/rust/implementation-coverage.json`
records actual files, reachable interfaces, exercised tests, retained artifact
identities and pending checks. It supplements the target module map without
claiming every target module or migration stage is complete. Controlled macOS
fixtures do not qualify live OCI/provider/GPU operation, Linux descriptor
execution or Windows behavior. The first interval results remain historical.
Complete transaction measurements use separately retained artifacts and the
expanded comparison protocol. The [expanded results](rust-expanded-results.md)
show faster captured NLL transactions, slower small transactions and higher peak
RSS in every measured workload. Python remains the default backend.

## Objective and constraints

Give maintainers a defensible adoption decision and give recipients a practical
path to independently verify InvarLock evidence. Preserve existing functionality,
historical interpretation, Python integration, trust boundaries and public error
behavior. Evaluate execution cost, installation, distribution and maintenance.

The user requested an architecture first, isolated implementation, and comparison
against current staging. No numerical speedup target, memory target, customer
adoption requirement or completion deadline was supplied. Standalone distribution
has relevance to independent verification, but its customer value remains an
assumption to evaluate. External projects' speedups and migration schedules are
not estimates for InvarLock.

## Target ownership

The experiment targets the broadest coherent Rust implementation justified by
InvarLock's responsibilities: the complete portable evidence core, a separate
execution/control layer and native command-line workflows. Python owns the ML
integration experience and existing public facade. No percentage-of-code target
applies.
The unit of architectural comparison is a complete transaction, including the
benefits of shared ownership and distribution across components.

Incremental ports are implementation and conformance checkpoints. A small port's
timing result cannot establish the value or ceiling of the complete Rust design.
Do not require each helper to win an isolated benchmark before implementing the
coherent path that depends on it. Preserve the external contracts while allowing
internal data structures, module boundaries and execution to change substantially.

```text
Python SDK / model and provider SDK adapters
                 |                 |
        coarse Python binding      | explicit execution adapter
                 |                 v
           Rust evidence core <--- Rust execution/control <--- native CLI
                 ^
          standalone verifier
```

Use one `native/` Cargo workspace. Start with `invarlock-core` and a private
Python-binding crate. Organize the core into the modules below; separate modules
into crates only when dependency enforcement, reuse or build costs justify it.
Add a standalone verifier when a complete supported verification path exists.
An executable that only checks a signature must not be presented as a semantic
verifier. Keep the execution/control layer in a separate crate so offline
verification has no dependency on process execution or network clients. Native
execution commands call that layer; the standalone verifier depends only on the
evidence core. Python entry points remain compatible adapters for their declared
capabilities during the experiment.

| Core module | Ownership | Allowed dependencies |
| --- | --- | --- |
| `contracts` | Strict JSON, versioned canonical encoders, schemas, limits, normalized requests, records, identities and runtime/scorer/judge types | Encoding, hashing and validation libraries; no Python, network or filesystem access |
| `analysis` | Scoring, pairing, exact match, native resampling, captured resampling, likelihood, judge statistics, policy qualification and precision | Contracts and numerical libraries; no keys, files, providers or subprocesses |
| `filesystem` | Bounded descriptor reads, pinned paths, snapshots, inventories, staging and atomic no-replace publication | Platform system calls and resource types; no analysis |
| `evidence` | Native, captured and judge authentication/replay, independent trust inputs, receipts, acceptance and evidence sets | Contracts, analysis, filesystem and established cryptographic libraries |
| Python binding | Owned input conversion, typed result/error conversion and explicit authorized scorer bridge | Core and Python binding runtime |
| Standalone verifier | Argument handling and explicitly supported offline verification transactions | Same core; no automatic Python discovery, provider SDKs or downloads |
| Execution/control | Portable transaction coordination, bounded process I/O, deadlines, cancellation, budgets, checkpoints and resume | Core, platform APIs and explicit execution adapters; never a dependency of offline verification |
| Native CLI | Portable command routing, preflight and reports; explicit Python-assisted commands where required | Core for offline commands; execution/control for live commands; no implicit provider or extension discovery |

The first experiment implemented native paired resampling inside `analysis`.
That historical feasibility slice preceded the broader candidate described above;
its results do not establish the performance or compatibility of the later core.

## Internal design and optimization

Use Rust to own a whole verification transaction. The following are design
opportunities to implement and measure, not benefits already demonstrated:

- Read bounded, stable input snapshots and construct typed records once. Keep
  validated state bound to the exact snapshot, contract version and independent
  trust/policy inputs. Reuse within that transaction must not authorize a later
  changed file or a different recipient's policy.
- Retain compact records, shared immutable bytes and reusable buffers inside the
  core. Avoid rebuilding Python dictionaries, serializing intermediate JSON or
  crossing the language boundary for each record or validation step. Borrow from
  owned stable buffers where safe; measure necessary input/output copies.
- Model parsing, authenticated facts, qualified analysis and publication as
  explicit states. Types can prevent invalid internal transitions, while public
  input validation, filesystem checks and rejection behavior remain required.
- Combine compatible traversal, hashing and validation work when the exact
  canonicalization and failure semantics permit it. Streaming must respect byte,
  nesting and work limits, and preserve stable snapshots and diagnostic order.
- Reuse numerical workspace and consider bounded parallel work across independent
  records or replicates. Preserve draw mapping, arithmetic order within each
  replicate, final ordering, error selection and exact historical outputs.
  A faster changed statistical method is a separate versioned experiment.
- Provide coarse Python calls and a standalone executable over the same core.
  Measure Python integration and Python-free verification separately, including
  startup, installation and dependency footprint.

Start comparisons with one computational worker. Measure concurrency separately
with an explicit CPU and memory budget; more available cores must not be reported
as a language-only speedup. Ownership does not itself establish zero allocation,
race freedom at filesystem boundaries or a security improvement.

Keep SDK execution, user Python extensions and notebook integration in Python.
Move portable execution coordination into the separate Rust control layer,
including subprocess lifetime, bounded I/O, deadlines, cancellation, output limits,
resource reservations, checkpoints and cleanup. Review executable-backed providers
individually; they do not need the same ownership boundary as Python model SDKs.
The source mechanisms include `_bounded_subprocess.py`, `evaluation_oci.py`,
`runtime_providers/llama_cpp_session.py` and judge `collector.py`/`runner.py`.
Moving these mechanisms requires their own execution qualification and must not
introduce network or process execution into offline verification.

Preserve who can access signing keys, execute extensions, authorize network access
and spend resources. The new native layer cannot inherit isolation from Python
socket monkey patches. Define the actual enforcement boundary before changing
execution. Use deterministic workers to measure local overhead, then qualify each
changed real provider path separately; simulated timings cannot establish live
provider throughput. No paid provider batch is implied by this design.

## Source migration map

The accompanying [module inventory](rust-module-map.md) assigns every Python
module at the reference commit. Mixed modules are split by responsibility;
moving a pure helper does not imply rewriting its provider or public wrapper.

| Current source area | Destination and retained Python responsibility |
| --- | --- |
| `evaluation_record_contracts/`, `public_contracts.py`, `_data/contracts/` | Rust contracts; Python exception adapters; one authoritative schema source with packaged-byte checks |
| `core/runtime_provider/` | Rust portable types, schedules, observations and request bindings; Python public data classes/protocol adapters |
| `core/checkpoint_identity.py`, `dataset_identity.py`, `schedule_preparation.py` | Rust identity, schedule and safe inspection logic; Python resource acquisition |
| `core/scoring.py`, `builtin_scorers.py`, `scorer_extension.py`, `output_text_contract.py` | Rust deterministic scoring and binding validation; Python explicitly authorized extension execution |
| `core/evaluation_request.py` | Rust normalized request semantics; initially Python strict authored-YAML parsing and provider resolution |
| `evaluation_records/cases.py`, `identity.py`, `io.py` | Rust portable identities, digests, validation and persistence |
| Other `evaluation_records/`, `evaluator_capture.py` | Python integration/capture/authoring; Rust validates resulting portable facts |
| `evaluation_comparison/`, `paired_exact_match.py` | Rust work admission, pairing, missing-data handling, analysis and policy verdicts |
| `evidence_pack*.py` | Rust native contracts, snapshots, integrity, replay and publication; Python value/error adapters |
| `captured_contracts.py`, `captured_normalization.py`, `captured_evidence_publication.py`, `captured_verification.py` | Rust captured transaction semantics and receipt v3; Python public wrappers |
| `runtime_manifest.py`, `runtime_provider_evidence.py`, `runtime_verify.py`, `runtime_import_authoring.py` | Rust portable evidence and import validation; Python acquisition/authoring wrappers |
| `judge_measurement_types.py`, `judge_evidence_types.py` | Rust typed contracts and Python value adapters |
| Judge contracts, statistics, analysis, precision, evidence and acceptance | Rust offline semantics, completeness, exact arithmetic, envelopes and receipts |
| Judge native capture/recipe, source import, retained HTTP contracts | Rust portable plan/fact binding; Python source acquisition and workflow wrappers |
| Judge collector and configured runner | Rust admission, budgets, pacing, checkpoints, retry accounting and portable coordination; Python provider SDK calls |
| Judge HTTP/provider integrations | Python SDK/session adapters; Rust portable plan, response and accounting contracts |
| Judge workflow and CLI modules | Rust portable routing, preflight and reports; compatible Python entry points and SDK adapters |
| `evidence_sets/contracts.py`, `verification.py` | Rust shared-input binding and fresh component verification |
| `evidence_receipt.py`, `acceptance_attestation.py`, `trust_inputs.py` | Rust independent pins, signatures, receipts, transport and recipient acceptance |
| `filesystem/` | Rust descriptor and publication primitives; temporary Python adapters |
| `evaluation_transaction.py`, `evaluation_run.py`, `evaluation_runtime.py`, `evaluation_oci.py`, `evaluation_side_worker.py`, `runtime_behavior/` | Rust portable transaction, container and process control; Python model execution and API adapters |
| `runtime_providers/`, registry and optional-profile modules | Python SDK-backed model execution/discovery; Rust portable identity and executable/session control with explicit Python worker adapters |
| `_bounded_subprocess.py` | Rust bounded I/O, deadlines, cancellation and cleanup; Python compatibility adapter |
| `runtime_security.py`, `runtime_security_helpers.py`, `security.py` | Rust native enforcement for migrated execution; Python guards for retained Python execution; qualify both actual boundaries |
| `engine.py`, `evidence_verification.py`, `captured_evaluation.py`, `evaluator_qualification.py` | Stable Python facade dispatching semantic work to Rust |
| CLI and portable report modules | Rust command workflows and deterministic reports; compatible Python adapters; consume authenticated facts without new acceptance authority |
| Diagnostics, sensitivity tooling and Python-specific presentation | Python integration; shared facts and numerical semantics from Rust |

## Data and trust model

Separate owned representations for each verified state:

```text
UntrustedInput -> CheckedSnapshot -> AuthenticatedEvidence -> ReplayedComparison
                                                                  |
independently supplied TrustInputs ----------------------> RecipientDecision
                                                                  |
                                                           SignedReceipt
```

Private constructors enforce successful transitions. A checked snapshot means
safe bounded bytes, not trusted assertions. Authentication binds bytes and
identity, not truthful model execution. A replayed comparison does not by itself
authorize recipient acceptance. Types should retain format/version, snapshot
identity, expected pins and completed checks rather than an ambiguous `valid` flag.

Validate public inputs on every transaction. Reuse immutable validated structures
inside that transaction to avoid repeated parsing and schema work. Do not cache
validation by mutable Python-object identity or reuse trust decisions across
independent recipients. Count all planned work before selective scoring.

An evidence pack cannot choose the recipient's policy, expected artifacts/runs,
schedule/request pins, signer, verifier identity or private key. Workers retain
their existing key/artifact separation. Receipts remain outside immutable packs.
Use established signing libraries; do not implement cryptographic primitives.

## Wire and historical compatibility

Keep the evidence families separate:

| Family | Existing interpretation that must survive |
| --- | --- |
| Native | Pack v1; report v1/v2/v3; receipt v1/v2; native artifact, schedule, runtime and applicable request anchors |
| Captured | Pack v2; complete run/request/policy/signer pins; trust profile v2; receipt v3 scoped to captured comparison |
| Judge | Judge envelope, planned trials, retained measurements, independent recipient policy and judge receipt |
| Acceptance and evidence sets | Current supported receipt scopes, transport bindings and component conjunction; no widening of authority |

The permanent v0.13 corpus retains original semantics and original bytes. Never
relabel historical evidence to simplify a new implementation. Acceptance predicate
v2's embedded raw receipt bytes and their digest must remain exact.

Existing canonicalization follows Python serialization, not a declared generic
canonical JSON standard. Preserve integer/float distinction, float spelling,
negative zero, UTF-8, escaping, key order and profile-specific final newlines.
Judge payload digests and other canonical documents do not all use the same
newline rule. Reject duplicate keys, invalid UTF-8, non-finite input and ambiguous
authored syntax as the current boundary does. Generic Rust serialization or
schema validation is not evidence of byte or diagnostic compatibility.

Native/captured replay compares regenerated canonical report bytes with retained
bytes. A matching verdict or numerically close interval is insufficient.

## Numerical compatibility

| Method | Required behavior |
| --- | --- |
| Native and captured NLL | SHA-256 of schedule seed plus big-endian replicate/draw counters; first eight digest bytes modulo count; 2,048 replicates; existing sum, division and percentile operation order |
| Captured scalar | SHAKE-256 blocks, little-endian uint64 values, rejection sampling and existing mean/percentile arithmetic |
| Exact match | Version-selected paired Newcombe method, exact McNemar arithmetic, count and side-accuracy controls |
| Judge | Exact fractional aggregation, deterministic high-precision Decimal operations, directed outward rounding, `1e-15` quantization and existing family corrections |

Port each algorithm independently. Do not substitute a faster PRNG, pooled-token
NLL, approximate sum, fused arithmetic or another statistically similar interval
under an existing method identifier. Preserve missing-data handling and unit accounting;
more observations within one judge unit do not create more independent units.

The initial interval binding uses one call per complete interval with owned
vectors and bounded working storage. It does not transfer every resampling index
across the Python boundary. Check valid-domain parity and public error behavior
separately; private helpers may assume inputs already checked by their callers.

## Python API and extension boundary

Keep `invarlock.engine` signatures, results and documented exceptions stable.
In particular, preserve verification errors' `exit_code`, `payload`, `details`
and `receipt_path`, and the CLI distinction between rejection, insufficient
evidence, unsupported capability and operational refusal.

Bindings accept owned bytes or typed vectors, perform a coarse operation, and
return typed results. They release Python execution ownership only while no
Python object or callback is used. Unexpected Rust failures must not unwind
across the language boundary or silently fall back to a different verifier.

An authorized scorer bridge explicitly receives its registry and bound
descriptor/configuration. Built-ins cannot be replaced. Preserve deterministic
double replay, immutable descriptors and result validation. Keep callback
lifetimes and error translation visible and test re-entry, failure and cleanup.

A standalone verifier publishes a closed format/scorer capability list. Until
an authorized extension has a qualified native implementation, reject that scope
explicitly. An optional Python-assisted mode would be separately selected and
would have a different dependency/trust description. No automatic import or
discovery may turn an unsupported verification into acceptance.

## Filesystem and network boundaries

Preserve no-follow traversal, regular-file checks, descriptor identities before
and after reading, bounded snapshot allocation, exact inventories, stability
checks, no-clobber publication and cleanup after interruption. Rust ownership
does not prove freedom from filesystem substitution races. Linux/macOS atomic
no-replace paths need separate adversarial qualification.

Rust core verification makes no network calls or subprocess launches. Existing
Python network guards do not automatically constrain new native socket code.
Provider execution, live qualification and containers remain separate paths.
Reuse unaffected retained provider evidence; any later changed provider boundary
requires its own actual qualification.

## Packaging and platform strategy

Preserve the root setuptools project during the experiment. Build an unpublished
companion binding distribution from `native/` with no overlapping ownership of
the `invarlock` namespace. Experiment drivers select it explicitly. Keep native
artifacts outside the root release `dist/` directory.

The current release expects exactly one `py3-none-any` wheel and one sdist, checks
source/archive contents, and binds those artifacts into provenance and publication
records. Adding a compiled extension to that wheel is a coordinated release
change, not a build-backend swap. A companion distribution is the initial
production preference if the experiment supports adoption; version coupling and
supported core protocol must be explicit.

Pin compiler, dependency lock, feature set, binding/build tool and target. Prove
that the binding sdist builds outside the checkout with its workspace core and
assets included. Do not advertise an offline source build without a retained
toolchain and complete vendored dependency set. A standalone archive must identify
OS minimum, shared-library requirements, embedded contract versions and digests.

Metadata currently declares Python 3.12+, macOS and Linux. CI configuration
contains Ubuntu 26.04 with Python 3.12/3.13, not a native wheel matrix. Initial
qualification targets are local macOS ARM64 and the existing Linux x86-64 path;
neither source inspection nor a local build establishes both. Windows, musl,
free-threaded Python and additional architectures need separate decisions.

Keep Python dependency auditing. Add exact Cargo license/advisory inventory,
target/feature-specific Rust SBOMs, native artifact inspection and provenance
before native production release. Current Python environment SBOMs do not
inventory statically linked Rust crates. Preserve all applicable notices.

## Migration stages and exit evidence

| Stage | Deliverable | Required evidence before expansion |
| --- | --- | --- |
| 0 | Architecture, complete ownership map, fixed baseline, protocol and isolated workspace | Source identity, inventory coverage and usable filesystem/build environment |
| 1 | Optional Rust native interval and installed experiment path | Exact vectors/distributions/errors, installed binding and source-build checks, varying signed offline replay and retained three-way measurements |
| 2 | Contracts/canonicalization plus immutable validated transaction structures | Byte and schema/diagnostic parity, malformed/large-input corpus, budget and allocation checks; compare Python validation-reuse control separately |
| 3 | Complete captured deterministic comparison/verification | Signed pass/regression/insufficient/rejection paths, native-to-Python and Python-to-native replay, capacity and receipts; closed standalone capability list |
| 4 | Native pack identities, filesystem transactions, replay and receipt v1/v2 | Historical corpus, scorer bridge, adversarial filesystem behavior and retained native packs |
| 5 | Judge retained evidence/statistics, evidence sets and acceptance | Exact calibration, missing/interrupted trials, component binding and recipient-policy/transport semantics |
| 6 | Rust execution/control and native command workflows | Process and resource accounting parity; bursts, slow consumption, cancellation, interrupted checkpoints, resume and cleanup; same-workload CLI/report comparisons and actual changed-provider qualification |
| 7 | Production integration and distribution | Every promised target installed and exercised, complete release/provenance/security path, default-dispatch/rollback decision |

Stages 2 through 5 build toward the complete core described above. At stage 3,
compare a complete captured verification transaction; at stage 5, compare the
supported native, captured, judge and combined evidence paths through the shared
core, including standalone execution for its declared capabilities. Retain
component timings to explain the result. Stage 6 extends the product comparison
to local execution overhead and native command workflows with explicitly qualified
provider paths. Production adoption is a later decision based on complete paths
and deployment costs, rather than a prerequisite for each implementation stage. Correctness failures, feasibility limits or an
explicit scope change can still require repair or reassessment.

Filesystem/trust primitives must be qualified before any stage that publishes or
accepts through them. The stage table permits prerequisite work to move earlier;
it does not waive a boundary because its final family is scheduled later.

Keep one production semantic implementation per adopted component. During the
experiment Python is the reference. After adoption retain a frozen conformance
reference, not two independently evolving policy engines. Until all supported
scopes are qualified, production behavior and release artifacts remain unchanged.

## Decisions and open risks

The shared-core/Python split, explicit companion binding, unchanged wire versions,
offline verifier scope and three-way comparison are chosen for the experiment.
They are not a conclusion that Rust is better.

The unresolved risks are exact canonical numeric encoding, equivalent schema
diagnostics, Decimal/transcendental arithmetic, scorer callback lifetime,
cross-platform filesystem behavior and native release cost. Each belongs to a
specific stage and must produce decisive evidence there. Stop expansion when a
material mismatch remains; fix its class and rerun affected evidence once.

The initial packaging and small signed qualification are bounded to one working
session, with a 45-minute implementation checkpoint. Initial small/medium timing
uses the finite matrix in the comparison protocol. If a boundary remains
unresolved at its checkpoint, retain the failure and reassess that slice rather
than extending an unbounded rewrite.

## Primary source anchors

Repository sources: `evidence_pack_contract.py`,
`evaluation_comparison/comparison.py`, `captured_verification.py`,
`judge_measurements/statistics.py`, `core/scorer_extension.py`, `filesystem/`,
`engine.py`, `pyproject.toml`, `.github/workflows/release.yml`,
`scripts/release/release_distribution_validation.py` and the
[compatibility covenant](compatibility.md).

Architecture examples informed the boundaries, not the acceptance thresholds:
[Copilot's incremental runtime migration](https://github.blog/ai-and-ml/generative-ai/migrating-the-github-copilot-runtime-to-rust-using-copilot/),
[Polars' Rust core and Python interface](https://pola.rs/), and
[uv's initial standalone distribution](https://astral.sh/blog/uv).
