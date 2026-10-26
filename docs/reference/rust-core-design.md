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

The Rust core owns portable evidence semantics. Python owns the ML integration
experience and existing public facade. No percentage-of-code target applies.

```text
Python SDK / existing CLI / capture and provider integrations
                        |
                explicit Python binding
                        |
                 Rust evidence core <---- standalone verifier
                 /        |        \
           contracts   analysis   filesystem
```

Use one `native/` Cargo workspace. Start with `invarlock-core` and a private
Python-binding crate. Organize the core into the modules below; separate modules
into crates only when dependency enforcement, reuse or build costs justify it.
Add a standalone verifier when a complete supported verification path exists.
An executable that only checks a signature must not be presented as a semantic
verifier.

| Core module | Ownership | Allowed dependencies |
| --- | --- | --- |
| `contracts` | Strict JSON, versioned canonical encoders, schemas, limits, normalized requests, records, identities and runtime/scorer/judge types | Encoding, hashing and validation libraries; no Python, network or filesystem access |
| `analysis` | Scoring, pairing, exact match, native resampling, captured resampling, likelihood, judge statistics, policy qualification and precision | Contracts and numerical libraries; no keys, files, providers or subprocesses |
| `filesystem` | Bounded descriptor reads, pinned paths, snapshots, inventories, staging and atomic no-replace publication | Platform system calls and resource types; no analysis |
| `evidence` | Native, captured and judge authentication/replay, independent trust inputs, receipts, acceptance and evidence sets | Contracts, analysis, filesystem and established cryptographic libraries |
| Python binding | Owned input conversion, typed result/error conversion and explicit authorized scorer bridge | Core and Python binding runtime |
| Standalone verifier | Argument handling and explicitly supported offline verification transactions | Same core; no automatic Python discovery, provider SDKs or downloads |

The first experiment implements native paired resampling inside `analysis`.
This is a feasibility slice within the larger design, not the finished core.

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
| Judge collector, configured runner, HTTP and runtime provider modules | Python provider calls, pacing, budgets, checkpoints and retries |
| Judge workflow and CLI modules | Python routing/orchestration calling Rust offline semantics |
| `evidence_sets/contracts.py`, `verification.py` | Rust shared-input binding and fresh component verification |
| `evidence_receipt.py`, `acceptance_attestation.py`, `trust_inputs.py` | Rust independent pins, signatures, receipts, transport and recipient acceptance |
| `filesystem/` | Rust descriptor and publication primitives; temporary Python adapters |
| `evaluation_transaction.py`, `evaluation_run.py`, `evaluation_runtime.py`, `evaluation_oci.py`, `evaluation_side_worker.py`, `runtime_behavior/` | Python runtime/container/session orchestration; Rust completed-evidence checks |
| `runtime_providers/`, registry and optional-profile modules | Python backend execution and discovery; selected portable identity helpers can move later |
| `runtime_security.py`, `runtime_security_helpers.py`, `security.py`, `_bounded_subprocess.py` | Python execution/network controls and bounded subprocess handling; shared local reads/hashes can call Rust |
| `engine.py`, `evidence_verification.py`, `captured_evaluation.py`, `evaluator_qualification.py` | Stable Python facade dispatching semantic work to Rust |
| CLI, report/explanation/presentation modules, diagnostics and sensitivity tooling | Python; consume authenticated facts without acquiring new acceptance authority |

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
| 6 | Production integration and distribution | Every promised target installed and exercised, complete release/provenance/security path, default-dispatch/rollback decision |

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
