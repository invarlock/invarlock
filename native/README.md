# Experimental native core

This workspace implements the portable InvarLock core and a separate execution
layer in Rust. The Python package remains the default backend. Select Rust
explicitly after installing the companion wheel; a missing native package or
failed operation never selects Python automatically.

The experiment compares complete transactions with the fixed Python reference.
It does not replace model frameworks, evaluator SDKs or Python scorer plugins.
Those integrations supply owned data to the native boundary. An explicitly
authorized scorer callback can run in Python; Rust checks its returned facts.

## Workspace

| Crate | Responsibility |
| --- | --- |
| `contracts` | Bounded JSON, canonical bytes, embedded schemas, typed runtime/schedule/request contracts |
| `core` | Deterministic scoring, exact binary64 statistics, comparison reports, judge replay and precision |
| `filesystem` | Pinned descriptors, bounded snapshots, identity checks, atomic no-clobber publication |
| `evidence` | Captured/native/judge packs, recipient trust, signatures, receipts, evidence sets, qualification, attestations, GGUF inspection and reports |
| `execution` | Bounded child processes, OCI workers, admission, budgets, checkpoint/resume and workspace ownership |
| `bindings` | Coarse PyO3 transactions and explicit Python callback adapters |
| `cli` | `invarlock-native-verify`, an offline executable without the execution crate |
| `control-cli` | `invarlock-native`, a separate process/checkpoint command executable |

Differential tests cover the existing digest domains, canonical newlines, numerical
methods, historical report versions and independent recipient trust on the
qualified fixtures and retained corpus. This is not an all-input equivalence claim. It uses
owned inputs, checked records and descriptor-backed snapshots. Long operations
release the Python GIL; an authorized callback reacquires it only for that call.
Publication cannot replace an existing destination. Interrupted admitted work
cannot be silently dispatched again.

## Build

Linux is the primary deployment and performance target. The core remains
portable across the supported Unix paths; macOS is useful for development and
regression checks. Platform-specific measurements are reported separately.

The earlier qualified build profile is Rust 1.94.0 and CPython 3.12 on macOS arm64.
The companion package declares Python `>=3.12,<3.13`; other profiles require their
own qualification. The toolchain and Cargo dependency versions are pinned.

ARM builds enable runtime SHA-256 acceleration through the locked `sha2`
assembly feature and require a working C/assembly compiler. Linux dispatch checks
CPU support and retains the software implementation when SHA instructions are
unavailable. The x86 dependency feature selection is unchanged.

Paired bootstrap replay uses at most eight workers, capped by the operating
system's reported available parallelism. Small workloads stay serial; failed
thread creation falls back to inline computation. Each replicate retains the
reference draw schedule and floating-point summation order, and result/error
selection follows replicate order. This is an execution change, not a change to
the statistical method.

From this directory:

```sh
python -m pip install maturin==1.12.0
cargo build --release --locked --offline --workspace --bins
maturin build --release --locked --offline -i /path/to/python --out /path/to/wheels
python -m pip install /path/to/wheels/invarlock_native_experiment-*.whl
```

Use `CARGO_TARGET_DIR` to keep compiled artifacts outside the checkout. A first
build needs the locked Cargo sources available locally before `--offline` works.
The source archive includes all workspace crates, embedded schemas, dependency
inventory, licenses and the Cargo lock. Differential tests additionally require
the Python reference checkout and its test dependencies.

## Select the backend

The Python command keeps its existing request and result formats:

```sh
invarlock evaluate request.yaml --signing-key signer.pem --backend rust
invarlock verify evidence --trust-profile trust.json --receipt receipt.json --backend rust
invarlock report evidence --html report.html --backend rust
```

`evaluate --execution-backend rust` remains an alias for the same selection.
Provider discovery, strict YAML decoding, external evaluator format adapters,
SDK calls and model execution remain Python integrations. Rust owns the portable
analysis/evidence path and the selected process, admission and checkpoint work.
OCI worker images must contain the native extension when Rust is selected.
The selection also scopes host provider hooks, GGUF inspection and worker-side
publication. Native publication checks the original SDK observation and receipt
before writing their exact bytes; it cannot replace inconsistent provider facts.

The offline binary uses bounded JSON requests:

```sh
invarlock-native-verify --capabilities
invarlock-native-verify verify-trust --request recipient-request.json
```

For captured/native evidence, `recipient-request.json` contains the caller's
`trust_path`, `evidence_path` and `receipt_path`. Judge and evidence-set recipient
policies use the corresponding `verify-judge` and `verify-evidence-set` commands.
`--help` lists every transaction. Request fields are experimental; the stable
external evidence formats remain authoritative.

The control executable has process and checkpoint authority and is deliberately
separate from the offline verifier. It does not embed Python provider SDKs.

## Verification

```sh
cargo fmt -p invarlock-native-core -p invarlock-native-contracts \
  -p invarlock-native-filesystem -p invarlock-native-evidence \
  -p invarlock-native-execution -p invarlock-native-bindings \
  -p invarlock-native-verifier -p invarlock-native-command --check
cargo clippy --locked --offline --workspace --all-targets -- -D warnings
cargo test --locked --offline --workspace --exclude invarlock-native-bindings
cargo test --release --locked --offline --workspace --exclude invarlock-native-bindings
cargo build --locked --offline --workspace --bins --examples
```

The extension boundary is exercised through its installed Python wheel.
Run `native/tests/` with pytest from the repository root, setting `PYTHONPATH=src:.`,
`CARGO_TARGET_DIR`, `INVARLOCK_NATIVE_VERIFY`, `INVARLOCK_NATIVE_COMMAND`,
`INVARLOCK_ANALYSIS_BIN`, `INVARLOCK_ATTESTATION_BIN` and
`INVARLOCK_NATIVE_REPORT_BINDINGS=1` and `INVARLOCK_NATIVE_GGUF_BINDINGS=1`
to the freshly built tools. The installed
companion wheel must match that source. Run the root `make verify` gates too.

The differential suite checks exact canonical output, binary64 results, signed
bytes, public error/status behavior, retained historical artifacts, bounded
resources, interrupted recovery and complete public API paths. Controlled SDK,
provider and container fixtures are distinct from live execution qualification.
Linux descriptor execution and real container/provider operation are not
qualified by macOS fixtures.

The original interval measurements are retained under `experiments/rust/`.
They describe that earlier slice only. Expanded measurements use separate frozen
artifacts, retained workloads, qualification attempts and timing records.

`dependency-inventory.json` and `sbom.cdx.json` describe the resolved dependency
graph. `THIRD_PARTY_NOTICES.md` and `licenses/` retain dependency and adapted
CPython notices. A vulnerability scan is evidence about its dated database and
exact lock, not a general security guarantee.

## Local resource limits and qualification boundaries

Native report replay reconstructs at most 10,000,000 paired binary outcomes.
Exceeding that budget is an incomplete local operation, not a finding that
authenticated evidence is invalid. The Python renderer can attempt larger
authored counts; parity above this native work budget is not claimed. Runtime
schedules retain their separate schema limits. The JSON parser caps nesting at
900 levels and rejects lone UTF-16 surrogates. Python parsing may accept those
surrogates before canonical UTF-8 serialization rejects them, so rejection stage
and diagnostic parity for those inputs are not claimed. The schema byte and
record limits remain independent of these implementation resource limits.

Some host preflight checks are repeated around provider hooks, including runtime
identity and configuration checks. Low-level Python helpers remain available;
selecting a coarse native transaction is the migration boundary. This candidate
does not claim that every target-map Python function has been removed or that
all internal copies and validation passes have been eliminated.

## Memory and publication changes

The optimization candidate deduplicates compiled schema references, streams
canonical comparisons and hashes, and releases captured file bytes as each
payload is decoded. Deferred validation results preserve diagnostic order, and
the source guard retains the final pinned inventory check. Immutable schema
proofs and digests belong to the values they describe. Planned case-set digests
stream sorted borrowed records; tests assert that the validated run schema
implies every projected case-set constraint, with the encoded-size limit retained.

The compact path decodes canonical, schema-valid captured runs one row at a time
into owned fields instead of row and likelihood object maps. Open fields, scores,
metadata and numeric values keep their existing `Value` representation.
Immutable `RunView`, `RecordRef` and `LikelihoodRef` borrows serve verification
and report assembly. Comparison sorts borrowed records by ID; each scope retains
a packed selection mask and count so predicates run once and pair arrays are
shared. Production captured reporting returns the view and required shared-input
projections without reconstructing complete JSON runs.

Existing public `Value` helpers remain available and materialize ordinary JSON
on demand. A failed, unsupported or noncanonical compact parse discards its
partial compact state and reparses the original bytes through the strict generic
parser. Deferred checks retain the established canonical, schema and semantic
error stages. Fallback stays within Rust and can add CPU work for rejected or
noncanonical inputs; its resource behavior requires separate qualification.

The ordered JSON map uses a locally patched `serde_json` representation: its
map header is boxed only when needed, reducing the size of each JSON value while
retaining insertion order. Empty objects allocate no map header. This patch adds
maintenance work: dependency updates must recheck the map APIs, schema diagnostic
order, canonical bytes, and complete output parity. Its upstream source, dual
licenses, exact patch and removal criteria are retained in
[`vendor/serde_json/LOCAL_CHANGES.md`](vendor/serde_json/LOCAL_CHANGES.md).

The schema dependency is a documented, locally patched third-party copy. Its
custom validators must retain reference traversal when reporting errors; dropping
that context changes which public diagnostic appears first. The retained upstream
archive identity, source inventory, exact patch and removal criteria live under
`vendor/`. Vendored dependency code is not InvarLock's own Rust implementation.

Publication uses `fsync` at the same boundaries as Python. On macOS this replaces
Rust's stronger `F_FULLFSYNC`; it is a deliberate alignment with the reference
contract, not preservation of the former native power-loss guarantee. Linux
already used `fsync`, so this change is not a Linux speedup claim.

The portable Linux comparison launcher is documented in
[`experiments/rust/linux/README.md`](../experiments/rust/linux/README.md).
Its environment and retained results determine the scope of each performance
claim; a Docker Desktop ARM64 VM does not establish production x86 performance.
