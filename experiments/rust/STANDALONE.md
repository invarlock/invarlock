# Standalone offline workflows

The experimental `invarlock-native-verify` executable now accepts public offline
workflow arguments as well as its existing JSON transaction interface. It runs
without a Python interpreter, model SDK, or companion extension. Python remains
the default public package; this is not a completed replacement of its entire
command surface.

## Implemented paths

- `evaluate REQUEST --signing-key KEY --json`: captured inputs through all eight
  adapter entry points, nineteen evaluator profiles, retained runtime import and
  retained judge import, signed
  publication, optional `--preflight`, `--unsigned`, `--fail-on-policy`, input/output
  overrides, and the bootstrap budget. JSON and bounded strict YAML requests use
  the reference scalar semantics; aliases, anchors, tags and directives are refused.
- `verify EVIDENCE --trust-profile TRUST --receipt RECEIPT --json`: captured,
  native, judge and evidence-set recipient verification. Direct native/captured
  anchor options are also accepted. Judge receipt signing requires explicit
  verifier key and identity arguments.
- `report EVIDENCE --html REPORT --markdown SUMMARY --junit RESULTS --json`:
  all four evidence families; optional explanations and judge case selection.
- `report EVIDENCE --review-context CONTEXT --package DIRECTORY --json`:
  standalone client handoff packaging, with optional `--retained-receipt`.
  Historical receipts remain unverified attachments; packaging never substitutes
  for independent recipient verification.
- `evaluate --init DIRECTORY --example NAME --json`: the four existing starters.
- `evaluate --keygen DIRECTORY --json`: OS-generated Ed25519 keys in the
  Python-compatible PKCS#8 form, with private files restricted to their owner.
- `evaluate --freeze-cases FILE --case-set-output OUTPUT --json`: validated,
  sorted case sets and their existing digest format.

Use `--json` for the machine contract. Native plain terminal output, help and
version information are also available. Signing keys, verifier identity, policy
and trust-pin environment defaults follow the Python CLI; explicit options win.
Reports print Markdown instead of reproducing Rich terminal layout. Exhaustive
diagnostic wording and every CLI/environment combination are not claimed.
Existing low-level JSON transactions remain available through
`COMMAND --request FILE`.

## Package and memory boundaries

Evidence copies and ZIP construction stream through pinned regular files rather
than collecting a whole evidence pack in memory. Copies check source identities
before and after reading. Staging retains owned descriptors, detects mutation,
refuses replacement of an existing destination, and cleans up owned files on
failure. Existing ordinary-publication limits remain unchanged; handoff staging
has its own larger limits for the copied evidence and archive.

Reports, copied evidence, authored context and uncompressed archive members are
compared against Python. Package generator source hashes identify different
implementations and must differ. ZIP compression and timestamps are not byte
identities across implementations. Those differences are declared in the
comparison protocol, not hidden by dropping a failed result.

The embedded recipient guide retains the reference Python installation guidance
for output compatibility. Experimental source-bound Linux bundles now ship for
ARM64 and x86_64.
Changing embedded recipient guidance remains separate release work because that
would change the currently compared handoff bytes.

## Qualification and remaining migration

`native/tests/test_standalone_workflows.py` exercises the standalone binary with a
PATH that cannot discover Python. It covers reports/packages across all four
families, signed and unsigned captured evaluation, receipts, setup, key
interoperability, request scalar rules and refused unsafe inputs. Filesystem
streaming tests independently exercise mutation detection and cleanup.

The remaining Python boundary is compatibility execution: model frameworks,
provider SDK calls, live judge collection, installed custom scorers/providers and
existing Python SDK callers. `invarlock-native --python-worker /absolute/python`
selects this path explicitly. The worker requires the matching native extension,
uses bounded request/output capture, and has a deadline and process-group
cancellation. It does not turn live SDK behavior into a qualified offline path.

Native import tests cover accepted and rejected evaluator exports; runtime
provider identity fixtures cover four built-in providers. Judge-import fixtures
compare signed and unsigned publication bytes and missing-input preflight.
Linux ARM64 and x86_64 tests exercise the process and descriptor boundaries.
Live provider/GPU execution, real OCI engines, Windows and older Linux C library
versions require separate qualification. Existing native report/parser work
limits remain; fixture parity is not an all-input equivalence claim.

See the [installation guide](../../docs/reference/rust-native-installation.md)
for the two native executables, bundle checksums and explicit worker setup.

## Performance protocol

`linux/standalone_workflows.py` measures 16 public-command workloads on the retained
Linux ARM64 and remote Linux x86_64 environments: captured evaluation at four sizes, then verification,
reporting and packaging for each evidence family. One qualification per candidate
and workload precedes five balanced rounds (seed 4127). Every attempt is retained;
a failure or mismatch stops the run, with no automatic retry or exclusion.

The fresh-exec supervisor measures elapsed time, user plus system CPU, and raw
process peak RSS. Identity checks happen before timing for both implementations.
This boundary differs from the older comparison that hashed installed wheels
inside Python/FFI workers; results must not be presented as an improvement relative
to those older ratios. No live provider or inference work is timed.

The completed checkpoint results and retained limitations are documented in
[standalone results](../../docs/reference/rust-standalone-results.md).

The expanded `linux/standalone_expanded.py` matrix adds one-, four- and twelve-row
captured inputs, small/large JSONL imports, DeepEval imports, retained runtime
import and judge import. It compares Python, the experimental Python/Rust facade
and standalone Rust, except handoff packages which have no FFI route. Each Linux
target runs 68 qualification executions and 340 timed executions. A separate
`linux/ffi_boundary.py` diagnostic measures immutable and mutable JSON buffers;
it does not substitute for end-to-end workflow results.

Completed expanded results, qualification limits and retained artifact identities
are in [the final comparison](../../docs/reference/rust-standalone-expanded-results.md).
