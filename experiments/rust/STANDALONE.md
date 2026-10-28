# Standalone offline workflows

The experimental `invarlock-native-verify` executable now accepts public offline
workflow arguments as well as its existing JSON transaction interface. It runs
without a Python interpreter, model SDK, or companion extension. Python remains
the default public package; this is not a completed replacement of its entire
command surface.

## Implemented paths

- `evaluate REQUEST --signing-key KEY --json`: canonical captured inputs, signed
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

The new command interface emits JSON. The Python terminal presentation and every
CLI/environment combination have not been replicated. Existing low-level JSON
transactions remain available through `COMMAND --request FILE`.

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
for output compatibility. Shipping a supported native distribution and updating
recipient instructions are separate release work.

## Qualification and remaining migration

`native/tests/test_standalone_workflows.py` exercises the standalone binary with a
PATH that cannot discover Python. It covers reports/packages across all four
families, signed and unsigned captured evaluation, receipts, setup, key
interoperability, request scalar rules and refused unsafe inputs. Filesystem
streaming tests independently exercise mutation detection and cleanup.

The target architecture is not complete. These paths still need work:

- Public runtime-import workflow assembly and provider discovery.
- External evaluator-format adapters beyond canonical captured inputs.
- Arbitrary Python scorer/plugin execution through an explicit worker boundary.
- Full public CLI diagnostic, environment-variable and terminal-output parity.
- Release packaging, additional Linux targets, and live execution qualification.

Python model frameworks and Python SDK callers remain compatibility integrations.
There is no silent fallback from the standalone executable to Python. Existing
native report work limits and parser limits still apply; fixture parity is not an
all-input equivalence claim.

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
