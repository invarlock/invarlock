# Rust architecture comparison

[FFI and corrected-reference results](../../docs/reference/rust-ffi-results.md)
compare the latest measured native revision with corrected Python 0.17.3 and
separately measure large-buffer memory.

[Expanded results](../../docs/reference/rust-expanded-results.md) cover the broad
Rust implementation. [First measured results](RESULTS.md) remain the historical
interval experiment. Machine-readable expanded timings and evidence identities
are in `results/expanded-macos-arm64-py312.json`.

The expanded candidate implements complete evidence transactions and separate
execution/control infrastructure. See `native/README.md` for building and selecting
it, and `implementation-coverage.json` for the exercised boundaries.

`full_fixtures.py` authors retained workloads through the installed fixed Python
reference. `full_runner.py` uses `matrix-expanded.json` to compare original Python,
the recorded narrow optimization, whole Rust transactions through Python, and a
standalone Rust driver. It qualifies each path before timing and retains every
attempt. The journey includes signed receipts and report publication. Fixture
inputs, command binaries, wheels and harness sources are pinned in metadata.
The metadata additionally records `standalone`, `offline_cli` and `harness_sources`.

## Expanded comparison

Use the retained metadata, fixture manifest and exact matrix hash with the
installed comparison environment. Set each path to the actual retained files;
the output directory must not exist. The fixture manifest includes the report
steps and their output files as well as receipt operations.

```sh
"$PYTHON" -I -c \
  'import sys;sys.path.insert(0,sys.argv.pop(1));from invarlock_rust_experiment.full_runner import main;raise SystemExit(main())' \
  "$EXPERIMENT" --python "$PYTHON" \
  --metadata /absolute/metadata.json --fixtures /absolute/fixtures-with-reports.json \
  --matrix /absolute/matrix-expanded.json --matrix-sha256 "$MATRIX_SHA" \
  --output /absolute/new-expanded-output
```

The retained run has 32 matching qualification attempts and 128 matching timings.
Its original budget reserve stopped after 64 timings. The declared budget-only
continuation used `recovery/continue_budget.py` for the 64 unmeasured attempts;
it preserved every original attempt and did not repeat qualification. That
script accepts only a matching stopped run, the original interpreter and frozen
inputs. It is not a retry mechanism for failed or incomplete attempts.

## Historical interval experiment

The following sections describe the original interval-only experiment and remain
available for reproducing those historical results.

This directory contains an explicit offline experiment. The original Python
package remains the installed baseline. The experiment substitutes both the
contract function and the likelihood module's imported alias inside a reversible
context manager. Missing or failed native code fails the candidate; every arithmetic Rust
sample must record a native invocation. Rejection before arithmetic records zero
invocations.

The candidates are the original Python implementation, Python with only the
schedule prefix and draw-counter encoding hoisted, and the separate native wheel.
The Python optimization preserves the SHA-256 input bytes, `math.fsum`, arithmetic
order and percentile interpolation.

The fixture contains authored, nonconstant synthetic captured NLL values. It makes
no model or API calls and supports no model-quality claim. Fixture publication
uses public `invarlock.engine` APIs, including `make_run`, `write_run`,
`freeze_case_set`, `normalize_captured_request` and `evaluate_request_file`.
It retains the policy, independent trust pins, and signer/verifier keys outside the
signed evidence directory. A fresh recipient process calls `engine.verify_evidence`,
issues a signed receipt and authenticates it with `verify_captured_receipt`.
Generated private keys are local experiment fixtures; do not publish run output.

## Environment and identities

Use an environment under `/Volumes/InvarLock-Rust-Dev/environments` with the
baseline root wheel and native wheel installed non-editably. The comparison
checkout is `/Volumes/InvarLock-Rust-Dev/invarlock-rust`; the unchanged baseline
checkout is `/Volumes/InvarLock-Rust-Dev/invarlock-baseline`. Both live on the same
physical external APFS/TB4 drive. Workers use `-I`, explicitly add only this
experiment's `python` directory, reject editable/source root imports, and block
network access. They do not add the source checkout's `src` or tests to `sys.path`.

Create a retained metadata JSON with these required fields (replace every example
value with the actual identity):

```json
{
  "source_commit": "actual commit",
  "source_tree_sha256": "actual Git tree identity (legacy field name)",
  "dependency_lock": {"path": "/absolute/retained/lock", "sha256": "actual hash"},
  "root_wheel": {"path": "/absolute/retained/root.whl", "sha256": "actual hash"},
  "native_wheel": {"path": "/absolute/retained/native.whl", "sha256": "actual hash"},
  "native_cargo_lock": {"path": "/absolute/retained/Cargo.lock", "sha256": "actual hash"}
}
```

The runner verifies the four file hashes. Every worker additionally compares the
installed root and native package bytes with their retained wheel members. The
source commit/tree identity is supplied by the caller; the runner records its own
harness source hashes and host facts separately. Each worker records the effective
Python, installed package locations, versions and dependency versions. The
metadata's dependency lock is retained evidence, not an assertion that version
numbers alone prove dependency wheel identity.

## Small qualification

Run this seven-record smoke before any matrix. Set the paths to actual retained
metadata and a new output directory. The command refuses an existing directory.

```sh
EXPERIMENT=/Volumes/InvarLock-Rust-Dev/invarlock-rust/experiments/rust/python
PYTHON=/Volumes/InvarLock-Rust-Dev/environments/comparison-py312/bin/python
"$PYTHON" -I -c \
  'import sys;sys.path.insert(0,sys.argv.pop(1));from invarlock_rust_experiment.smoke import main;raise SystemExit(main())' \
  "$EXPERIMENT" --python "$PYTHON" \
  --metadata /absolute/metadata.json --output /absolute/new-smoke-output
```

The smoke covers signed pass and regression decisions, wrong-signer rejection
before arithmetic and a zero-baseline rejection in each candidate.
This qualifies only the exercised fixtures and installation. It does not establish
performance or capacity at 1,000, 5,000, 12,000 or 50,000 records.

## Frozen measurement matrix

Run the authorized bounded matrix only after freezing its definition and retained
identities and passing the installed-path qualification. `matrix-initial.json` declares 1,000 and 5,000 records, five paired
rounds per stage/candidate and seed 1729. `matrix-large.json` declares the optional
12,000 and 50,000 record follow-up with three rounds; it is a later stage and is
not part of the initial dispatch.
The runner accepts only these declared size/round combinations. Compute the hash
of the actual matrix bytes before dispatch.

```sh
MATRIX=/Volumes/InvarLock-Rust-Dev/invarlock-rust/experiments/rust/matrix-initial.json
MATRIX_SHA=$(shasum -a 256 "$MATRIX" | cut -d ' ' -f 1)
"$PYTHON" -I -c \
  'import sys;sys.path.insert(0,sys.argv.pop(1));from invarlock_rust_experiment.runner import main;raise SystemExit(main())' \
  "$EXPERIMENT" --python "$PYTHON" --matrix "$MATRIX" \
  --matrix-sha256 "$MATRIX_SHA" --metadata /absolute/metadata.json \
  --output /absolute/new-measurement-output --execute-frozen-matrix
```

Fixture generation is retained separately. The kernel and fresh recipient verify
stages are reported separately. Each stage has a seeded candidate permutation,
rotated across paired rounds so position counts differ by at most one across
the declared rounds. Fresh-process wall time includes startup, imports and identity checks;
worker operation wall time is also retained. Kernel summaries and paired ratios
use only the kernel call timer, with argument parsing, identity scans and output
serialization outside that timer. Verification operation time covers evidence
verification and receipt authentication; the primary fresh-process verification
measurement includes process startup and the separately recorded identity-check
time. `wait4` supplies whole-worker user/system CPU and peak RSS, including imports,
identity scans and serialization; these are not kernel-only resource figures; Darwin bytes and Linux KiB are normalized correctly.
These repeated warm-filesystem processes do not characterize cold-cache behavior.

Every attempt retains stdout, stderr, exit status, timeout status, wall time,
CPU, RSS and worker identity. `sample.json` records execution with output comparison
pending. `assessment.json` records exact comparison. Reports and entire receipt
statements are compared without field exclusions; this receipt format contains no
timestamp. The shared verifier key also makes receipt bytes directly comparable.
Kernel float outputs are compared through hexadecimal representations. The runner
stops further dispatch on an observed candidate failure, retains failed attempts,
and does not retry or exclude them from the attempt denominator. A failed or
inconsistent observed reference makes its comparison group unevaluable. A stop
caused by another candidate preserves already comparable successful samples. A
reference that has not run is recorded as missing, separately from reference
failure and requested round completeness. Crashes, timeouts,
launch failures, nonzero exits and wrong outputs are separate statuses.

Summaries contain raw valid samples, medians, ranges and paired reference/candidate
wall ratios. Kernel ratios use kernel operation time; verification ratios use
whole fresh-process wall time. Instrumented process totals are retained for both
stages. There is no p95 estimate from three or five rounds, no automatic
performance threshold and no inference from a small conformance result to a large
workload. Repeated seven-record smoke runs must also use new directories.

## Checks

```sh
python -m pytest -q \
  experiments/rust/tests/test_harness.py --override-ini addopts=''
python -m ruff check \
  experiments/rust/python experiments/rust/tests
```

Those tests use the workspace test environment and are source-level conformance
evidence. Only the isolated smoke/measurement workers establish the installed
recipient path. Tests cover exact small optimized/reference outputs, restoration
of both aliases on exceptions, deterministic order balance, real process crash and
timeout retention, launch failures, no-clobber behavior and failure denominators.
