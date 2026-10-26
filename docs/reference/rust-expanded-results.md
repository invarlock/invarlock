# Expanded Rust comparison results

> **Reference**
>
> **Surface:** Experimental Rust implementation and retained local comparison
>
> **Stability:** Experimental; Python remains the default backend
>
> **Use this page when:** Deciding whether the expanded Rust architecture warrants adoption

The expanded architecture is implemented as an experimental candidate. It is
faster for the measured captured NLL workloads, slower for the four smaller
workloads, and uses more peak memory in every measured workload. These results
support further work on the candidate, not a default-backend switch.

At 50,000 pairs, standalone Rust reduced median fresh-process wall time from
158.535 to 28.762 seconds, about 5.5 times faster. Peak RSS increased from
588.7 to 1,662.7 MiB, about 2.8 times as much. Through the Python binding, Rust
took 29.940 seconds and 1,684.4 MiB. At 5,000 pairs the standalone result was
4.053 versus 16.250 seconds, with 489.7 versus 104.8 MiB peak RSS.
The native, judge and evidence-set fixtures were slower under both Rust modes.
No allocation profile establishes the cause of the memory increase.

## Implemented scope

The candidate has eight crates: contracts, analysis, filesystem, evidence,
execution, bindings, offline CLI and control CLI. Rust owns bounded JSON and
canonicalization, deterministic and judge statistics, snapshot reads and
publication, captured/native/judge replay, trust and signed receipts, evidence
sets, attestations, reporting and GGUF inspection. The separate execution layer
owns selected process I/O, cancellation, admission, budgets and checkpoint/resume.
The offline verifier has no dependency on that execution layer.

The Python SDK and CLI select Rust explicitly with `--backend rust`; evaluation
also accepts `--execution-backend rust`. A failed or missing native backend
raises an error. Python still supplies ML and vendor SDKs, provider discovery,
YAML and external format decoding, resource negotiation and authorized callbacks.
Some host preflight validation remains repeated around provider hooks, and
low-level Python helpers remain Python unless a supported native transaction is
selected. This is the broadest coherent implemented ownership boundary in the
experiment, not a claim that every possible internal optimization is complete.
See the [design](rust-core-design.md) and the repository's
`experiments/rust/implementation-coverage.json` for actual interfaces and limits.

## Comparison design and observed results

The fixed baseline and freshly fetched `staging/next` both resolve to
`ce04f20ed7ccefc1bc826098f1b0aa4102b63ef3` on 2026-10-26. Its Git tree is
`95553894421fc6d3de83f50360457fff763ccda2`. The measured candidate's source manifest,
wheels, executables, Cargo lock, fixtures and harness are individually hashed in
`experiments/rust/results/expanded-macos-arm64-py312.json`. The 510 recorded source
files and nine frozen harness files were unchanged at measurement completion.
Final hygiene then changed only test parentheses and a manifest blank line;
Python AST and parsed TOML equality were checked, and measured binaries remain
unchanged. Those deltas and the refreshed source archive are recorded separately.

The host was an Apple M2 Max with 12 CPUs and 64 GiB RAM, macOS 27 arm64,
CPython 3.12.13, Rust 1.94.0 and Maturin 1.12.0. Artifacts and outputs shared an
APFS image on external TB4 storage. Candidates ran sequentially in fresh
processes, with one computational worker and no concurrent builds or tests.
Qualification warmed the filesystem; these are not cold-cache measurements.
Thermal state was not established.

Each journey includes independent evidence verification, receipt production and
authentication, and HTML, Markdown and JUnit reporting. The four candidates are
the installed Python reference, that reference with only the recorded interval
hash-prefix optimization, whole Rust transactions through the companion binding,
and a standalone driver using the same evidence crate as the native verifier.
The Python optimization is not an exhaustive optimization of the reference.

Python measurement workers import the fixed baseline wheel and invoke Rust
transactions directly for the Rust candidate. The candidate's public SDK and CLI
were qualified separately through an installed candidate wheel. These timings
therefore measure the transaction boundary, not every public SDK wrapper.
The standalone driver executes without Python; its result does not measure
customer installation experience.

There were **32 matching qualification attempts and 128 matching timed attempts**:
eight workloads, four candidates and four balanced rounds per workload, seed
4127. All returned outputs and signed receipt/report file hashes matched after
normalizing only declared report destination paths. There were no crashes,
timeouts, retries or exclusions. A regression decision is a valid outcome when
it matches the reference; matching does not mean accepting every evidence pack.

The following table reports fresh-process wall seconds as **median [minimum–maximum]**.

| Workload | Python | Python narrow optimization | Rust through Python | Rust standalone |
| --- | ---: | ---: | ---: | ---: |
| NLL 1,000 pairs | 3.456 [3.445–3.473] | 3.287 [3.240–3.345] | 1.599 [1.393–2.082] | 1.389 [1.279–1.426] |
| NLL 5,000 pairs | 16.250 [15.626–23.492] | 15.184 [14.923–15.445] | 4.001 [3.849–5.823] | 4.053 [3.828–4.527] |
| NLL 12,000 pairs | 37.331 [37.163–38.233] | 47.225 [35.507–60.583] | 7.587 [7.373–12.136] | 11.593 [7.373–24.561] |
| NLL 50,000 pairs | 158.535 [156.719–159.143] | 149.437 [147.137–150.998] | 29.940 [29.262–30.772] | 28.762 [28.142–29.917] |
| Captured regression, 64 pairs | 0.480 [0.461–0.485] | 0.475 [0.465–1.296] | 1.053 [0.923–1.698] | 0.858 [0.787–1.104] |
| Small native pack | 0.346 [0.339–0.353] | 0.347 [0.345–0.354] | 0.995 [0.740–1.725] | 1.331 [0.626–1.463] |
| Small judge pack | 0.329 [0.308–0.350] | 0.345 [0.315–0.349] | 1.238 [0.914–1.416] | 1.397 [0.737–1.601] |
| Small evidence set | 0.434 [0.420–0.637] | 0.425 [0.414–0.435] | 1.000 [0.977–1.099] | 0.995 [0.916–1.118] |

The small native pack has two records; the judge and composite evidence-set
fixtures are also small. Their results do not establish large native/judge
capacity or live collection throughput. The captured NLL values vary; these
workloads exercise resampling rather than a constant-difference shortcut.

Peak RSS medians are **MiB for the entire process**, including startup and imports.
They are not incremental extension allocations.

| Workload | Python | Python narrow optimization | Rust through Python | Rust standalone |
| --- | ---: | ---: | ---: | ---: |
| NLL 1,000 pairs | 79.1 | 78.7 | 433.8 | 390.1 |
| NLL 5,000 pairs | 104.8 | 114.2 | 516.6 | 489.7 |
| NLL 12,000 pairs | 170.0 | 169.4 | 705.6 | 580.4 |
| NLL 50,000 pairs | 588.7 | 590.7 | 1684.4 | 1662.7 |
| Captured regression, 64 pairs | 67.8 | 67.8 | 413.9 | 370.1 |
| Small native pack | 67.1 | 67.2 | 209.8 | 165.7 |
| Small judge pack | 68.2 | 67.9 | 290.6 | 247.1 |
| Small evidence set | 68.4 | 68.4 | 525.6 | 482.3 |

Every cell has only four timings. The retained 12,000-pair optimized-Python range
and several Rust wall-time ranges are wide. No samples were discarded, and no
percentile or population-level confidence claim is made. The JSON result retains
each timing, operation time, user/system CPU, peak RSS and sample digest, plus
per-cell medians and ranges. CPU and wall time differ materially in several cells;
no profile attributes the waiting time to a particular cause.

## Budget stop and continuation

The original 3,600-second budget reserves four full 300-second process timeouts
before starting a round. It stopped at that reserve after all 32 qualifications
and the first 64 timings, even though the remaining small workloads fit the
observed-duration estimate. The original stopped summary remains unchanged.

A budget-only continuation rule was recorded before the stop. It reused the 32
matching qualifications and executed only the 64 unmeasured attempts, in the
original order, with a separate 1,800-second budget and the same process timeout.
It rechecked artifacts, inputs, output hashes, interpreter/runtime identity and
frozen harness identity, and rejected partial or orphaned attempts. Its 32 guard
tests passed before continuation. The combined denominator is 128, split 64/64;
no completed measurement or qualification was repeated. The two execution chunks
are identified separately in the JSON result.

## Functional and packaging evidence

- Source checks: 8 compatibility, 6,191 core, 2,914 runtime and 3,230 example tests
  passed: 12,343 total, with 464 skips across those suites. Lint, formatting,
  typing, contract, public-evidence and CLI checks also passed. The aggregate
  `make verify` command retained exit 2 for a documentation link warning. After
  that documentation-only correction, separate `make docs-check` and
  `make examples-check` passed; this is not a claim that the aggregate exited 0.
- Native differential checks: 387 tests and 16,448 subtests passed against the
  final installed extension and tools. The one skip is a non-UTF-8 filename
  that this host's filesystem rejects before either implementation can read it.
  Exercised behavior includes exact numeric/canonical bytes, historical evidence,
  trust and receipts, report bytes, parser/resource boundaries, filesystem
  substitution, interrupted execution, public backend routing, worker publication
  and GGUF identity. These fixtures do not prove all-input equivalence.
- Rust checks: 57 tests passed in debug and the same 57 in release; workspace
  formatting and all-target Clippy checks passed. The binding crate's extension
  interface was exercised through Python instead of linking its Rust test binary.
- Installed delivery: an isolated candidate root wheel and companion extension
  passed public SDK/CLI smoke checks, including tuple payload compatibility,
  reporting and GGUF routing. The source archive contains 424 required files
  across all eight crates. The previous archive built outside the checkout using
  cached locked dependencies. Later archives contain formatting-only test
  changes and a manifest blank-line correction. AST/parsed-manifest checks and
  unchanged binary identities support explicit reuse of that build evidence.
- Dependency evidence: the exact Cargo lock was scanned against 1,378 RustSec
  advisories with zero reported vulnerabilities or warnings. The retained database
  revision is `a78282a1414d6c28cc228c4eae3570451affb0dc`. The 139-crate external
  dependency inventory, SBOM and licenses are retained. This dated result is not
  a general security guarantee.

The offline verifier file is 8,286,576 bytes, about 7.9 MiB, and requires no Python
runtime for its declared operations. The comparison does not establish lower
installation effort or support cost. Earlier failed development checks and the
budget-stopped measurement are retained alongside their repairs and continuation.
The earlier native-test formatting check used Ruff 0.14.6; the final check uses
the repository-pinned 0.16.7. The original interval-only measurements remain historical and are not pooled
with this expanded candidate.

## Adoption decision and next work

Keep Python as the default and retain this Rust candidate for further evaluation.
The NLL speedups justify investigating the broad core, while the small-workload
latency and universal RSS regressions prevent a general claim that migration is
better. Allocation and waiting-time profiling should identify the sources before
changing ownership boundaries or setting a new default. Any repair needs fresh
measurements with new artifact identities; it must not replace these outcomes.

Live OCI, GPU and provider execution have not been qualified. Linux descriptor
execution remains untested, and current Unix-specific code needs a Windows port.
The companion is qualified only for CPython 3.12/macOS arm64. Release integration,
additional targets and actual installed-user experience remain separate work.
Native report reconstruction has a 10,000,000-pair local limit; parser and unusual
Unicode/diagnostic boundaries are documented in `native/README.md`. These are
explicit compatibility limits, not evidence of universal replacement readiness.
