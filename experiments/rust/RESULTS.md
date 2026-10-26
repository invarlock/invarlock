# First Rust interval comparison

The first bounded experiment supports continued work on the Rust evidence core.
It demonstrates a faster native interval and a smaller improvement in complete
signed recipient replay. It does not establish a case for migrating every module.

## Scope and reference

Measured on 2026-10-26 against staging commit
`ce04f20ed7ccefc1bc826098f1b0aa4102b63ef3`, refreshed again after measurement with
no change. The source tree is `95553894421fc6d3de83f50360457fff763ccda2`.
The machine was an Apple M2 Max running macOS on ARM64, CPython 3.12.13 and Rust
1.94.0. Fixtures and both checkouts used the same APFS image on TB4. Native builds
used the release profile, thin LTO and one code-generation unit.

The unchanged Python wheel was installed alongside an explicitly selected native
companion. The Rust operation performs the complete 2,048-replicate SHA-256 paired
interval in one binding call. Python still owns parsing, validation, signatures,
policy handling, publication and the public API. The optimized Python control
only reuses invariant hash-prefix/counter construction; it preserves arithmetic.

These fixtures contain authored varying NLL facts, not new model outputs. They
support implementation and replay claims, not model-quality or provider claims.

## Observed times

Five paired rounds per cell, with a recorded seeded rotating candidate order.
Values below are medians in seconds. Every attempted timing sample is retained.

| Scope | Pairs | Original Python | Optimized Python | Rust |
| --- | --- | --- | --- | --- |
| Full interval operation | 1,000 | 1.112 | 0.925 | 0.373 |
| Full interval operation | 5,000 | 5.483 | 4.505 | 1.860 |
| Fresh signed recipient process | 1,000 | 2.465 | 2.329 | 1.783 |
| Fresh signed recipient process | 5,000 | 11.440 | 10.777 | 8.058 |

At 5,000 pairs, the Rust interval used about 66% less time than original Python
and 59% less than the optimized control. The fresh recipient process used about
30% less time than original Python and 25% less than the optimized control.
The remaining Python work limits the complete-path improvement.

The recipient measurement includes startup, installed-artifact identity checks,
parsing, authentication, comparison replay, signed receipt publication and receipt
authentication. It is an instrumented SDK recipient process, not a measurement of
an uninstrumented CLI or a standalone native verifier. Identity checking is also
recorded separately. Kernel operation time excludes fixture reading and startup
but includes binding conversion and native allocation.

The 5,000-pair recipient median peak RSS was 98.34 MiB for original Python,
98.36 MiB for optimized Python and 98.59 MiB for Rust. This is whole-worker
high-water memory, including imports and identity checks. There is no demonstrated
memory reduction. CPU/RSS and raw time ranges are in the machine-readable results;
five rounds do not establish tail latency or universal speedups.

## Correctness and qualification

All 60 measurement samples completed and matched the reference outputs: two
sizes, two stages, three candidates and five rounds. There were no timing retries,
exclusions, wrong outputs or candidate failures. Kernel results matched as exact
binary64 hexadecimal values. Complete canonical report content, receipt statements
and deterministic signed receipt bytes matched across implementations. Each Rust
arithmetic sample recorded an actual native call; there was no Python fallback.

Before timing, the installed Python 3.12 path matched across 18 checks covering
pass and regression replay, untrusted-signer rejection before arithmetic and
zero-baseline rejection. It used independent policy/trust pins, separate signing
and verification keys and authenticated fresh receipts. The fixture generation
path remained Python. The installed experiment makes no model/API calls; a Python audit hook rejects
socket operations, while the Rust core has no network code. This is not a claim
of operating-system network isolation.

The Rust tests cover independent golden intervals and errors. Differential checks
compare all replicates for selected numerical cases, draw mappings/counter
boundaries and 3,000 deterministic wide-exponent summation cases, with explicit
rounding, cancellation, subnormal, overflow and computed non-finite behavior.
The existing likelihood and historical compatibility tests passed with the Rust
interval selected and 21 native calls observed. These exercised cases do not
prove exhaustive binary64 equivalence.

The native wheel installs in a clean matching environment. The source archive
contains the workspace, pinned toolchain, lock and licenses; the independently
extracted source build succeeded. Python 3.14 smoke evidence is retained
separately and is not part of the reported timing matrix. Linux and other targets
have not been qualified for this native component.

## Corrections retained with the evidence

Early qualification attempts exposed harness defects in virtual-environment path
handling, selecting immutable wheel members, and interpreting the regression
verdict. Those attempts remain retained. A later review corrected kernel timing
summaries and failure accounting before the matrix. Final qualification and the
60 timing samples are separate from those earlier attempts.

The metadata originally labeled the repository `uv.lock` as its dependency lock.
The actual installation used the hash-pinned
`requirements/workflows/release-install-py312.txt`. A retained provenance
clarification identifies that exact file and confirms that all 60 workers'
recorded dependency versions match it. No execution configuration changed and
no samples were rerun for this clarification.

The first broad repository verification used a shared environment whose installed
InvarLock metadata was 0.17.1 while the checkout was 0.17.0. That mismatch triggered
provider-discovery failures. New documentation also initially missed reader
contracts and the explicit page inventory. Those documentation issues were fixed;
183 focused provider/documentation checks passed in a clean matching environment.
The default-Python core suite passed. Runtime, CLI, lint, documentation and
example checks then completed after aligning the test-only typing dependency and
documentation-tool resolution. The failed initial commands remain retained;
completion is recorded by component rather than describing those failed commands
as successful. Optional SDK, container, browser and Linux-specific paths skipped
where unavailable. These repository checks exercise the existing Python default;
the Rust-selected checks are the narrower cases described above.

## Decision and next stage

Keep this implementation optional and experimental. The measured interval benefit
is real for the exercised workloads, including against a straightforward Python
optimization. Complete replay improves less, and memory use is effectively
unchanged. The next useful comparison is transaction-owned validated data and
complete captured verification, with a Python validation-reuse control.

The full target architecture and 141-module inventory remain proposals beyond
this interval. Larger 12,000/50,000-pair measurements, other statistical families,
standalone installation, cross-platform wheels, default production dispatch and
release qualification are still outstanding. No source implementation or release
configuration in the root Python package was replaced by this experiment.

## Reproduction and retained records

See [the harness instructions](README.md) and
[the machine-readable results](results/initial-macos-arm64-py312.json).
The latter includes raw per-cell time samples, paired ratios, resource scopes,
artifact hashes and limitations. The architecture and protocol are in
`docs/reference/rust-core-design.md` and `docs/reference/rust-comparison-protocol.md`.

The TB4 evidence directory retains the frozen manifest/protocol, actual installation
lock, every process record/stdout/stderr, signed fixture packs, independent trust
inputs, receipts, source/build identities and qualification repair history.
Generated test keys and machine-specific paths stay in local evidence.
