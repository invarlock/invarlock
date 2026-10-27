# Rust optimization results

> **Reference**
>
> **Surface:** Experimental native memory changes and Linux comparison
>
> **Stability:** Experimental; Python remains the default backend
>
> **Use this page when:** Assessing resource use before selecting the Rust backend

Linux is the primary deployment and performance target. The core remains
portable; macOS supplies development and regression evidence. The original
[expanded macOS comparison](rust-expanded-results.md) is retained separately.
A Linux ARM64 virtual machine does not establish production x86 performance.

## Implemented changes

Schema compilation now shares referenced validators. A documented third-party
repair preserves reference traversal in custom-keyword diagnostics: without it,
shared validators selected the wrong first public error. All 54 targeted
reference cases and the complete contracts differential suite pass after repair.

The comparison seed streams borrowed fields into its canonical hash, preserving
exact bytes and numerical methods. Report assembly moves loaded evidence into
its result instead of copying both runs. Snapshot and source-identity checks
remain in place.

Publication retains each synchronization boundary and now uses the Python
reference's `fsync` contract. On macOS this replaces the stronger `F_FULLFSYNC`
operation previously used by Rust. Linux already used `fsync`; this change is
not a Linux speedup claim.

Linux qualification also exposed a lease-lifetime defect: closing the owner file
could leave its lock held by a descriptor inherited during child-process launch.
All four execution lease types now explicitly release owner locks and reject
inherited mutation authority. Deterministic retained-descriptor tests and actual
Linux executable-descriptor launch cover these boundaries.

## Evidence boundaries

The macOS allocator probe measures requested live allocations, not peak RSS.
The first empty-input schema validation used about 6 MiB RSS after repair,
compared with 128 MiB before; eight schemas used about 8 MiB versus 212 MiB.
These diagnostic observations are not complete-workflow performance results.

The Linux comparison is being qualified with the fixed Python reference and
native builds of both Rust versions. Its elapsed time and peak-RSS results will
be retained separately before an adoption conclusion is drawn. Live provider,
GPU, production x86 and cold-cache performance remain unqualified.
