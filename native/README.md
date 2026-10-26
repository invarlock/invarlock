# Isolated paired interval experiment

This workspace contains an ordinary Rust library (`core`) and an explicitly
invoked Python companion (`bindings`). The root Python distribution and release
configuration are independent. There is no automatic backend selection and no
standalone evidence verifier.

The only product operation in scope is the fixed 2048-replicate paired SHA-256
percentile interval in `src/invarlock/evidence_pack_contract.py`. The core uses
owned scalar data, SHA-256 over the decoded 32-byte schedule digest plus big-endian
u64 replicate/draw counters, the first digest u64 modulo pair count, CPython-style
`math.fsum` arithmetic, and the original percentile operation order. Ordinary Rust
binary64 operations are used without fast math or fused replacement expressions.
The single-pair path does not resample.

The Python module exposes:

```python
from invarlock_native import paired_resampling_interval
lower, upper = paired_resampling_interval(
    metric="normalized_nll_per_utf8_byte",
    baseline_scores=[1.0, 2.0, 3.0],
    subject_scores=[1.1, 1.9, 2.8],
    schedule_sha256="a" * 64,
)
```

The named NLL metric computes the ratio of separate rounded means. All other
metric names use the reference's paired difference mean multiplied by 100.
Scores are copied into `Vec<f64>` before the complete computation runs with the
GIL released using PyO3 `Python::detach`. The binding accepts no borrowed Python
objects in the detached computation. Diagnostic functions `sample_indexes`,
`resampling_distribution`, and `_diagnostic_fsum` support protocol verification;
the normal interval call never returns a full replicate array across the FFI.

Supplied vectors must be nonempty, equal-length, and finite. This is stricter
than calling the raw private Python helper with unvalidated arguments and agrees
with its intended validated domain. Seed validation occurs first, including for
one pair. Native invalid-input errors are `ValueError`; intermediate `fsum`
overflow is `OverflowError`; opposite infinite summands are `ValueError`.
Seed and nonpositive-baseline messages match the Python reference. Computed
infinity and NaN are preserved for the existing outer report validator. The
Python integration adapter remains responsible for public exception classes.

## Build and verify

From this directory, with an isolated Python environment and Rust 1.94.0:

```sh
python -m pip install maturin==1.12.0
export CARGO_TARGET_DIR=/Volumes/InvarLock-Rust-Dev/build/native
cargo test --locked --offline -p invarlock-native-core
cargo test --locked --offline --release -p invarlock-native-core
cargo clippy --locked --offline --workspace --all-targets -- -D warnings
cargo fmt --all --check
maturin build --release --locked --offline -i /path/to/python --out /path/to/wheels
python -m pip install /path/to/wheels/invarlock_native_experiment-*.whl
python -m unittest discover -s tests -v
maturin sdist --out /path/to/sdists
```

The differential tests require this checkout's unchanged Python source at
`../src/invarlock/evidence_pack_contract.py`. They compile the exact helper AST
nodes to avoid importing optional evaluation dependencies. They compare every
replicate and interval endpoint bit-for-bit (NaN by classification), complete draw
sets and u64 counter boundaries, singleton behavior, CPython summation final
rounding/cancellation/subnormals, deterministic randomized finite inputs, and
relevant failure classes/messages. Rust golden tests also exercise complete
nonconstant intervals independently of Python. Tests establish these exercised
behaviors; they do not establish parity over all binary64 inputs or platforms.

The source distribution includes both workspace crates, Cargo.lock,
`rust-toolchain.toml`, tests and licenses. Its Rust workspace builds independently of the checkout; differential
reference tests still require the checkout source. Version pins are sha2 0.10.9,
PyO3 0.29.0 and maturin 1.12.0. Apache 2.0 covers original code; adapted CPython
summation retains PSF terms as described in `THIRD_PARTY_NOTICES.md`.

## Qualification scope

`rust-toolchain.toml` pins Rust 1.94.0 with the minimal profile. The manifest
`rust-version = "1.83"` is a declared minimum, not tested compatibility; only
Rust 1.94.0 is qualified here. Rustup honors the toolchain file automatically.
This host uses Homebrew Rust without rustup, so the build records the actual
`rustc --version` and `cargo --version`; both report 1.94.0. Other installations
should verify those versions before running the experiment.

The primary experiment targets CPython 3.12 on macOS arm64 with Rust 1.94.0;
CPython 3.14 builds and differential runs are exploratory smoke coverage. Wheels
for the two interpreter ABIs are retained separately. The test suite exercises
both implementations on this host; it makes no cross-platform or performance
claim. Benchmarking and any public integration decisions belong to the separate
experiment harness. The original Python package remains the default.
