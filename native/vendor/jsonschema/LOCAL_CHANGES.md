# Vendored jsonschema dependency

This is external upstream code from `jsonschema` 0.39.0, not part of the
InvarLock Rust conversion. Its MIT license and copyright notice are retained
unchanged in `LICENSE`.

The source is the published crate at
<https://static.crates.io/crates/jsonschema/jsonschema-0.39.0.crate>.
Its SHA-256 is `757e9e064986e27b3b33ef3f0926a08637745002a8aa1d5c178291061a4530e6`.
All archive files are retained byte-for-byte except the dependency's own
`Cargo.lock` (omitted) and the two files listed in `reference-tracker.patch`.
`upstream-files.sha256.json` records the original hashes of retained files.
The workspace lock controls dependency resolution. No global Cargo cache is modified.

## Local repair

The upstream custom-keyword wrapper discarded `RefTracker` while filling error
context. A validator shared by multiple `$ref` use sites could consequently
report the schema location from its first compilation as its evaluation path.
This caused the wrong first diagnostic even when the earlier use site was absent.

The patch passes the current tracker into `ValidationError::with_context` and
captures its evaluation path with the same upstream function used by built-in
keywords. Schema paths, instance paths, error kinds, validation results and
InvarLock's custom arbitrary-integer semantics remain unchanged. No instance-path
rewrite or schema-order heuristic is added by this dependency patch.

The 54 differential reference cases in
`native/tests/test_contracts_differential.py` cover all eight custom numeric
keywords, a built-in control, both instance orders, nesting, arrays and escaped
reference pointers. Existing contract tests additionally exercise numerical and
packaged diagnostic parity. Qualification results are retained separately.

## Removal condition

Remove this override only when an upstream release preserves custom-keyword
reference traversal, the reference and complete contract differential suites pass,
and retained workload output parity and resource measurements remain acceptable.
Upgrading a version number alone does not satisfy this condition.
