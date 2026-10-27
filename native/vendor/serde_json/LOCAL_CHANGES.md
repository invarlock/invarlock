# Vendored serde_json dependency

This is external upstream code from `serde_json` 1.0.149, distributed under
`MIT OR Apache-2.0`. The original copyright and license notices are retained
unchanged in `LICENSE-MIT` and `LICENSE-APACHE`.

The published crate is
<https://static.crates.io/crates/serde_json/serde_json-1.0.149.crate>.
`source-provenance.json` records its archive SHA-256, the retained original file
hashes, the complete patch and the exact patched source-tree identity.
All archive files are retained byte-for-byte except the dependency's own
`Cargo.lock` (omitted), `Cargo.toml` and `src/map.rs`. The workspace lock controls
resolution.
No global Cargo cache is modified.

## Local representation change

With `preserve_order`, an object stores its `IndexMap` header behind a lazy
`Option<Box<_>>`. Empty objects share an immutable empty map for borrowed reads
and allocate their own map on insertion. Mutable empty iterators and unsuccessful
removals do not allocate a header. This keeps the large map header out of each
JSON value slot while retaining the original insertion order, key lookup,
entry API, hashing, equality, deep cloning, mutation and owned iteration.
The default BTreeMap representation without `preserve_order` is unchanged.
Only safe Rust is added. Numeric representation and serialization are unchanged.

This patched dependency requires Rust 1.94, matching the qualified workspace
toolchain. Its manifest raises the upstream Rust 1.68 minimum accordingly; the
lazy empty-map implementation uses `std::sync::OnceLock`, which was introduced
after Rust 1.68. The original upstream manifest is retained in `Cargo.toml.orig`.

Nonempty objects acquire an additional allocation and pointer indirection.
The smaller JSON value slots can reduce storage for records containing many
object entries, but the net resource effect depends on workload and allocator.
This change adds a maintained third-party patch; it does not establish a
universal memory or speed advantage.

`native/contracts/tests/compact_map.rs` exercises the public map behavior.
Contract, numerical and complete-workflow differential checks must additionally
cover ordering, diagnostics, authentication, snapshot checks and output bytes
before deployment. Retained resource measurements are qualified separately.

## Removal condition

Remove this override when an upstream representation provides equivalent
compact storage, or when its maintenance or workload costs outweigh its benefit.
Any replacement must pass the complete differential checks and retain acceptable
resource measurements on the deployment platform. A version update alone does
not satisfy this condition.
