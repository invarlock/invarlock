# Third-party notices

The `fsum` function in `core/src/lib.rs` is adapted from `math_fsum` in CPython
3.12.13, `Modules/mathmodule.c`:
[CPython mathmodule.c](https://github.com/python/cpython/blob/v3.12.13/Modules/mathmodule.c)

Copyright (c) 2001–2023 Python Software Foundation. All Rights Reserved.
The complete upstream license is retained in `licenses/CPython-LICENSE.txt`.
Changes: translated finite and special-value summation, partial storage, final
rounding and error handling to Rust; replaced Python iteration/allocation with
Rust iterators and Vec; returned typed errors instead of Python exceptions.
The summation algorithm retains its PSF license. Original experiment code is
Apache License 2.0; see `LICENSE`.

The compensated sum in `evidence/src/qualification.rs` follows the floating
addition operation order in CPython 3.12.13,
[Python/bltinmodule.c](https://github.com/python/cpython/blob/v3.12.13/Python/bltinmodule.c).
The generated Unicode and IDNA tables are derived from the qualified Python
3.12 Unicode and Unicode 3.2 string preparation data. These adaptations retain
the PSF terms in `licenses/CPython-LICENSE.txt`.

`dependency-inventory.json` records every resolved crate version, registry
archive checksum, declared license expression and retained notice hash.
`sbom.cdx.json` records the complete Cargo dependency graph, including dependencies
for platforms that are not linked on the qualified host. These files describe
the lockfile; they do not claim every platform or dependency has been tested.

Dependency notices are bundled under `licenses/dependencies/`. Most are copied
from the exact registry archive. The `uuid-simd` and `vsimd` notices are retained
from their recorded upstream revision
`d74c030d9dc4f3cae02146d1f497ff62726ef09a`, because those archives omit the root
license file. The r-efi notice is its archive's `AUTHORS` file, which contains
the license grant and copyright holders.

## Modified schema dependency

`vendor/jsonschema` retains jsonschema 0.39.0 under its MIT license with a
limited repair to preserve reference traversal for custom-keyword diagnostics.
The accompanying provenance record separates the upstream archive hash from
the exact patch and patched source-tree hashes. The dependency inventory and
SBOM identify this as modified third-party code rather than a workspace crate.
The upstream archive hash does not identify the patched compiled source.

## Modified JSON dependency

`vendor/serde_json` retains `serde_json` 1.0.149 under `MIT OR Apache-2.0` with
its original license and copyright notices. Its local representation patch
stores ordered map headers behind lazy boxes, keeping them out of each JSON
value slot while preserving insertion order and public map behavior. The source
provenance separates the unchanged upstream archive identity from the patch
and patched source-tree identities. This dependency patch adds maintenance
cost; resource effects require workload and deployment-platform qualification.
