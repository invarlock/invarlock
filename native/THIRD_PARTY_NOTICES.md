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

Pinned Rust dependencies: sha2 0.10.9 (MIT OR Apache-2.0), PyO3 0.29.0
(MIT OR Apache-2.0). Cargo.lock records all transitive versions. Dependency
license files are bundled under `licenses/dependencies/`; their filenames
identify the crate and version. Their original copyright notices are retained.
