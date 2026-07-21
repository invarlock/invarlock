# Pinned Requirements

Pinned, hash-checked requirements support continuous integration,
documentation, release, security, and runtime-image builds.

## Layout

`requirements/workflows/` contains the complete maintained lock surface. The
core and tooling locks stay independent of a model runtime. Hugging Face locks
resolve the published Torch distribution for their target platform, while
runtime-image locks select the container's Torch backend explicitly.
The CPU and aarch64 locks share `runtime-image.in`. The CUDA runtime uses
`runtime-image-cu126.in` so its platform-specific wheel closure remains
explicit while matching the patched Torch release used by the CPU images.
`lm-evaluation-harness-py312.txt` pins the complete Python 3.12 Linux x86_64
dependency closure used only by the LM Evaluation Harness integration image;
it is compiled from `lm-evaluation-harness.in` against the canonical CPU
runtime input so the derived image cannot resolve a second inference stack at
build time.
`release-install-py312.txt` and `release-install-py313.txt` are the
Python-version-specific, hash-pinned dependency closures installed before the
coordinated local release wheels. Both are compiled from
`release-install.in`, which is the exact union of the external base
dependencies declared by the core and four optional first-party
distributions. That closure includes NumPy for diagnostics and Pillow for the
vision-text host package. Heavy inference stacks exposed only through optional
runtime extras are deliberately outside this coordinated base-install gate.

These workflow locks cover repository automation and runtime-image builds; they
are not a substitute for each distribution's declared metadata. The release
build validates the `invarlock`, diagnostics, GGUF connector, Hugging Face
vision-text connector, and TensorRT-LLM connector distributions separately,
then installs all five wheels together against the matching Python 3.12 or
3.13 closure in a disposable environment.

## Refresh

Refresh them with:

```bash
bash scripts/security/refresh_pinned_requirements.sh
```

That compiler owns every generated workflow lock except two deliberately
minimal, hand-maintained bootstrap surfaces:

- `pip-bootstrap.txt` contains the Python-version-independent pip bootstrap
  wheel and source hashes already reviewed in `release-security-py313.txt`;
- `runtime-wheel-build-py312.txt` contains only the versions and hashes needed
  to build the core and runtime-provider wheels without build isolation.

When their source pins change, update those two files in the same review,
confirm each hash against the downloaded index artifact, and keep their inline
ownership comments. The refresh script does not rewrite them.

After refreshing, run the lock and security checks:

```bash
make lock-sync
make cve-audit
```
