# Pinned Requirements

Pinned, hash-checked requirements support continuous integration,
documentation, release, security, and runtime-image builds.

## Layout

`requirements/workflows/` contains the complete maintained lock surface. The
core and tooling locks stay independent of a model runtime. Hugging Face locks
resolve the published Torch distribution for their target platform, while
runtime-image locks select the container's Torch backend explicitly.

These workflow locks cover repository automation and runtime-image builds; they
are not a substitute for each distribution's declared metadata. The release
build validates the `invarlock`, diagnostics, GGUF connector, Hugging Face
vision-text connector, and TensorRT-LLM connector distributions separately.

## Refresh

Refresh them with:

```bash
bash scripts/security/refresh_pinned_requirements.sh
```

That compiler owns every generated workflow lock except two deliberately
minimal, hand-maintained bootstrap surfaces:

- `pip-bootstrap-py313.txt` copies the exact `pip` version and hashes already
  reviewed in `release-security-py313.txt`;
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
