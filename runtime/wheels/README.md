# Generated runtime wheels

Run from the repository root before installing a runtime dependency group or
runtime requirements lock:

```bash
python scripts/security/build_hardened_accelerate_wheel.py bootstrap
```

This verifies the pinned upstream Accelerate wheel, derives
`accelerate-1.14.0+invarlock.1-py3-none-any.whl`, and checks its expected digest.
An existing generated wheel is verified again before use. Generated wheels are
ignored by Git. Keep this directory present so core-only `uv` commands can use
the lock without downloading runtime dependencies.

The `hf`, `runtime-test`, `example-peft`, and `example-torchao` repository groups
select this wheel. Bootstrap first, then use `uv sync --locked --group hf` or
`make dev-install`. Hash-locked pip installs also require
`--find-links runtime/wheels`. Public package extras do not supply the runtime.
OCI builds derive the same wheel from the authenticated source archive and
retain it at `/opt/invarlock/runtime-wheels` for dependent runtime images.
