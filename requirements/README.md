# Pinned Requirements

Pinned, hash-checked requirements used by GitHub Actions, runtime-image builds,
security tooling, and evidence-pack helper installs.

## Layout

- `requirements/workflows/`: CI, docs, release, runtime-image, and security
  workflow inputs.
- `requirements/evidence-packs/`: helper dependency pins used by evidence-pack
  setup and remote repair flows.

Evidence-pack helper locks are backend-neutral by default. They must not select
Torch, TorchVision, Triton, bitsandbytes, CUDA runtime/toolkit packages, or
NVIDIA runtime libraries. Explicit Torch CPU/CUDA backend selection belongs in
runtime-image workflow locks, or in an explicit host setup step for remote
validation. Optional build-helper locks must be installed with `--no-deps` so
they do not change the selected runtime backend.

## Refresh

Refresh them with:

```bash
bash scripts/security/refresh_pinned_requirements.sh
```

After refreshing, run the relevant lock/security checks before opening a PR:

```bash
make lock-sync
make cve-audit
```
