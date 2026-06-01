# Pinned Requirements

Pinned, hash-checked requirements used by GitHub Actions, runtime-image builds,
security tooling, and evidence-pack helper installs.

## Layout

- `requirements/workflows/`: CI, docs, release, runtime-image, and security
  workflow inputs.
- `requirements/evidence-packs/`: helper dependency pins used by evidence-pack
  setup and remote repair flows.

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
