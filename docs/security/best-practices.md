# Security Best Practices

Recommended practices for research and production deployments.

## Highlights

- Keep the default network-off posture; opt in per command with
  `INVARLOCK_ALLOW_NETWORK=1` only when required.
- Use isolated environments (pipx/virtualenv/conda) and lock dependencies.
- Validate configuration inputs and paths; avoid user-controlled write
  locations and implicit directory creation.
- Treat models/datasets from untrusted sources as potentially malicious; avoid
  unsafe deserialization.
- Always run `invarlock verify` on certificates before promotion.

## Production Deployment Checklist

Copy-paste checklist for production or CI deployments:

```markdown
## Pre-deployment
- [ ] Network-off by default (`INVARLOCK_ALLOW_NETWORK` unset)
- [ ] Dependencies locked (`pip freeze > requirements.lock`)
- [ ] Python ≥ 3.12
- [ ] pip-audit clean or exceptions documented
- [ ] SBOM generated (`scripts/generate_sbom.sh`)

## Model & Data
- [ ] Model source verified (local path or trusted HF repo)
- [ ] Dataset cached locally; `HF_DATASETS_OFFLINE=1` enforced
- [ ] No `trust_remote_code=true` unless explicitly audited

## Runtime
- [ ] Isolated environment (venv/conda/container)
- [ ] Write paths validated (no user-controlled output dirs)
- [ ] Secrets excluded from configs and logs

## Certification
- [ ] `invarlock certify` completed with `--profile release`
- [ ] `invarlock verify` passes on generated certificate
- [ ] Certificate + baseline report archived together
- [ ] Evidence artifacts retained per retention policy
```

## Environment flags to know

- `INVARLOCK_ALLOW_NETWORK=1` — enable downloads for a command.
- `HF_DATASETS_OFFLINE=1` — force offline reads after warming caches.
- `INVARLOCK_EVIDENCE_DEBUG=1` — write a small guards_evidence.json next to the
  certificate (no large arrays; safe for local debugging).

## Operational tips

- Prefer `pipx` or conda-managed environments for clean installs.
- Keep Python at 3.12+ and update dependencies regularly.
- Use the supply-chain workflow (SBOM + pip-audit + secret scan) as a
  reference; see the allowlist page for current exceptions.

## See also

- [Threat Model](threat-model.md)
- [Security Architecture](architecture.md)
- [pip-audit Allowlist](pip-audit-allowlist.md)
