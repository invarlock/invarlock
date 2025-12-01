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

- Threat Model: ../security/threat-model.md
- Security Architecture: ../security/architecture.md
