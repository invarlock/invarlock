# Security Architecture

Overview of the core security-related components and defaults.

## Network policy (default deny)

- Outbound network is blocked by default. The CLI calls
  `invarlock.security.enforce_default_security()` at startup.
- Set `INVARLOCK_ALLOW_NETWORK=1` (or `true/yes/on`) to enable downloads per
  command. Example: `INVARLOCK_ALLOW_NETWORK=1 invarlock certify ...`.
- Use `invarlock.security.temporarily_allow_network()` in controlled code blocks
  when you must fetch artifacts during an otherwise offline run.

## Secure temp directories

- `invarlock.security.secure_tempdir()` creates 0o700 temp dirs and removes them on
  exit. Use for transient artifacts that should not be world-readable.
- `invarlock.security.is_secure_path(path)` verifies expected permissions.

## Certificate verification

- `invarlock verify` re-checks schema, pairing math (Δlog → ratio), and drift/overhead gates.
- Use it before promotion or downstream automation to prevent policy regressions.

## Supply chain (reference)

- SBOM generation (see `scripts/generate_sbom.sh`).
- `pip-audit` with a small allowlist (see pip-audit page) in CI.
- Pre-commit formatting/linting and version checks in CI to reduce drift.

## Design principles

- Secure by default (fail safe): network off; strict gates; no secrets in code.
- Explicit enablement for higher-risk operations (downloads, GPU extras).
- Deterministic runs with recorded seeds and env flags for auditability.

## Related docs

- Security Best Practices: ../security/best-practices.md
- Threat Model: ../security/threat-model.md
