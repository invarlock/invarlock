# Security Architecture

Overview of the core security-related components and defaults.

## Network policy (default deny)

- Outbound network is blocked by default. The CLI calls
  `invarlock.security.enforce_default_security()` at startup.
- Set `INVARLOCK_ALLOW_NETWORK=1` (or `true/yes/on`) to enable downloads per
  command. Example: `INVARLOCK_ALLOW_NETWORK=1 invarlock evaluate ...`.
- Use `invarlock.security.temporarily_allow_network()` in controlled code blocks
  when you must fetch artifacts during an otherwise offline run.

## Runtime boundary (default containerized execution)

- `invarlock evaluate`, `invarlock advanced calibrate`, and internal
  config-driven runner flows delegate to the runtime container by default.
- Use `--execution-mode host` on `invarlock evaluate` for public host-side runs, or
  `INVARLOCK_ALLOW_HOST_EXECUTION=1` / `--allow-host-execution` for advanced
  and internal workflows that intentionally bypass that boundary.
- Third-party plugin discovery and remote model code execution are separate
  explicit opt-ins (`INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS=1`,
  `INVARLOCK_ALLOW_REMOTE_CODE=1`).

## Secure temp directories

- `invarlock.security.secure_tempdir()` creates 0o700 temp dirs and removes them on
  exit. Use for transient artifacts that should not be world-readable.
- `invarlock.security.is_secure_path(path)` verifies expected permissions.

## report verification

- `invarlock verify` re-checks schema, pairing math (Δlog → ratio),
  drift/overhead gates, and runtime provenance via `runtime.manifest.json`.
- `invarlock-runtime-verify` is the low-level package-native CLI for direct
  report/manifest checks, and it uses the same Python verifier implementation
  as runtime provenance.
- Product runtime-provenance verification does not depend on an external verifier binary or
  `PATH` lookup, so verifier behavior stays stable across installs.
- Use it before promotion or later automation to prevent policy regressions.

## Supply chain (reference)

- SBOM generation (see `scripts/generate_sbom.sh`), run against the installed-artifact environment in PR/release jobs and the tool environment in the scheduled backstop.
- `pip-audit` with a small allowlist (see pip-audit page) in CI, run against the base install surface plus the pinned `hf` and `advanced` shipped surfaces in PR jobs, and against the installed-artifact release surface in release jobs.
- `gitleaks` history scanning with JSON/SARIF artifacts in PR and release jobs.
- Pre-commit formatting/linting and version checks in CI to reduce drift.

## Design principles

- Locked-down defaults: network off; strict gates; no secrets in code.
- Explicit enablement for higher-risk operations (downloads, GPU extras).
- Deterministic runs with recorded seeds and env flags for auditability.

## Related docs

- Security Best Practices: ../security/best-practices.md
- Threat Model: ../security/threat-model.md
