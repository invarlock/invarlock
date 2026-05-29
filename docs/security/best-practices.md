# Security Best Practices

Recommended practices for research and production deployments.

## Highlights

- Keep the default network-off posture; opt in per command with
  `INVARLOCK_ALLOW_NETWORK=1` only when required.
- Keep model-loading commands on the runtime container by default; use
  `invarlock evaluate --execution-mode host` for public host-side workflows.
- Use isolated environments (pipx/virtualenv/conda) and lock dependencies.
- Validate configuration inputs and paths; avoid user-controlled write
  locations and implicit directory creation.
- Treat models/datasets from untrusted sources as potentially malicious; avoid
  unsafe deserialization.
- Always run `invarlock verify` on reports before promotion.

## Production Deployment Checklist

Copy-paste checklist for production or CI deployments:

```markdown
## Pre-deployment
- [ ] Network-off by default (`INVARLOCK_ALLOW_NETWORK` unset)
- [ ] Dependencies locked (`pip freeze > requirements.lock`)
- [ ] Python ≥ 3.12
- [ ] pip-audit clean or exceptions documented
- [ ] SBOM generated from the installed release surface (`scripts/security/generate_sbom.sh --scope install-surface --python ...`)

## Model & Data
- [ ] Model source verified (local path or trusted HF repo)
- [ ] Dataset cached locally; `HF_DATASETS_OFFLINE=1` enforced
- [ ] No `trust_remote_code=true` unless explicitly audited

## Runtime
- [ ] Isolated environment (venv/conda/container)
- [ ] Host execution disabled unless explicitly required
- [ ] Write paths validated (no user-controlled output dirs)
- [ ] Secrets excluded from configs and logs

## Evaluation & Verification
- [ ] `invarlock evaluate` completed with `--profile release`
- [ ] `invarlock verify` passes on generated report bundle
- [ ] `runtime.manifest.json` archived with `evaluation.report.json`
- [ ] report + baseline report archived together
- [ ] Evidence artifacts retained per retention policy
```

## Environment flags to know

- `INVARLOCK_ALLOW_NETWORK=1` — enable downloads for a command.
- `--execution-mode host` — public evaluate opt-in for host-side model loading.
- `INVARLOCK_ALLOW_HOST_EXECUTION=1` — advanced/internal host-bypass opt-in.
- `INVARLOCK_ALLOW_REMOTE_CODE=1` — permit trusted remote model code.
- `INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS=1` — permit trusted third-party plugin discovery.
- `HF_DATASETS_OFFLINE=1` — force offline reads after warming caches.
- `INVARLOCK_EVIDENCE_DEBUG=1` — write a small guards_evidence.json next to the
  report (no large arrays; safe for local debugging).

## Operational tips

- Prefer `pipx` or conda-managed environments for clean installs.
- Keep Python at 3.12+ and update dependencies regularly.
- Use the PR-time supply-chain workflow for pre-merge checks (install-surface
  SBOM + `pip-audit` on the base/`hf`/`advanced` shipped surfaces + `gitleaks`
  history JSON/SARIF artifacts), and keep the
  scheduled/tag CI supply-chain job as the slower backstop. See the allowlist
  page for documented exceptions.

## Release verification

- PyPI is the canonical source for published wheels and source tarballs.
- Git tags are the canonical source for released repository contents and version
  history.
- The release workflow validates the requested tag before build/publish and
  resolves the release from an immutable tag commit SHA.
- The release workflow also generates an install-surface SBOM, gitleaks scan
  artifacts, and build provenance during publishing. Those checks remain part of
  the maintainer verification path even though tagged releases no longer ship a
  separate public bundle page.
- Use a fresh environment to install and smoke-test the published wheel, then
  compare PyPI metadata and the tagged source tree as described in
  [Release Verification](release-verification.md).

## See also

- [Threat Model](threat-model.md)
- [Security Architecture](architecture.md)
- [pip-audit Allowlist](pip-audit-allowlist.md)
