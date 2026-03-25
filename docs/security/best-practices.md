# Security Best Practices

Recommended practices for research and production deployments.

## Highlights

- Keep the default network-off posture; opt in per command with
  `INVARLOCK_ALLOW_NETWORK=1` only when required.
- Keep model-loading commands on the runtime container by default; use
  `invarlock evaluate --mode local` for trusted public local workflows.
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
- [ ] SBOM generated (`scripts/generate_sbom.sh`)

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
- `--mode local` — public evaluate opt-in for trusted host-side model loading.
- `INVARLOCK_ALLOW_HOST_EXECUTION=1` — advanced/internal host-bypass opt-in.
- `INVARLOCK_ALLOW_REMOTE_CODE=1` — permit trusted remote model code.
- `INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS=1` — permit trusted third-party plugin discovery.
- `HF_DATASETS_OFFLINE=1` — force offline reads after warming caches.
- `INVARLOCK_EVIDENCE_DEBUG=1` — write a small guards_evidence.json next to the
  report (no large arrays; safe for local debugging).

## Operational tips

- Prefer `pipx` or conda-managed environments for clean installs.
- Keep Python at 3.12+ and update dependencies regularly.
- Use the supply-chain workflow (SBOM + pip-audit + secret scan) as a
  reference; see the allowlist page for current exceptions.

## Release verification

- GitHub Releases are the canonical place to fetch published wheels, source
  tarballs, the CycloneDX SBOM, and the Sigstore/provenance sidecar files for a
  tagged release.
- Tagged releases also include `invarlock-<version>-offline-bundle.tar.gz`, a
  procurement-friendly archive that groups the signed distributions, their
  Sigstore sidecars, the GitHub provenance bundle, the CycloneDX SBOM, a
  release manifest, and verification hints for offline review.
  See [Release Verification](release-verification.md) for the exact offline
  verification flow.
- The `*.whl` and `*.tar.gz` files are the signed distribution artifacts.
- The `*.sigstore` and related certificate files are the verification material
  emitted for those distributions.
- The release bundle also includes the GitHub build-provenance bundle captured
  during publishing so consumers can verify the CI origin of the published
  artifacts.
- The offline bundle tarball itself is also Sigstore-signed on release, so
  buyers can verify the archive before extraction and then verify the inner
  distributions individually.

## See also

- [Threat Model](threat-model.md)
- [Security Architecture](architecture.md)
- [pip-audit Allowlist](pip-audit-allowlist.md)
