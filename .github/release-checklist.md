# Release Evidence Checklist

Use this maintainer checklist before cutting a release candidate or tagging a
release. It is intentionally outside the published docs tree because it tracks
release evidence and operator gates rather than the stable user contract.

## Required Gates

- [ ] `make verify`
- [ ] `make coverage-enforce`
- [ ] `make dist-check`
- [ ] `make security`
- [ ] `make container-front-door-smoke` or the matching Podman target
- [ ] `make release-evidence-check`

## Required Evidence

- [ ] Wheel and sdist artifacts exist under `dist/`.
- [ ] Wheel and sdist hashes are recorded in
  `artifacts/release/wheel-sdist-hashes.txt`.
- [ ] SBOM artifact exists at `artifacts/supply-chain/sbom.json`.
- [ ] Runtime image digest is recorded in
  `artifacts/release/runtime-image-digest.txt`.
- [ ] Strict example report exists at
  `artifacts/release/strict/evaluation.report.json`.
- [ ] Strict verifier output exists at `artifacts/release/strict/verify.json`.
- [ ] Offline release bundle was generated with
  `scripts/release/make_offline_bundle.sh`.

## Reviewer Notes

- [ ] Changelog entries are under `[Unreleased]` until the release branch is cut.
- [ ] Version metadata and citation metadata match the intended release.
- [ ] GitHub Actions for the integration branch are green.
- [ ] Security findings, dependency updates, and open release-blocking PRs were reviewed.
