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
- [ ] `make guard-validation-smoke`
- [ ] `make release-evidence-check`
- [ ] `make empirical-guard-evidence-check` when the release claims new or
  expanded guard calibration, model-family calibration, or support promotion.

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
- [ ] Guard-validation smoke JSON exists at
  `artifacts/guard-validation/guard-validation-smoke.json`.
- [ ] Guard-validation smoke Markdown exists at
  `artifacts/guard-validation/guard-validation-smoke.md`.
- [ ] Offline release bundle was generated with
  `scripts/release/make_offline_bundle.sh` under `artifacts/release/offline/`.
- [ ] Remote GPU evidence manifest exists under `artifacts/release/` when the
  release claim includes external GPU validation, and it records remote paths,
  SHA-256 hashes, runtime image digest, source commit, strict pack verification,
  and backend inventory/smoke evidence for every quantized-subject adapter run.
- [ ] Empirical guard-evidence manifest exists at
  `artifacts/guard-validation/empirical/manifest.json` when empirical guard
  evidence is required by the release claim.

## Reviewer Notes

- [ ] Changelog entries are under `[Unreleased]` until the release branch is cut.
- [ ] Version metadata and citation metadata match the intended release.
- [ ] GitHub Actions for the integration branch are green.
- [ ] Security findings, dependency updates, and open release-blocking PRs were reviewed.
