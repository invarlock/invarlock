# GitHub workflows

GitHub Actions validates the same maintained surfaces exposed by the Makefile.
Workflow YAML is linted with `make workflow-lint`.

## Continuous integration

- `ci.yml` runs `verify-fast`, the Python 3.12 suite, coverage enforcement,
  manual full verification, distribution checks, and the tag supply-chain
  backstop. Its fast job also downloads a checksum-pinned KitOps executable
  and exercises real package creation, repackaging, and recipient validation.
- `container-front-door-smoke.yml` builds the final runtime image and exercises
  `evaluate`, `verify`, and `report` through the installed command surface.
  It also checks network isolation with positive controls, resource limits,
  interruption, exact-container cleanup, and failed evidence publication.
- `pre-commit.yml` runs the repository pre-commit hooks.
- `repo-hygiene.yml` rejects generated artifacts and oversized files.

## Documentation

- `docs-ci.yml` lints and builds the current documentation and smoke-checks the
  documented CLI command surface.
- `docs-publish.yml` serializes MkDocs publication to `gh-pages`. `main` pushes
  update `latest`, while the production release workflow calls it from the
  exact release tag to update the immutable version path, `latest`, and
  `stable` in one commit. Release branches cannot publish implicitly.

## Security and release

- `codeql.yml` performs static analysis.
- `supply-chain-pr.yml` audits the core and Hugging Face install surfaces and
  scans the pull-request delta for secrets.
- `scorecards.yml` publishes OpenSSF Scorecard results.
- `secret-history.yml` is the scheduled full-history secret scan.
- `dependabot-main-guard.yml` keeps dependency updates on `staging/next`.
- `release.yml` validates pre-tag candidates and builds, attests, and publishes
  tagged Python distributions. After verified production publication, it calls
  the reusable documentation publisher from the same tag; TestPyPI and
  bootstrap runs cannot publish documentation. A manual run with publication
  disabled and a candidate version exercises the Linux release gates without
  creating or moving a tag.

The release workflow builds, validates, attests, and publishes five Python
distributions: `invarlock`, `invarlock-diagnostics`,
`invarlock-runtime-gguf`, `invarlock-runtime-hf-vision-text`, and
`invarlock-runtime-tensorrt-llm`. The optional
packages live under `addins/`; their provider-specific runtime dependencies
stay outside the core wheel.
Candidate and published core wheels exercise the standalone pipeline workflow,
including signing, independent verification, reports, and rejection exit codes,
before the optional packages are installed.

## Local checks

```bash
make verify-fast
make workflow-lint
make docs-check
make security
make dist-check
```

The container journey is opt-in because it builds an image from authenticated
committed source. Create the archive with `scripts/qualification_source.py`
and supply `RUNTIME_SOURCE_COMMIT`, `RUNTIME_SOURCE_BUNDLE`, and
`RUNTIME_SOURCE_BUNDLE_SHA256` to `make container-front-door-smoke`.
The [workflow](workflows/container-front-door-smoke.yml) contains the complete
source authentication and installed-wheel procedure.

## Dependency updates

Dependabot version updates target `staging/next`. Security updates opened
against the default branch are blocked until the equivalent change has passed
through the integration branch.

The Python configuration uses the `uv` ecosystem to update `pyproject.toml` and
`uv.lock` together, following the [uv integration guide](https://docs.astral.sh/uv/guides/integration/dependabot/).
Code-owner review is requested through `CODEOWNERS`.

Before merging a dependency update, synchronize each affected hashed workflow
lock, build-tool pin, and tool hook. Dependabot's root lock update does not
regenerate those independent files. Use the existing compile recipes in
`scripts/security/refresh_pinned_requirements.sh`, keeping unrelated versions
fixed, and run the affected installed-package checks as well as `make
lock-sync` and `make security`. Audit failures require remediation even when
the affected dependency predates the pull request.

The fixed evaluator images derive explicitly versioned wheels from
SHA-256-pinned upstream wheels. Their upstream input requirement files remain
in the dependency-audit inventory. Preserve historical signed evidence and its
declared dependency identities when changing the current image build inputs.
