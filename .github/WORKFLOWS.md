# GitHub workflows

GitHub Actions validates the same maintained surfaces exposed by the Makefile.
Workflow YAML is linted with `make workflow-lint`.

## Continuous integration

- `ci.yml` runs `verify-fast`, the Python 3.12 suite, coverage enforcement,
  manual full verification, distribution checks, and the tag supply-chain
  backstop.
- `container-front-door-smoke.yml` builds the final runtime image and exercises
  `evaluate`, `verify`, and `report` through the installed command surface.
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

## Local checks

```bash
make verify-fast
make workflow-lint
make docs-check
make security
make dist-check
```

The container journey is opt-in because it builds an image:

```bash
make container-front-door-smoke
```

Dependabot version updates target `staging/next`. Security updates opened
against the default branch are blocked until the equivalent change has passed
through the integration branch.
