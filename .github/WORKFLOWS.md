# GitHub workflows

GitHub Actions validates the same maintained surfaces exposed by the Makefile.
Workflow YAML is linted with `make workflow-lint`.

## Continuous integration

- `ci.yml` runs repository checks, the Python 3.12 suite, Python 3.13 coverage,
  manual full verification, distribution checks, and the tag supply-chain
  backstop. Its `verify-fast` job runs `make verify-checks`, downloads a
  checksum-pinned KitOps executable, exercises installed package journeys and
  checks signed verification at full capacity. Four separate coverage jobs run
  disjoint test groups. The required `coverage` job accepts only complete,
  successful measurements from the same source and enforces every coverage
  threshold. Per-test timings are retained to diagnose slow runs.
- `container-front-door-smoke.yml` builds the final runtime image and exercises
  `evaluate`, `verify`, and `report` through the installed command surface.
  It also checks network isolation with positive controls, resource limits,
  interruption, exact-container cleanup, and failed evidence publication.
- `pre-commit.yml` runs the repository hooks for every pull request, including
  changes to shell scripts, TOML files and dependency locks.
- `repo-hygiene.yml` rejects generated artifacts and oversized files. Obsolete
  runs are cancelled; only the checks that inspect a change's history fetch it.
- `evaluator-sdk.yml` checks the 19 pinned evaluator SDK capture profiles without
  model or service calls. Its separate environments run only for capture-related
  changes or an explicit dispatch. The ordinary distribution jobs also exercise
  all 57 evaluator/scorer journeys in an SDK-free installed recipient.

Python dependency caches use each job's installed workflow locks as their keys.
When adding an installation step or locked environment, include its lockfile in
that job's `cache-dependency-path`.

## Documentation

- `docs-ci.yml` lints and builds the current documentation once and smoke-checks
  the documented CLI command surface through `make docs-live-fast`.
- `docs-publish.yml` serializes MkDocs publication to `gh-pages`. `main` pushes
  update `latest`, while the production release workflow calls it from the
  exact release tag to update the immutable version path, `latest`, and
  `stable` in one commit. Release branches cannot publish implicitly.

## Security and release

- `codeql.yml` analyzes the complete core distribution and maintained scripts.
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

The release workflow builds, validates, attests, and publishes one Python
distribution: `invarlock`. Judge collection, diagnostics, and all maintained
runtime providers are in core; optional dependencies are installed through the
`judge`, `diagnostics`, `vision-text`, and `hf` extras. Provider-specific native
runtime dependencies stay outside the base wheel.
Candidate and published core wheels use `scripts/release/core_wheel_consumers.py`,
the same consumer suite as local installed-wheel validation. It stages quickstart,
captured, judge, three-scorer and retained approval journeys outside the checkout
and checks signing, independent verification, reports and rejection exit codes
without installing optional dependency extras.

## Local checks

```bash
make verify-fast
make workflow-lint
make docs-check
make security
make dist-check
make evaluator-parity-test
make evaluator-sdk-test EVALUATOR=ragas
```

The SDK probes use the maintained evaluator package pins in isolated producer
environments. These pins do not lock every transitive SDK dependency. The
installed recipient gate uses the hash-locked core dependency set. SDK serializer
tests establish compatibility with the exercised source shape; they do not
establish new model or service measurements.

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
