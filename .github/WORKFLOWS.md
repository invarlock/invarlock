# GitHub Workflows Documentation

## Overview

The `.github` directory contains CI/CD workflows under `workflows/` and supporting automation
configuration (CODEOWNERS, Dependabot, PR templates) for the InvarLock repository.

### Actionlint Configuration

The `.github/actionlint.yaml` file configures actionlint to recognize custom labels for self-hosted runners:

```yaml
self-hosted-runner:
  labels:
    - gpu
```

## Workflows

### Core CI Workflows (Tracked in Git)

- **`ci.yml`** - Main CI (curated tests, docs build, PR-time `coverage-enforce`, typed-surface `mypy`, manual full verify, and tag-gated supply-chain backstop)
- **`pre-commit.yml`** - Pre-commit hook validation
- **`repo-hygiene.yml`** - PR hygiene checks (no generated artifacts, no large files, no duplicate tests)

### Local Composite Actions

- **`actions/invarlock-report-gate`** - Verifies an existing
  `evaluation.report.json`, renders HTML, writes `invarlock-verify.json`,
  exports MLflow tags, exports a release-review Markdown packet, appends a PR
  summary, and uploads those files as an Actions artifact. The action exposes
  `baseline`, `policy-pack`, `profile`, `assurance`, `runtime-provenance`,
  `expected-runtime-image-digest`, and `warning-policy` inputs so workflows can
  keep verification policy explicit. Strict assurance requires the baseline,
  policy pack, and expected digest to come from independent reviewed channels.

### Security Workflows

- **`codeql.yml`** - CodeQL static analysis (SAST) for security vulnerabilities
- **`supply-chain-pr.yml`** - PR-time supply-chain checks (install-surface SBOM, `pip-audit` on base/`hf`/`advanced` shipped surfaces, `gitleaks` git-delta JSON artifacts)
- **`secret-history.yml`** - Scheduled/manual full-history `gitleaks` backstop
- **`dependabot-main-guard.yml`** - Blocks direct Dependabot PRs to `main`; maintainers must land equivalent dependency fixes on `staging/next` first
- **`dependabot.yml`** (config file) - Automated dependency updates (Python, GitHub Actions, npm)

See also: [`SECURITY.md`](../SECURITY.md) for vulnerability reporting policy.

### Documentation Workflows

- **`docs-ci.yml`** - Documentation validation (build, links, examples, preview deploys)

### Release Workflows

- **`release.yml`** - Tag-gated build and publish workflow for PyPI/TestPyPI

#### Runtime evidence release asset handoff

The compact native-runtime evidence archive is a GitHub Release asset, not a
Python distribution. Before upload, stage it under a tag-and-digest-bound name
and generate its canonical SHA-256 sidecar:

```bash
mkdir -p artifacts/release-runtime
python scripts/release/runtime_release_asset_handoff.py stage \
  --asset runtime-release-evidence.tar.gz \
  --output-dir artifacts/release-runtime \
  --release-tag "$RELEASE_TAG" \
  --expected-source-commit "$SOURCE_COMMIT" \
  --expected-source-archive-sha256 "$SOURCE_ARCHIVE_SHA256" \
  --expected-asset-sha256 "$ASSET_SHA256" \
  --expected-provider llama_cpp \
  --expected-provider tensorrt_llm \
  --expected-qualification llama_cpp:cpu-reference \
  --expected-qualification tensorrt_llm:pair-a \
  --expected-qualification tensorrt_llm:pair-b \
  --require-behavioral-claim
```

The release manager creates the GitHub Release only after the tag publication
has passed. Attach the already-staged pair with the same independent bindings:

```bash
ASSET="artifacts/release-runtime/invarlock-${RELEASE_TAG}-runtime-evidence-source-${SOURCE_COMMIT:0:12}-${ASSET_SHA256}.tar.gz"
python scripts/release/runtime_release_asset_handoff.py upload \
  --asset "$ASSET" \
  --digest-file "${ASSET}.sha256" \
  --repository "$GITHUB_REPOSITORY" \
  --release-tag "$RELEASE_TAG" \
  --expected-release-commit "$RELEASE_COMMIT" \
  --expected-source-commit "$SOURCE_COMMIT" \
  --expected-source-archive-sha256 "$SOURCE_ARCHIVE_SHA256" \
  --expected-asset-sha256 "$ASSET_SHA256" \
  --expected-provider llama_cpp \
  --expected-provider tensorrt_llm \
  --expected-qualification llama_cpp:cpu-reference \
  --expected-qualification tensorrt_llm:pair-a \
  --expected-qualification tensorrt_llm:pair-b \
  --require-behavioral-claim
```

The helper revalidates the archive, provider set, behavior-claim requirement,
evidence source bindings, source-labeled immutable filenames, and sidecar.
The repeated qualification arguments are an exact set, so a missing or
substituted named qualification fails closed.
Upload separately requires an existing non-draft release whose remote tag
resolves to `RELEASE_COMMIT`; it does not claim that this release-management
commit is the commit that produced the evidence. Existing assets are never
replaced, and the uploaded names and sizes are checked before success is
reported. Omit `--require-behavioral-claim` only when the release publishes
runtime qualification receipts without a schedule-level behavior claim.

### Benchmark Workflows

- **`guard-effect-benchmark.yml`** - Paired guard overhead and stability benchmark (manual `workflow_dispatch`, not part of default CI or a detection-efficacy study)

## Environment Variables

Key environment variables used across workflows:

| Variable | Description | Default |
|----------|-------------|---------|
| `INVARLOCK_ALLOW_NETWORK` | Enable network access for model downloads | `0` |
| `INVARLOCK_ALLOW_HOST_EXECUTION` | Permit host-side model-loading commands in workflows that need it | `0` |
| `INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS` | Permit trusted third-party plugin discovery or management | `0` |
| `INVARLOCK_ALLOW_REMOTE_CODE` | Permit trusted remote model code execution when a workflow explicitly needs it | `0` |
| `INVARLOCK_LIGHT_IMPORT` | Light import mode (skip heavy dependencies) | `0` |
| `INVARLOCK_OMP_THREADS` | OpenMP thread count | System default |

Workflows should keep these privilege toggles scoped to the narrowest possible
job or step, emit container-backed outputs, and verify them without bypasses.

## Dependency Update Policy

- Dependabot version-update PRs target `staging/next`.
- Dependabot security-update PRs still originate against the default branch (`main`) because GitHub security updates do not honor `target-branch`.
- The `dependabot-main-guard.yml` workflow intentionally fails direct Dependabot PRs to `main`.
- Maintainers must land the equivalent dependency fix on `staging/next`, validate it there, and let it reach `main` through the normal staging-to-release flow.
- `github/codeql-action` is tracked by Dependabot again; maintainers should review the resulting PRs like any other security-sensitive workflow change.
- The PR supply-chain workflow scans the pull request git delta with
  `gitleaks`, uploads JSON artifacts, audits the built wheel install surface
  for SBOM generation, and runs `pip-audit` against the base, `hf`, and
  `advanced` shipped dependency surfaces.
- The release workflow peels annotated tags to immutable commit SHAs before
  checkout/publish, scans the release delta since the previous release tag,
  uses an installed-wheel environment for its release SBOM, and publishes the
  Python distributions. Compact runtime evidence uses the separately verified
  GitHub Release asset handoff documented above.
- The scheduled secret-history workflow keeps the slower full-history
  `gitleaks` scan out of the tag publish critical path.
- The tag-gated CI supply-chain job remains the slower release backstop and keeps the tool-environment SBOM.
- The PR typed-surface lane covers observability, config loading/runtime, metric resolution, report schema/verification helpers, MI probes, registry metadata including the built-in plugin catalog, runtime-security modules, the split run-orchestrator owner modules, and CLI entrypoints.

## Troubleshooting

### Jobs Pending Indefinitely

If GPU jobs are stuck in "pending" state, ensure:

1. A self-hosted runner with matching labels is online
2. The runner has access to the repository
3. The runner service is running (`./run.sh` or systemd service)

### Missing Config Files

If workflows fail with "config file not found" errors, check that the referenced config paths exist in the repository under `configs/`.

### Python/Node.js Version

All workflows use Python 3.12+ and Node.js 22.18+ where needed. Ensure your self-hosted runners have these versions available.

## Running CI Locally with `act`

You can run GitHub Actions workflows locally using [nektos/act](https://github.com/nektos/act), which emulates GitHub's runner environment in Docker containers. This Docker requirement applies to the local `act` helper path, not to InvarLock's general default runtime-container support.

### Installation

```bash
# macOS
brew install act

# Linux (using Go)
go install github.com/nektos/act@latest

# Or download from: https://github.com/nektos/act/releases
```

**Prerequisites**: Docker must be running for this documented `act` flow.
`make workflow-lint` also requires the CI-pinned `actionlint` binary:

```bash
go install github.com/rhysd/actionlint/cmd/actionlint@v1.7.7
```

`make security` runs SBOM generation and `pip-audit` in an isolated `uv`
security toolchain so it does not add supply-chain tools to the project
virtual environment.

### Quick Start

```bash
# List all available workflows and jobs
make ci-local-list

# Dry run (see what would execute without running)
make ci-local-dry

# Run the main CI tests-docs job
make ci-local

# Run a specific job
make ci-local-job JOB=supply-chain

# Run pre-commit workflow
make ci-local-precommit

# Run direct supply-chain security checks
make security

# Verbose output for debugging
make ci-local-verbose
```

### Direct `act` Commands

```bash
# Run all jobs triggered by push event
act push

# Run a specific workflow file
act push --workflows .github/workflows/ci.yml

# Run a specific job with environment variables
act push --job tests-docs --env INVARLOCK_LIGHT_IMPORT=1

# Use a different event (pull_request)
act pull_request

# Interactive mode - select which jobs to run
act push --interactive

# See the execution graph
act push --graph
```

### Configuration

The repository includes `.actrc` with default settings:

- Uses `catthehacker/ubuntu:act-22.04` image (good balance of size/compatibility)
- Container reuse enabled for faster iteration
- Reads `.env.local` for secrets (create this file locally)

### Creating `.env.local` for Secrets

If workflows need secrets, create `.env.local`:

```bash
# .env.local (gitignored)
GITHUB_TOKEN=ghp_xxxx
PYPI_API_TOKEN=pypi-xxxx
NETLIFY_AUTH_TOKEN=xxxx
```

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "Cannot find Docker" | Start Docker Desktop or `systemctl start docker` |
| Job takes too long | Use smaller image: `-P ubuntu-latest=catthehacker/ubuntu:act-latest` |
| Missing system tools | Use full image: `-P ubuntu-latest=catthehacker/ubuntu:full-22.04` |
| Network issues | Check Docker network; try `act --container-options "--network host"` |
| macOS-specific jobs fail | `act` only supports Linux runners; skip with `--job <linux-job>` |
| Secrets not found | Create `.env.local` or use `--secret-file .secrets` |
| Out of disk space | Run `docker system prune -a` to clean up |

### Limitations

- **macOS runners not supported**: `act` only emulates Linux runners. Use `--job` to skip macOS jobs.
- **GPU jobs not supported**: Jobs requiring `self-hosted, gpu` labels won't run locally.
- **Some GitHub features unavailable**: Caching may behave differently; some `github.*` context values differ.

### Debugging Tips

1. **Add `-v` or `--verbose`** for detailed execution logs
2. **Use `--dryrun`** to see the execution plan without running
3. **Shell into container**: `act push --job tests-docs --reuse -b` keeps container running
4. **Check act logs**: Stored in `~/.local/share/act/` or check Docker logs
