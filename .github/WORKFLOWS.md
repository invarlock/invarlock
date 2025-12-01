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

- **`ci.yml`** - Main CI (curated tests, docs build, supply chain checks)
- **`pre-commit.yml`** - Pre-commit hook validation
- **`repo-hygiene.yml`** - PR hygiene checks (no generated artifacts, no large files, no duplicate tests)

### Security Workflows

- **`codeql.yml`** - CodeQL static analysis (SAST) for security vulnerabilities
- **`dependabot.yml`** (config file) - Automated dependency updates (Python, GitHub Actions, npm)

See also: [`SECURITY.md`](../SECURITY.md) for vulnerability reporting policy.

### Documentation Workflows

- **`docs-ci.yml`** - Documentation validation (build, links, examples, accessibility, preview deploys)

### Release Workflows

- **`release.yml`** - Build and publish to PyPI/TestPyPI

### Benchmark Workflows

- **`guard-effect-benchmark.yml`** - Guard effect benchmarks (manual `workflow_dispatch`, not part of default CI)

## Environment Variables

Key environment variables used across workflows:

| Variable | Description | Default |
|----------|-------------|---------|
| `INVARLOCK_ALLOW_NETWORK` | Enable network access for model downloads | `0` |
| `INVARLOCK_LIGHT_IMPORT` | Light import mode (skip heavy dependencies) | `0` |
| `INVARLOCK_DISABLE_PLUGIN_DISCOVERY` | Disable plugin auto-discovery | `0` |
| `INVARLOCK_OMP_THREADS` | OpenMP thread count | System default |

## Troubleshooting

### Jobs Pending Indefinitely

If GPU jobs are stuck in "pending" state, ensure:
1. A self-hosted runner with matching labels is online
2. The runner has access to the repository
3. The runner service is running (`./run.sh` or systemd service)

### Missing Config Files

If workflows fail with "config file not found" errors, check that the referenced config paths exist in the repository under `configs/`.

### Python/Node.js Version

All workflows use Python 3.12+ and Node.js 18 where needed. Ensure your self-hosted runners have these versions available.

## Running CI Locally with `act`

You can run GitHub Actions workflows locally using [nektos/act](https://github.com/nektos/act), which emulates GitHub's runner environment in Docker containers.

### Installation

```bash
# macOS
brew install act

# Linux (using Go)
go install github.com/nektos/act@latest

# Or download from: https://github.com/nektos/act/releases
```

**Prerequisites**: Docker must be running.

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
