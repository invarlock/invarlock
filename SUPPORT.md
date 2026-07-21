# Support and questions

This project is maintained on a best-effort basis. There is no guaranteed
response or fix SLA.

## Where to ask

- Use GitHub Discussions, when enabled, for usage questions and design
  discussions. Good topics include preparing an evaluation request, choosing a
  runtime provider, and interpreting a signed verification receipt.
- Open a GitHub issue with the bug report template for reproducible defects or
  regressions.
- Open a feature request for changes to the request, CLI, provider, evidence,
  or documentation surfaces.
- Follow [SECURITY.md](SECURITY.md) for suspected vulnerabilities. Do not
  disclose them in a public issue.

## What to include in a bug report

- InvarLock version from `invarlock --version`.
- Python version, operating system, architecture, and relevant CPU, GPU, or
  accelerator details.
- The exact command or minimal script that failed.
- A minimal redacted evaluation request and, when relevant, its policy.
- The full error and traceback.
- For verification or reporting defects, the smallest safe reproducer using an
  evidence bundle or signed verification receipt. Do not attach signing keys,
  private model artifacts, credentials, or sensitive data.
- Whether the issue reproduces on the latest released version.

## Before filing

1. Search existing issues and discussions.
2. Read the [getting-started guide](docs/user-guide/getting-started.md),
   [CLI reference](docs/reference/cli.md), and
   [troubleshooting guide](docs/user-guide/troubleshooting.md).
3. Confirm the installed command surface with `invarlock --help` and
   `invarlock --version`.
4. Reduce third-party runtime failures to the smallest InvarLock request or
   provider interaction that still reproduces the problem.

The issue tracker is not a support channel for general model training,
fine-tuning, PyTorch, Transformers, hardware, or cloud-platform questions that
do not involve an InvarLock contract or transaction.
