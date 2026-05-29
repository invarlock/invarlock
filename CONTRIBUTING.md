# Contributing to InvarLock

Thank you for your interest in contributing to InvarLock.
This guide summarizes how to set up a dev environment, run checks, and open
high-quality PRs that match the repo layout and tooling.

---

## 1. Development Environment

### 1.1 Prerequisites

- **Python 3.12+** (required)
- **Git**
- **PyTorch / extras** are pulled in via optional dependencies when needed
- **Node.js 22.18+ + npm** (required for docs linting: markdownlint/cspell via local installs or `npx`)

InvarLock runs offline by default. For commands that need downloads
(models/datasets), enable network explicitly per run. Model-loading commands use
the runtime container by default; host-side development outside that
container should use `--execution-mode host` on the public `evaluate` path:

```bash
INVARLOCK_ALLOW_NETWORK=1 invarlock evaluate --execution-mode host --baseline gpt2 --subject distilgpt2 --baseline-adapter auto --subject-adapter auto
# repo preset example:
INVARLOCK_ALLOW_NETWORK=1 invarlock evaluate --execution-mode host --baseline gpt2 --subject gpt2 --baseline-adapter auto --subject-adapter auto --preset configs/presets/causal_lm/wikitext2_512.yaml --profile ci
```

### 1.2 Quick setup (recommended)

```bash
git clone https://github.com/invarlock/invarlock
cd invarlock

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Editable install + dev tooling (pytest, ruff, mypy, mkdocs, etc.)
make dev-install

# Optional if you are not touching docs: install the Node-based linters on demand.
# The repo requires Node 22.18+ for these tools and `npm ci` will fail early on older versions.
npm install --save-dev markdownlint-cli2 cspell

# Pre-commit hooks (lint/format on commit)
pre-commit install
```

If you prefer a direct `pip` invocation:

```bash
pip install -e ".[dev]"
```

### 1.3 Quick verification

You can mirror the main local workflow from `README.md`:

```bash
make test            # pytest -q tests/
make lint            # ruff + mypy
make format          # ruff format/check
make docs            # mkdocs build --strict
make verify          # tests, lint, format, markdownlint
```

For a basic smoke check:

```bash
invarlock --help
invarlock doctor --json
```

For maintainer-facing CLI smoke coverage, use the lane scripts directly:

```bash
bash scripts/smoke/cli_smoke_fast.sh
bash scripts/smoke/cli_smoke_negative.sh
bash scripts/smoke/cli_smoke_realistic.sh

# or dispatch the full matrix
bash scripts/smoke/cli_exhaustive_smoke.sh
```

Lane intent:

- `cli_smoke_fast.sh` covers broad command-surface and positive-path tiny-model flows
- `cli_smoke_negative.sh` covers malformed, policy-fail, and fail-closed categories
- `cli_smoke_realistic.sh` wraps the GPT-2-sized smoke campaign

Delegated config execution and calibration internals re-enter through the
package-internal `python -m invarlock.cli.internal_config_run` module, not a
public CLI command. Public docs and user examples should continue to use
`evaluate`, `verify`, `report`, `doctor`, and `advanced ...`.

---

## 2. Repository Layout

Top-level structure (simplified):

```text
invarlock/
├── src/invarlock/           # Library + CLI implementation
│   ├── core/            # Runner, registry, contracts, auto-tuning, events, types
│   ├── guards/          # Safety mechanisms (invariants, spectral, rmt, variance, policies)
│   ├── eval/            # Evaluation metrics and helpers
│   ├── reporting/       # Evaluation report + reporting surface
│   ├── cli/             # Typer-based CLI app and commands
│   ├── adapters/        # External model adapters (HF, etc.)
│   ├── edits/           # Edit abstractions and demos
│   ├── observability/   # Logging, telemetry, diagnostics
│   ├── plugins/         # Pluggable edits/adapters/guards
│   └── _data/runtime/   # Profiles/tiers included with the package
├── tests/               # Test suite (pytest)
│   ├── unit/            # Focused unit tests
│   ├── integration/     # End-to-end and pipeline tests
│   ├── cli/, core/, ... # Module-focused tests
│   ├── fixtures/        # Shared data + helpers
│   └── README.md        # Extra testing guidance
├── configs/             # Task/edit presets and CI configs
├── scripts/             # Dev/CI scripts (coverage, docs, matrix, etc.)
├── docs/                # MkDocs documentation site
├── Makefile             # Common dev targets (install, test, lint, docs)
└── pyproject.toml       # Packaging + tooling configuration
```

Source code lives exclusively under `src/invarlock/` (namespace `invarlock.*`).
New modules should follow this layout and naming pattern.

---

## 3. Coding Standards & Tooling

### 3.1 Style, typing, and linting

Configured in `pyproject.toml`:

- **Formatter**: Black‑compatible style (88‑char lines)
- **Linting**: Ruff (`python -m ruff check`)
- **Typing**: MyPy (`mypy src/`)
- **Tests**: pytest (configured via `[tool.pytest.ini_options]`)

Typical local invocations (equivalent to Makefile targets):

```bash
make lint
# or, explicitly:
python -m ruff check src/ tests/ scripts/
mypy src/
```

Avoid introducing new one‑letter variable names, keep functions small,
and ensure all public APIs are type‑annotated.

### 3.2 Tests and markers

Tests live under `tests/` and follow the `test_*.py` pattern.
Markers (see `pyproject.toml`) include:

- `unit`: focused, fast unit tests
- `integration`: cross‑component / end‑to‑end paths
- `regression`: stability and regression checks
- `slow`: long‑running tests (often skipped locally)
- `gpu`: CUDA/MPS‑dependent tests
- `notebook`: notebook‑related flows
- `extras`: optional‑dependency tests (e.g., `gptq`, `awq`)
- `manual`: require manual setup or inspection

Common patterns:

```bash
# Fast local run (no integration/slow/manual)
INVARLOCK_LIGHT_IMPORT=1 INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS=0 \
pytest -q -m "not integration and not slow and not manual" tests

# Same lane via Makefile
make test-fast

# Full suite (can be slow)
pytest -q

# Integration tests (auto‑marked via tests/integration/conftest.py)
pytest -q tests/integration

# Same integration/smoke backstop via Makefile
make test-integration
```

For more curated examples (including the CI subset), see `tests/README.md`.

### 3.3 Coverage policy

Coverage configuration lives in `pyproject.toml` under `[tool.coverage.*]`.
The canonical threshold/source-of-truth lives in
`scripts/coverage/coverage_policy.py`; both `scripts/coverage/check_coverage_thresholds.py` and
the Makefile coverage targets consume that shared policy. Per-file branch
coverage thresholds are enforced by `make coverage-enforce`.

Key points:

- Generate coverage with:

  ```bash
  make coverage
  ```

- `make coverage` intentionally includes grouped test sets for the
  core/guards/reporting/calibration suites, the split run/verify/config
  surfaces, targeted CLI command suites, and a small helper-coverage set
  (for example adapter auto-resolution, no-color console handling, and
  guard-overhead extraction) so the project-wide floor reflects the real
  command surface instead of only the split core paths.

- The project-wide floor is enforced at **90%** via pytest-cov in `make coverage`.

- Enforce thresholds:

  ```bash
  make coverage-enforce
  # internally: python scripts/coverage/check_coverage_thresholds.py --coverage reports/cov.xml --json reports/thresholds.json
  ```

- **Critical surface** includes (see `THRESHOLDS`, `CORE_PREFIXES`, and
  `CORE_FILES` in `scripts/coverage/coverage_policy.py`):
  - Core runtime: everything under `src/invarlock/core/`
    (runner, registry, contracts, auto_tuning, events, types, checkpoint, api, retry)
  - Guards: everything under `src/invarlock/guards/`
    (invariants, spectral, spectral_analysis, rmt, variance, policies)
  - Observability/runtime support: everything under `src/invarlock/observability/`
    plus explicit runtime/config entry points such as
    `src/invarlock/config.py` and `src/invarlock/adapters/auto.py`
  - Evaluation/reporting entry points and contracts:
    `src/invarlock/eval/metrics.py`,
    `src/invarlock/reporting/report_contract.py`,
    `src/invarlock/reporting/report_types.py`,
    `src/invarlock/reporting/report_schema.py`,
    `src/invarlock/reporting/validate.py`,
    `src/invarlock/public_contracts.py`,
    `src/invarlock/policy_pack.py`,
  - CLI commands:
    `src/invarlock/cli/commands/run.py`,
    `src/invarlock/cli/commands/evaluate.py`,
    `src/invarlock/cli/app.py`,
    `src/invarlock/cli/commands/policy.py`,
    `src/invarlock/runtime_verify.py`

- Split-aware enforcement uses four levels:
  - **100% branch** for lifecycle shells:
    `src/invarlock/core/runner.py`,
    `src/invarlock/guards/variance.py`,
    `src/invarlock/guards/spectral.py`
  - **100% branch** for pure contract / policy / result / selection helpers:
    `src/invarlock/reporting/report_schema.py`,
    `src/invarlock/public_contracts.py`,
    `src/invarlock/policy_pack.py`,
    `src/invarlock/cli/commands/policy.py`,
    `src/invarlock/cli/verify_output.py`,
    `src/invarlock/core/runner_lifecycle.py`,
    `src/invarlock/core/runner_pairing.py`,
    `src/invarlock/core/runner_services.py`,
    `src/invarlock/guards/variance_policy.py`,
    `src/invarlock/guards/variance_results.py`,
    `src/invarlock/guards/spectral_analysis.py`,
    `src/invarlock/guards/spectral_policy.py`,
    `src/invarlock/guards/spectral_results.py`,
    `src/invarlock/guards/spectral_selection.py`
  - **≥95% branch** for numerical / mutation helpers and large validation helpers:
    `src/invarlock/core/runner_context.py`,
    `src/invarlock/core/runner_eval_phase.py`,
    `src/invarlock/core/runner_latency.py`,
    `src/invarlock/core/runner_eval_windows.py`,
    `src/invarlock/cli/verify_checks.py`,
    `src/invarlock/guards/variance_batching.py`,
    `src/invarlock/guards/variance_evaluation.py`,
    `src/invarlock/guards/variance_prepare.py`,
    `src/invarlock/guards/variance_ops.py`,
    `src/invarlock/guards/variance_scaling.py`,
    `src/invarlock/guards/spectral_control.py`,
    `src/invarlock/guards/spectral_measurement.py`
  - **≥90% branch** for the rest of the critical surface until it is split further,
    including orchestration helpers such as
    `src/invarlock/core/runner_eval_metrics.py`,
    `src/invarlock/core/runner_finalize.py`, and
    `src/invarlock/core/runner_guards.py`

When you modify a file covered by thresholds, please:

- Add or extend tests to keep its measured coverage at or above its enforced floor
- Update/add entries in `scripts/coverage/coverage_policy.py` if you
  expand the critical surface or add new core modules

If the checker reports **“no coverage data present”**, ensure the module is
included in the `--cov=` targets and that a fresh XML report was generated.

---

## 4. Docs and Markdown

The documentation site is built with MkDocs (Material theme) using `mkdocs.yml`.

### 4.1 Local docs workflow

Common commands:

```bash
make docs           # mkdocs build --strict
make docs-serve     # mkdocs serve -a 127.0.0.1:8000
make docs-check     # consolidated docs validation plus curated live examples
make docs-live-fast # curated deterministic live docs/notebook subset
make docs-lint      # markdown + spell lint (via scripts/docs/docs_lint.py)
```

Granular helpers:

```bash
make docs-check-build     # strict build + link checks
make docs-check-links     # link checks only
make docs-lint-markdown   # markdownlint only
make docs-lint-spell      # cspell only
```

### 4.2 Docs linting (markdownlint + cspell)

`python scripts/docs/docs_lint.py` wraps common linters:

- Requires Node 22.18+.
- Uses `markdownlint`, `markdownlint-cli2`, or `npx markdownlint-cli*`
- Uses `cspell` or `npx cspell`

This script runs over `README.md`, `CONTRIBUTING.md`, and all `docs/**/*.md`.
To keep docs CI‑clean, please run at least:

```bash
python scripts/docs/docs_lint.py --markdown   # style
python scripts/docs/docs_lint.py --spell      # spelling
```

If you only have Node installed, the script will look for `node_modules/.bin` first.
To allow `npx` fetching, set `DOCS_LINT_ALLOW_NPX_INSTALL=1` (CI sets this);
otherwise it skips with a warning.

### 4.3 Writing docs

- Keep CLI behavior and configuration examples in sync with `mkdocs.yml`,
  `configs/`, and the actual CLI (`invarlock` commands).
- For API changes, update:
  - In‑code docstrings and type hints
  - Relevant pages under `docs/reference/` and `docs/user-guide/`
- When adding new CLI switches or config fields, update:
  - `docs/reference/cli.md`
  - `docs/reference/config-schema.md` (and run `scripts/checks/check_config_schema_sync.py`)

---

## 5. Development Workflow

### 5.1 Issues and branches

- Prefer opening an issue before larger changes (new guards, adapters, or edits)
- Use descriptive branch names:

  ```bash
  git checkout -b feat/add-hello-guard
  git checkout -b fix/guards-spectral-thresholds
  git checkout -b docs/update-config-schema-reference
  ```

### 5.2 TDD and tests

We aim for test‑driven changes wherever practical:

- Start by adding/expanding tests under `tests/` to describe the behavior
- Run focused invocations:

  ```bash
  pytest -q tests/guards/test_spectral_guard.py::test_basic_spectral_gate
  pytest -q tests/cli/test_doctor_json.py::test_doctor_json_round_trip
  ```

- For broader validation, run:

  ```bash
  make test
  ```

When touching core runtime or guards, bias toward deterministic tests
(fixed seeds, CPU or MPS where possible, controlled randomness).

### 5.3 Code quality checks

Before opening a PR, please run at least:

```bash
make test
make lint
make format
make docs
python scripts/docs/docs_lint.py --markdown
```

For a more complete sweep, use:

```bash
make verify
```

This runs tests, a smoke regression (`scripts/smoke/run_smoke_regression.sh`),
CLI command-surface smokes, ruff lint, ruff format (check mode), packaged
contract sync checks, and markdown lint over docs.

---

## 6. Commits, PRs, and Releases

### 6.1 Commit messages

We use a conventional‑commit style:

- Types: `feat`, `fix`, `docs`, `refactor`, `chore`, `build`, `test`
- Optional scope: `feat(guards): …`, `fix(cli): …`, etc.

Examples:

```text
feat(guards): add variance gate toggle

fix(cli): improve doctor JSON error handling

docs(reference): sync config-schema with runtime
```

Keep commits reasonably scoped and focused on one logical change set.

### 6.2 Pull requests

PRs should:

1. Have a clear title and description
2. Link to any relevant issues
3. Include tests for new or changed behavior
4. Update docs when CLI/config/schema or user‑visible behavior changes
5. Pass CI (tests, lint, docs) before requesting review

A simple checklist to include in the PR description:

```markdown
## Testing
- [ ] Unit tests added/updated
- [ ] Integration/regression tests added/updated (if applicable)

## Docs
- [ ] User docs updated (if user-facing behavior changed)
- [ ] Reference docs updated (if CLI/config/API changed)

## Quality
- [ ] make test
- [ ] make lint
- [ ] make format
- [ ] make docs
```

### 6.3 Release flow (maintainers)

At a high level, maintainers:

1. Bump the version in `pyproject.toml`.
2. Update `CHANGELOG.md` with release notes.
3. Run the full verification and coverage gates (for example, `make verify` and `make coverage-enforce`).
4. Build distribution artifacts (for example, `python -m build` to produce wheel and sdist under `dist/`).
5. Run a pre‑release smoke test from the built artifacts, including a minimal `invarlock evaluate`/`invarlock verify` flow.
6. Tag the release and push tags to GitHub.
7. Let CI publish to PyPI/TestPyPI.

If you are not a maintainer, you do not need to run the release tooling.

---

## 7. Getting Help

- **Bug reports and feature requests**: GitHub Issues
- **Usage questions and discussion**: GitHub Discussions (or issues if in doubt)
- **Support expectations**: see `SUPPORT.md`

When filing issues, please include:

- CLI command (or minimal Python snippet) you ran
- InvarLock version (`invarlock --version`) and Python version
- Platform (OS, CPU/GPU, CUDA/MPS details)
- Relevant logs (e.g., `events.jsonl`, `report.json`, or stack traces)

For security‑sensitive reports, please follow the guidance in `SECURITY.md`.
