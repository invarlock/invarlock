# Tests

The test tree follows the paired release-regression product: authenticated
schedule preparation, baseline and subject execution, canonical evidence
publication, independent verification, and report rendering.

## Markers

- `unit` covers isolated component behavior.
- `integration` covers external tools or complete user journeys.
- `regression` covers stability and reproducibility checks.
- `slow` covers long-running tests excluded from the fast lane.
- `manual` covers checks that require an environment not present in CI.
- `gpu` covers tests that require an accelerator.
- `notebook` covers Jupyter notebook behavior.
- `extras` covers tests that require optional provider dependencies.

The complete `tests/integration/` subtree receives the `integration` marker
through `tests/integration/conftest.py`.

## Organization

- `tests/core/` covers requests, identities, provider contracts, the public
  engine, schedules, scorer extensions, paired statistics, and runtime
  resources.
- `tests/runtime_providers/` covers the built-in Hugging Face provider and the
  strict identity contracts shared with the GGUF and TensorRT-LLM connectors.
- `tests/runtime/` covers evaluation transactions, runtime evidence, network
  policy, image contracts, and verification helpers.
- `tests/evidence_packs/` covers canonical publication, integrity,
  verification, signed receipts, and rendering.
- `tests/reporting/` covers public evidence schemas and behavioral observation
  validation.
- `tests/cli/` covers the installed `evaluate`, `verify`, and `report` surface,
  including fail-closed behavior.
- `tests/integration/` covers the offline public example, packaging isolation,
  and the opt-in container journey.
- `tests/ci/`, `tests/docs/`, `tests/lint/`, and `tests/scripts/` protect
  repository, documentation, release, and supply-chain contracts.
- `tests/filesystem/` covers atomic filesystem operations.
- Narrow `_support*.py` helpers live beside the tests that share them;
  root-level `_repo_root.py` and `_support_evidence_pack_signing.py` provide
  repository discovery and test-only signing. Tests otherwise construct their
  inputs in pytest temporary directories or use the checked-in public
  transaction under `examples/`.
- Optional-package tests live beside their packages under `addins/*/tests/` and
  run together through `make addins-test`.

Keep new tests with the production surface that owns the behavior. Shared
helpers should use an explicit support name such as `_support_*.py`. Stable,
maintainer-reviewed public fixtures belong under `examples/`; test-specific
material should normally be constructed under `tmp_path`.

## Typical invocations

Run the offline fast lane:

```bash
make test-fast
```

Run the complete integration subtree separately:

```bash
make test-integration
make addins-test
```

Useful trust and release checks include:

```bash
make trust-smoke
make docs-check
make dist-check
```

The maintained verification and coverage targets use pytest-xdist by default:

```bash
make verify-fast
make coverage-enforce
make verify
```

Set `PYTEST_WORKERS=0` for an explicit sequential diagnostic run.

## Test quality and artifacts

Tests must execute production behavior and assert meaningful outcomes. A test
that merely restates fixture contents does not validate the user journey.

The evaluation evidence destination comes from `output.evidence` in the
request. Verification receipts and HTML reports use the explicit destinations
passed to the CLI. These outputs are no-clobber and should be written to pytest
temporary directories. The maintained public fixture stays under `examples/`;
tests should not create ad hoc repository-root output directories.
