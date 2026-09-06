# Contributing to InvarLock

Thank you for improving InvarLock. The runtime transaction executes or imports
paired model evaluations:

```text
invarlock evaluate request.yaml
invarlock verify evidence/
invarlock report evidence/
```

The separate `invarlock-pipeline` CLI and Python SDK compare existing evaluator
exports, apply metric and slice policies, and verify signed pipeline evidence.
See the [pipeline integration guide](docs/user-guide/pipeline-integration.md).
Changes should make these workflows easier to understand, safer to execute,
or easier to verify. Preserve the distinct assurance meaning of each evidence
format.

## Development setup

InvarLock requires Python 3.12 or newer. Use Python 3.13 to match the main CI
jobs; Python 3.12 has a separate minimum-version gate. Documentation tooling
requires Node.js 22.18 or newer and npm. Clone the repository, create a virtual
environment, and install the development dependencies:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,hf]"
npm ci
```

Install `uv` for lock, distribution, and dependency-audit targets; CI currently
uses version `0.10.10`. Workflow changes also require `actionlint` on `PATH`.
With Go installed:

```bash
python -m pip install uv==0.10.10
go install github.com/rhysd/actionlint/cmd/actionlint@v1.7.7
```

Add Go's binary directory to `PATH`. These tools are not installed by the
Python development extra.

For an exact Linux x86_64 CI reproduction, create a separate Python 3.13
environment and run:

```bash
python -m pip install --require-hashes -r requirements/workflows/ci-hf-py313.txt
python -m build --wheel --no-isolation
python -m pip install --no-deps --force-reinstall dist/*.whl
```

Use the matching `py312` lock for Python 3.12. These locks include the build
requirements needed by `--no-isolation`. Do not use an editable install as
evidence that the wheel contains every module, entry point, and schema.

Run the local gate before opening a pull request:

```bash
make verify-fast
```

`make verify`, `make verify-fast`, and `make coverage-enforce` run independent
suites concurrently and use bounded pytest-xdist workers. Pass overrides as
Make command-line arguments when diagnosing a failure sequentially:

```bash
make verify-fast VERIFY_TARGET_JOBS=1 PYTEST_WORKERS=0
make coverage-enforce COVERAGE_TARGET_JOBS=1 PYTEST_WORKERS=0
```

`make verify` adds the complete test suite and strict documentation build.
Some integration tests download pinned model artifacts. It does not include
the separate coverage, distribution, workflow, lock, or dependency-audit gates
listed below. Run them for the affected surface before requesting review.

## Repository shape

- `src/invarlock/` contains the request transaction, provider ABI, canonical
  evidence bundle, independent verifier, and report renderer. Its `pipeline/`
  package contains the separate paired-export workflow.
- `contracts/` contains the shipped JSON contracts.
- `addins/` contains the independently installable GGUF, TensorRT-LLM,
  Hugging Face vision-text, and diagnostics packages.
- `tests/` mirrors the maintained runtime, contract, evidence, CLI, and release
  surfaces.
- `scripts/` contains repository checks, release validation, and security
  utilities used by maintainers; user-facing operations remain in the installed
  CLI.

Hugging Face Transformers is the built-in reference provider. New runtime
integrations implement the provider ABI in an optional package. Keep runtime-
specific dependencies out of the core distribution.

## Contract changes

The request, runtime manifest, provider evidence, evidence pack, signed
verification receipt, and pipeline run, policy, and evidence formats are
security boundaries. Contract changes must:

1. start with adversarial tests that demonstrate the intended failure mode;
2. keep schemas closed with `additionalProperties: false` where applicable;
3. authenticate every interpreted path, digest, identity, and policy input;
4. preserve independent verifier-owned trust anchors;
5. fail closed on missing, ambiguous, or unsupported inputs; and
6. update the root and packaged contract copies together with
   `make contracts-sync`.

The project is pre-1.0 and currently uses one canonical version of each public
contract. Do not add compatibility readers or parallel contract generations
without a concrete interoperability requirement.

## Provider changes

A provider owns artifact identification, deterministic paired scoring, runtime
facts, and its authenticated evidence sidecars. The core owns pairing, canonical
publication, policy evaluation, verification, and reporting. Providers must not
select acceptance thresholds or mutate the evidence contract.

Provider tests should cover:

- exact artifact identity and immutable runtime bindings;
- offline and no-remote-code behavior;
- pairing and record-order preservation;
- deterministic settings and timeout behavior;
- receipt and scoring-observation cross-binding; and
- rejection of unsupported settings and missing dependencies.

## Tests and quality

Use pytest for behavior and adversarial tests, Ruff for lint and formatting,
mypy for the typed public surface, MkDocs plus markdownlint/cspell for docs, and
actionlint for workflows. Avoid repository-specific meta-frameworks when an
established tool expresses the same check.

Useful targets:

```bash
make test-fast
make trust-smoke
make lint
make docs-check
make workflow-lint
make coverage-enforce
make dist-check
```

Choose additional gates from the change, rather than treating `verify-fast`
as the complete pull-request check:

| Changed surface | Additional validation |
| --- | --- |
| Python behavior or example launcher | Focused failing test first, then `make verify` and the applicable coverage target |
| Coverage across the repository | `make coverage-enforce` on Linux; CI enforces 95% combined and branch coverage, including per-file checks |
| Documentation or public command examples | `make docs-check`; exercise the documented commands |
| Entry points, imports, packaged schemas, or dependencies | `make addins-install-smoke`; this includes `dist-check` and isolated wheel consumers |
| Pipeline CLI or evidence behavior | Build and install the candidate wheel, then run `python examples/pipeline/wheel_smoke.py` |
| Evidence interpretation or verification | `make release-retained-evidence-compatibility`; retain the declared outcomes of historical evidence |
| Inspect qualification semantics | `make evaluator-inspect-semantics`; run a fresh source-bound qualification and preserve historical profiles and evidence |
| Batch evaluator qualification semantics | `make evaluator-batch-semantics`; replay the current profile's native rows and retain separate source-bound qualification artifacts |
| ModelKit package handoff | Run `tests/examples/test_modelkit_handoff.py` and the pinned real-CLI test described in the [handoff guide](docs/user-guide/modelkit-handoff.md) |
| Dependency declarations or locks | `make lock-sync` and `make security`, plus affected installed-package checks |
| GitHub Actions | `make workflow-lint` |

Run `make pre-commit` for the repository hooks. Some hooks rewrite files;
review their changes and repeat affected validation before committing.

The complete coverage gate requires Linux descriptor execution. On another
operating system, run the relevant portable target such as `make
coverage-examples`, and report the full Linux result from CI separately.
Coverage includes newly added example launchers: successful execution in a
separate smoke job does not collect their branch coverage. Add meaningful
tests under `tests/examples/`; do not weaken thresholds to make a change pass.

Tests must exercise production code and assert meaningful outcomes. A passing
test that only restates fixture data is not evidence that a user journey works.
`make dist-check` builds and validates the core, diagnostics, GGUF connector,
Hugging Face vision-text connector, and TensorRT-LLM connector distributions.

For runtime launcher changes, run the opt-in real-container journey with a
working Docker or Podman engine. Commit the source being tested, create its
archive with `scripts/qualification_source.py create`, and supply
`RUNTIME_SOURCE_COMMIT`, `RUNTIME_SOURCE_BUNDLE`, and
`RUNTIME_SOURCE_BUNDLE_SHA256` to `make container-front-door-smoke`. The
[Container Front Door workflow](.github/workflows/container-front-door-smoke.yml)
shows the complete source authentication and installed-wheel procedure.
A skipped container test does not establish isolation or cleanup behavior.
The container gate includes network positive controls, resource limits,
interruption, exact-container cleanup, and failed-transaction publication checks.

Investigate dependency-audit failures even when the affected lock predates the
pull request. Follow the [dependency-audit policy](docs/security/dependency-audit.md)
for remediation and any explicitly approved, time-bounded exception.

## Documentation and public text

Document what the current repository does. Keep local paths, hosts, private
artifact locations, credentials, execution notes intended only for maintainers,
and unrelated product planning out of public files and pull requests. Examples
should use portable request-relative paths and placeholder digests.

Documentation lint discovers every tracked Markdown file through Git. It checks
formatting, spelling, machine-specific paths, credential-like values, and
review-process wording. Update affected public surfaces together when a product
or release boundary changes. Stage intended new source and documentation files
before the final gates so checks that enumerate Git's file inventory include
them. Inspect the staged diff to keep generated artifacts, local evidence, and
secrets out of the change.

### Documentation type contracts

Choose the document type from the reader's task, then make that type visible
in a short opening admonition. The fields in the admonition are a reader
contract, not decorative metadata: they should say why the page exists, who or
what it applies to, and what a reader can decide or accomplish with it.

| Type | Opening contract | Page responsibilities |
| --- | --- | --- |
| User guide | `Outcome`, `Audience`, `Prerequisites` | Lead the reader through a complete task. Show the expected result, how to validate it, and how to recover or continue. |
| Assurance note | `In plain language`, `Question`, `Decision use`, `Evidence` | State the scoped claim or question, develop the argument or derivation, identify runtime enforcement and observable evidence, and name assumptions, defeaters, and limits. |
| Reference | `Surface`, `Stability`, `Use this page when` | Describe the exact current interface or contract. Include defaults, accepted forms, outputs, and failure behavior, with a minimal example and security notes where they affect correct use. |
| Security guidance | `In plain language`, `Objective`, `Assets or boundary`, `Use this page when` | Identify threats and trust assumptions, connect controls to residual risks, and state operational response, non-goals, and authoritative references where applicable. |

Use headings that fit the subject instead of reproducing a rigid outline. A
glossary, acceptance checklist, CLI reference, and threat model should remain
recognizably different documents. They still share the opening contract and
must cover the responsibilities relevant to their type. Omit an inapplicable
section rather than adding filler.

Use `In plain language` to translate assurance and security reasoning into one
direct statement before introducing formal terms. It must explain the practical
meaning rather than repeat the title or weaken the scoped claim.

Keep tutorials and explanations in the user guide, lookup material in
reference pages, claim reasoning in assurance notes, and adversarial analysis
in security guidance. Link across types instead of duplicating substantial
content. End a page with references or related documentation when the reader
needs a next source, but do not add a link list merely to satisfy a template.

Update `CHANGELOG.md` under `Unreleased` by logical user-facing change, not by
enumerating commits.

## Pull requests

Create branches with the `work/` prefix. Keep commits small enough to review,
but group implementation, tests, and documentation for one logical change.
Target `staging/next` for normal integration work. Run the relevant checks on
the final changes, inspect `git diff --check` and `git status`, and include
their outcomes in the pull request. Distinguish passed, failed, and skipped
checks; an earlier commit's result does not validate a later behavior change.
Pull requests should explain:

- the user or verifier problem being solved;
- the contract and trust-boundary impact;
- the validation performed; and
- any intentionally unsupported scope.

Do not include generated build output, local evidence, model weights, signing
keys, or caches. Report security issues through [SECURITY.md](SECURITY.md), not
a public issue.

By contributing, you agree that your work is licensed under Apache-2.0.
