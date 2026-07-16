# Contributing to InvarLock

Thank you for improving InvarLock. The repository is intentionally centered on
one user journey:

```text
invarlock evaluate request.yaml
invarlock verify evidence/
invarlock report evidence/
```

Changes should make that transaction easier to understand, safer to execute,
or easier to verify.

## Development setup

InvarLock requires Python 3.12 or newer. Clone the repository, create a virtual
environment, and install the development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,hf]"
npm ci
```

Run the local gate before opening a pull request:

```bash
make verify-fast
```

`make verify`, `make verify-fast`, and `make coverage-enforce` use pytest-xdist
automatically. Set `PYTEST_WORKERS=0` when diagnosing a failure sequentially.
`make verify` adds the complete test and documentation build. The opt-in
container journey builds the final runtime image and exercises all three
commands:

```bash
make container-front-door-smoke
```

## Repository shape

- `src/invarlock/` contains the request transaction, provider ABI, canonical
  evidence bundle, independent verifier, and report renderer.
- `contracts/` contains the shipped JSON contracts.
- `addins/` contains the independently installable GGUF, TensorRT-LLM,
  Hugging Face vision-text, and diagnostics packages.
- `tests/` mirrors the maintained runtime, contract, evidence, CLI, and release
  surfaces.
- `scripts/` contains repository checks, release validation, and security
  utilities. It is not a second user interface.

Hugging Face Transformers is the built-in reference provider. New runtime
integrations implement the provider ABI in an optional package. Keep runtime-
specific dependencies out of the core distribution.

## Contract changes

The request, runtime manifest, provider evidence, evidence pack, and signed
verification receipt are security boundaries. Contract changes must:

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

Tests must exercise production code and assert meaningful outcomes. A passing
test that only restates fixture data is not evidence that a user journey works.
`make dist-check` builds and validates the core, diagnostics, GGUF connector,
Hugging Face vision-text connector, and TensorRT-LLM connector distributions.

## Documentation and public text

Document what the current repository does. Keep local paths, hosts, private
artifact locations, credentials, internal run notes, and unrelated product
planning out of public files and pull requests. Examples should use portable
request-relative paths and placeholder digests.

Documentation lint covers maintained Markdown at the repository root and under
`.github/`, `addins/`, `docs/`, `examples/`, `public_evidence/`,
`requirements/`, `scripts/`, and the test-tree overview. Update these public
surfaces together when a product or release boundary changes.

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
Pull requests should explain:

- the user or verifier problem being solved;
- the contract and trust-boundary impact;
- the validation performed; and
- any intentionally unsupported scope.

Do not include generated build output, local evidence, model weights, signing
keys, or caches. Report security issues through [SECURITY.md](SECURITY.md), not
a public issue.

By contributing, you agree that your work is licensed under Apache-2.0.
