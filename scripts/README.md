# Maintainer scripts

The core user journey is implemented by the installed CLI:

```bash
invarlock evaluate request.yaml
invarlock verify evidence/
invarlock report evidence/
```

Scripts in this directory support repository maintenance. They are not a
second product workflow.

## Maintained families

- `checks/` contains the packaged-contract synchronizer, the public-evidence
  index audit, and the source-tree cruft check.
- `release/` validates a clean release checkout and built distributions.
  `make dist-check` binds the core wheel/sdist and all four first-party
  optional wheel/sdist pairs to their exact checkout sources, metadata, and
  entry points. `make addins-install-smoke` then installs the pinned base
  dependency closure and all five wheels in a disposable environment, runs
  `pip check`, and exercises provider discovery and conformance without using
  the checkout or user site. It selects the maintained Python 3.12 or 3.13
  lock for the invoking interpreter.
- `security/` generates the SBOM and runs dependency vulnerability checks.
- `authenticated_runtime_build.py` consumes an authenticated Git archive,
  validates Dockerfile base overrides as named `repository@sha256:...`
  manifest references, and can publish a no-clobber build statement. Raw local
  config IDs remain valid execution identities but are rejected as `FROM`
  inputs.
- `tensorrt_llm_canary_preflight.py` authenticates the closed engine tree,
  tokenizer contract, immutable image reference, and canonical input root
  before the TensorRT-LLM wrapper starts a GPU container.
- `select_workspace_python.sh` selects the repository Python interpreter used
  by the Makefile.

Ruff, mypy, pytest/pytest-cov, MkDocs, markdownlint, cspell, actionlint, build,
twine, pip-audit, and OSV provide the general lint, test, documentation,
packaging, and security gates. The Makefile composes those established tools.

## Public evidence

`scripts/checks/check_public_evidence.py` validates the closed publication
index, carrier layout, local byte summaries, the receipt's Ed25519 signature
and embedded verifier-key fingerprint, and the receipt-to-manifest binding. It
does not substitute for cryptographic `invarlock verify` or signed-receipt
authorization against independent policy, runtime, evidence-signer, and verifier
anchors. An empty index is valid only when it uses the status label
`Evidence not yet created`. The wheel contains a compact index; evidence may
be carried separately as a release asset.

Refresh the byte-identical source and packaged indexes, then check both with:

```bash
make public-evidence-sync
make public-evidence-audit
```

## Release checks

Release preflight is intentionally read-only and does not publish, tag, or
merge:

```bash
make addins-install-smoke
make release-preflight RELEASE_PREFLIGHT_ARGS="\
  --release-sha COMMIT_SHA \
  --expected-version X.Y.Z \
  --hash-manifest PATH/TO/core-dist.sha256"
```

`COMMIT_SHA` must be the lowercase 40-character SHA of the clean checkout. The
hash manifest uses `sha256sum` format and lists exactly the core wheel and sdist
from `dist/` by base name. Pass `--json` when machine-readable output is
required. Preflight does not approve or publish a release.

GGUF, TensorRT-LLM, and Hugging Face vision-text conformance commands are
shipped by their optional first-party distributions under `addins/`. The
release workflow publishes those runtime packages, the diagnostics package,
and the core distribution together.
