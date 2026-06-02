<p align="center">
  <picture>
    <source
      media="(prefers-color-scheme: dark)"
      srcset="docs/assets/invarlock-logo-dark.svg"
    />
    <img src="docs/assets/invarlock-logo.svg" alt="InvarLock" />
  </picture>
</p>

<p align="center"><em>Auditable strict verification for edited model checkpoints</em></p>

<p align="center">
  <a href="https://github.com/invarlock/invarlock/actions/workflows/ci.yml">
    <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/invarlock/invarlock/ci.yml?branch=main&logo=github&label=CI" />
  </a>
  <a href="https://pypi.org/project/invarlock/">
    <img alt="PyPI" src="https://badge.fury.io/py/invarlock.svg" />
  </a>
  <a href="https://invarlock.github.io/invarlock/0.9.0/">
    <img alt="Docs" src="https://img.shields.io/badge/docs-quickstart-blue.svg" />
  </a>
  <a href="LICENSE">
    <img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" />
  </a>
  <a href="https://www.python.org/downloads/release/python-3120/">
    <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12+-blue.svg" />
  </a>
</p>

<p align="center">
  <strong>Catch silent quality regressions in edited model checkpoints before they ship.</strong>
</p>

Quantizing, pruning, or otherwise editing a model's weights can silently degrade quality.
InvarLock compares an edited **subject** checkpoint against a fixed **baseline** with paired
evaluation windows, enforces the canonical guard chain (`invariants` -> `spectral` -> `RMT`
-> `variance` -> `invariants`), and produces a machine-readable evaluation report you can gate
in CI.

InvarLock validates baseline-vs-subject checkpoint comparisons. The subject can
come from any external edit workflow: quantization, pruning, LoRA merge,
fine-tuning, or another weight-edit pipeline. The built-in `quant_rtn` edit is
for demos and smoke tests; production workflows are
bring-your-own-edited-checkpoint (BYOE). The repo ships strict-verifiable BYOE
fixtures for dense magnitude pruning and LoRA-merge style subjects under
`public_evidence/byoe_examples/`. Real model runs under
`public_evidence/real_runs/` include an external magnitude-prune BYOE run and a
tiny GPT-2 quantization smoke.

The `public_evidence/` tree separates verifier fixtures from real runs. Fixtures
validate report, runtime-manifest, failure-policy, and evidence-pack contracts;
`public_evidence/real_runs/` contains concrete GPT-2-family `invarlock evaluate`
runs with signed, fingerprint-pinned evidence packs.

## Why InvarLock?

- **Quality gates for edited checkpoints**: catch regressions before deployment.
- **Paired statistical evidence**: primary metrics with confidence intervals.
- **Auditable evidence**: deterministic pairing metadata + policy digests in `evaluation.report.json`.
- **CI/CD-friendly**: stable exit codes, `--json` outputs, and portable “evidence packs”.
- **Offline-first**: network is disabled by default; enable downloads per command.
- **Explicit assurance boundary**: the [trust model](docs/assurance/14-trust-model.md) states the scope of a strict pass.

## Who is this for?

- ML engineers shipping edited model checkpoints, including quantized, pruned,
  fine-tuned, adapter-merged, or otherwise weight-modified variants.
- MLOps and platform teams building CI gates, runtime-provenance verification, and reviewable evaluation artifacts.
- Researchers validating weight-edit, compression, and model-comparison methods with reproducible paired evaluation across text and image-text workflows supported here.

## How it works

```text
Baseline checkpoint         Subject checkpoint
        |                          |
        +------------+-------------+
                     |
                     v
             evaluate command
       container by default; host opt-in
                     |
     deterministic paired windows
     invariants -> spectral -> RMT -> variance -> invariants
                     |
        +------------+-------------+
        |                          |
        v                          v
runs/.../report.json        reports/<name>/
baseline/subject traces       evaluation.report.json
        |                     runtime.manifest.json
        v                             |
report generate / explain             v
                             verify command
                  schema + pairing + gates + provenance
                                      |
                         +------------+------------+
                         |                         |
                         v                         v
                    PASS: promote            FAIL: block
                         |
                         v
                 report html command
                 optional evidence pack
```

## Quick Start

Colab (CPU-friendly):
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/invarlock/invarlock/blob/v0.9.0/notebooks/invarlock_quickstart_cpu.ipynb)

The public front door is `evaluate -> verify -> report html`. The README keeps
the three common onboarding paths separate:

- **Wheel user / reviewer**: install `invarlock`, inspect an existing
  `evaluation.report.json`, and render HTML without cloning the repository.
- **Evaluator**: install `invarlock[hf]` when you want `evaluate` to load
  Hugging Face models and emit a fresh evaluation bundle.
- **Repo maintainer**: clone the repo and build the local runtime image when you
  need maintainer smokes, repo presets, or local container-image iteration.

The default `evaluate` path runs model-loading commands inside the runtime
container and expects an OCI engine such as `podman` or `docker`. Host-side
workflows can opt into `--execution-mode host`, but the default verification
path below expects a container-backed report with sibling runtime provenance.
`evaluate` also defaults to the current strict assurance contract: CI/release
profile, balanced/conservative tier, canonical guard order, complete evidence,
strict paired metrics, and verified runtime provenance are required for an
assurance pass. Use `--assurance off` only for exploratory reports.

```bash
# Evaluator path: create a fresh bundle
pip install "invarlock[hf]"

invarlock --version

# Compare baseline vs subject (downloads require explicit network enable)
INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate --allow-network \
  --baseline gpt2 \
  --subject  distilgpt2 \
  --baseline-adapter auto --subject-adapter auto \
  --profile ci \
  --assurance strict \
  --report-out reports/eval \
  --quiet

# Validate the container-backed evaluation report
test -f reports/eval/runtime.manifest.json
invarlock verify reports/eval/evaluation.report.json

# Render HTML for sharing
invarlock report html -i reports/eval/evaluation.report.json -o reports/eval/evaluation.html
```

Wheel-only verification path:
`pip install invarlock`, `invarlock doctor`,
`invarlock verify /path/to/evaluation.report.json`,
`invarlock report html -i /path/to/evaluation.report.json -o /path/to/evaluation.html`,
and `invarlock report explain --evaluation-report /path/to/evaluation.report.json`.

Repo maintainers can build the local runtime image once with `make runtime-image`;
InvarLock automatically prefers `invarlock-runtime:local` when it is present.

Artifact model:

| Artifact | Produced by | Primary consumers |
| --- | --- | --- |
| `evaluation.report.json` | `invarlock evaluate`, `invarlock report generate --format report` | `invarlock verify`, `invarlock report html`, `invarlock report validate`, `invarlock report explain --evaluation-report`, `invarlock advanced runtime-verify` |
| `report.json` | Baseline/subject run directories under `runs/...` | `invarlock report generate`, `invarlock report explain --subject-report ... --baseline-report ...` |

`invarlock verify` expects `evaluation.report.json`; if you only have a raw
run directory containing `report.json`, first build the reviewer bundle with
`invarlock report generate --run <subject report.json> --baseline-run-report <baseline report.json> --format report -o <output-dir>`.
`invarlock advanced runtime-verify` is narrower: it checks runtime manifest
binding/provenance; report/gate verification remains the reviewer-facing gate.

Example output (abridged; counts vary by profile/config):

```text
INVARLOCK v<version> - EVALUATE
Baseline: gpt2 -> Subject: distilgpt2 - Profile: ci
Status: PASS - Gates: <passed>/<total> passed
Primary metric ratio: <ratio>
Output: reports/eval/evaluation.report.json
Runtime provenance: reports/eval/runtime.manifest.json
```

## Command Surface

- First touch in a fresh install: `invarlock --help`, `invarlock --version`,
  `invarlock report --help`, and `invarlock advanced --help`.
- Core workflow: `invarlock evaluate` -> `invarlock verify` ->
  `invarlock report html`.
- Follow-on report analysis after the core loop: `invarlock report generate`,
  `invarlock report explain`, and `invarlock report validate`.
- Environment and release checks: `invarlock doctor` plus the JSON surfaces
  emitted by `doctor --json` and `advanced plugins ... --json`.
- Runtime-manifest verifier: `invarlock advanced runtime-verify --report <evaluation.report.json> --manifest <runtime.manifest.json>`.
- The public contract catalog exposed by those JSON surfaces includes
  `validation_keys`, `console_labels`, and `metric_kinds`.
- Advanced workflows: `invarlock advanced evidence-pack`, `invarlock advanced policy`,
  `invarlock advanced plugins`, and `invarlock advanced calibrate`.
- Host execution for the core evaluate path uses `--execution-mode host`.
- Optional adapter/backend installs use normal Python extras such as
  `pip install "invarlock[hf]"` rather than CLI install commands.

## Evidence packs (portable evidence bundles)

Evidence packs bundle reports + verification metadata into a distributable artifact.

- Guide: <https://invarlock.github.io/invarlock/0.9.0/user-guide/evidence-packs/>
- Verify from an installed wheel:
  `invarlock advanced evidence-pack verify <dir> --strict --report-assurance strict --expected-fingerprint sha256:<64-hex-chars>`
- Repo harness alternative: `scripts/evidence_packs/verify_pack.sh --pack <dir> --strict --report-assurance strict --expected-fingerprint sha256:<64-hex-chars>`
- For recurring signers, use `--trust-store <json>` or
  `~/.config/invarlock/trusted-signers.json` with the package-native verifier.

Note: `configs/` and most `scripts/` remain repo resources and are not included in
wheels. Installed wheels include the public contracts and the
`invarlock advanced evidence-pack verify` verifier, so installed packages can
check bundles without cloning the repository.

## Installation

```bash
# Minimal CLI (no torch/transformers)
pip install invarlock

# HF workflows (torch/transformers)
pip install "invarlock[hf]"
```

Optional extras: `invarlock[probes]`, `invarlock[gpu]`,
`invarlock[awq,gptq]`, `invarlock[torchao]`, `invarlock[hqq]`,
`invarlock[quanto]`, and `invarlock[compressed-tensors]`. The `awq` and
`gptq` extras use GPTQModel-backed subject loading. Full setup:
<https://invarlock.github.io/invarlock/0.9.0/user-guide/getting-started/>.

The minimal install covers the core verification and reporting flows. Add
`invarlock[hf]` only for model-loading evaluate runs, and use the installed
wheel's evidence-pack verifier when you need to inspect a bundle without cloning
the repository.

## Documentation

- Docs home: <https://invarlock.github.io/invarlock/0.9.0/>
- Quickstart: <https://invarlock.github.io/invarlock/0.9.0/user-guide/quickstart/>
- Compare & evaluate (BYOE): <https://invarlock.github.io/invarlock/0.9.0/user-guide/compare-and-evaluate/>
- Reading a report: <https://invarlock.github.io/invarlock/0.9.0/user-guide/reading-report/>
- CLI reference: <https://invarlock.github.io/invarlock/0.9.0/reference/cli/>
- Assurance case: <https://invarlock.github.io/invarlock/0.9.0/assurance/00-assurance-case/>
  (repo source: `docs/assurance/00-assurance-case.md`)
- Threat model: <https://invarlock.github.io/invarlock/0.9.0/security/threat-model/>

## Community

- Questions/ideas: <https://github.com/invarlock/invarlock/discussions>
- Bug reports: <https://github.com/invarlock/invarlock/issues>
- Contact: <mailto:support@invarlock.dev>

## Citation

If you use InvarLock in scientific work, please cite it (canonical metadata is in `CITATION.cff`):

```bibtex
@software{invarlock,
  title  = {InvarLock: Auditable strict verification for edited model checkpoints},
  author = {{InvarLock}},
  url    = {https://github.com/invarlock/invarlock},
}
```

## Limitations

- Results are baseline-relative to a specific configuration and evidence profile.
- The project scope is edited-checkpoint regression evidence; application-level policy and alignment assessment require separate review.
- Linux is the primary support target; Windows users should use WSL2 or Linux.

## Support matrix

<!-- markdownlint-disable MD060 -->
| Platform               | Status          | Notes                                     |
| ---------------------- | --------------- | ----------------------------------------- |
| Python 3.12+           | ✅ Required      | CI covers 3.12 minimum and 3.13 primary   |
| Linux                  | ✅ Full          | Primary dev target                        |
| macOS (Intel/M-series) | ✅ Full          | MPS supported (default on Apple Silicon)  |
| Windows                | ❌ Not supported | Use WSL2 or a Linux container if required |
| CUDA                   | ✅ Recommended   | For larger models                         |
| CPU                    | ✅ Fallback      | Slower but functional                     |
<!-- markdownlint-enable MD060 -->

## Project status

InvarLock is pre-1.0 as a package, but the core evidence-artifact surfaces are
versioned and intended to be stable within their declared contract versions.
Minor releases may still change non-contract package APIs before 1.0. See
[`docs/reference/contracts.md`](docs/reference/contracts.md) and
[`CHANGELOG.md`](CHANGELOG.md).

For guidance on where to ask questions, how to report bugs, and what to expect in terms of response times, see
[`SUPPORT.md`](SUPPORT.md).

## Contributing

- Contributing guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Local setup: `make dev-install`
- Everyday checks: `make test`, `make lint`, and `make docs-check`
- Maintainer PR gate: `git diff --check origin/staging/next...HEAD`,
  `make lock-sync`, `pre-commit run --all-files --show-diff-on-failure`,
  `make workflow-lint`, `make docs-check`, `make mypy-typed-surface`,
  `make coverage-enforce`, `make packaging-smoke-minimal`, and `make security`
- Broader local confirmation before protected-branch PRs: `make verify`

## License

Apache-2.0 — see `LICENSE`.
