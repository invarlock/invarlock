<p align="center">
  <picture>
    <source
      media="(prefers-color-scheme: dark)"
      srcset="docs/assets/invarlock-logo-dark.svg"
    />
    <img src="docs/assets/invarlock-logo.svg" alt="InvarLock" />
  </picture>
</p>

<p align="center"><em>Edit‑agnostic robustness reports for weight edits</em></p>

<p align="center">
  <a href="https://github.com/invarlock/invarlock/actions/workflows/ci.yml">
    <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/invarlock/invarlock/ci.yml?branch=main&logo=github&label=CI" />
  </a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/invarlock/invarlock">
    <img alt="OpenSSF Scorecard" src="https://api.scorecard.dev/projects/github.com/invarlock/invarlock/badge" />
  </a>
  <a href="https://pypi.org/project/invarlock/">
    <img alt="PyPI" src="https://badge.fury.io/py/invarlock.svg" />
  </a>
  <a href="https://github.com/invarlock/invarlock/blob/main/docs/user-guide/quickstart.md">
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
  <strong>Catch silent quality regressions from quantization, pruning, and weight edits before they ship.</strong>
</p>

Quantizing, pruning, or otherwise editing a model’s weights can silently degrade quality.
InvarLock compares an edited **subject** checkpoint against a fixed **baseline** with paired
evaluation windows, enforces the canonical guard chain (`invariants` → `spectral` → `RMT`
→ `variance` → `invariants`), and produces a machine-readable evaluation report you can gate
in CI.

## Why InvarLock?

- **Quality gates for edited checkpoints**: catch regressions before deployment.
- **Statistical guarantees**: paired primary metrics with confidence intervals.
- **Auditable evidence**: deterministic pairing metadata + policy digests in `evaluation.report.json`.
- **CI/CD-friendly**: stable exit codes, `--json` outputs, and portable “proof packs”.
- **Offline-first**: network is disabled by default; enable downloads per command.

## Who is this for?

- ML engineers shipping edited model checkpoints, including quantized, pruned, fine-tuned, or otherwise weight-modified variants.
- MLOps and platform teams building CI gates, attested verification, and reviewable evaluation artifacts.
- Researchers validating weight-edit, compression, and model-comparison methods with reproducible paired evaluation across text and image-text workflows supported here.

## How it works

```text
┌───────────────────────┐     ┌────────────────────────────────────────────┐
│ Baseline (checkpoint) │────►│                                            │
└───────────────────────┘     │  invarlock evaluate                        │
                              │  ├─► Paired windows (deterministic)        │
┌───────────────────────┐     │  ├─► GuardChain pipeline                   │
│ Subject  (checkpoint) │────►│  │   └─► invariants → spectral → RMT → VE  │
└───────────────────────┘     │  └─► Emit: evaluation.report.json          │
                              │                                            │
                              └────────────────────────────────────────────┘
                                                     │
                                     ┌───────────────┴───────────────┐
                                     ▼                               ▼
                                 ✅ PASS                          ❌ FAIL
                                 (ship)                          (rollback)

```

## Quick start

Colab (CPU-friendly):
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/invarlock/invarlock/blob/main/notebooks/invarlock_quickstart_cpu.ipynb)

The secure-default CLI path runs model-loading commands inside the runtime
container and expects an OCI container engine such as `docker` or `podman`.
In a repo checkout, build the local runtime image once with
`make runtime-image`; InvarLock automatically prefers
`invarlock-runtime:local` when it is present. Trusted local workflows can opt
into host execution explicitly with `--assurance trusted-local` on `invarlock evaluate`, but
the attested verification step below expects container execution. The
quickstart block below assumes a repo checkout; do not skip
`make runtime-image` if you want the attested container path.

```bash
# Repo-checkout quickstart for the attested container path
# HF adapter stack (torch/transformers)
pip install "invarlock[hf]"

# Required in a repo checkout for the attested path; do not skip this step.
make runtime-image

# Version + report schema (when available)
invarlock --version

# Compare baseline vs subject (downloads require explicit network enable)
# Secure-default execution uses the runtime container and writes
# reports/eval/runtime.manifest.json next to evaluation.report.json.
invarlock evaluate --allow-network \
  --baseline gpt2 \
  --subject  distilgpt2 \
  --adapter auto \
  --profile ci \
  --report-out reports/eval \
  --quiet

# Validate the attested evaluation report
test -f reports/eval/runtime.manifest.json
invarlock verify --json reports/eval/evaluation.report.json

# Render HTML for sharing
invarlock report html -i reports/eval/evaluation.report.json -o reports/eval/evaluation.html
```

If you pass a directory to `invarlock report generate` or
`invarlock report explain`, it must contain canonical `report.json`.
`invarlock report html` expects canonical `evaluation.report.json`.

Example output (abridged; counts vary by profile/config):

```text
INVARLOCK v<version> · EVALUATE
Baseline: gpt2 -> Subject: gpt2 · Profile: dev
Status: PASS · Gates: <passed>/<total> passed
Primary metric ratio: <ratio>
Output: reports/eval/evaluation.report.json
Attestation: reports/eval/runtime.manifest.json
```

## Command Surface

- Core workflow: `invarlock evaluate` → `invarlock verify` →
  `invarlock report html`.
- Advanced workflows live under `invarlock advanced ...`.
- Trusted host execution for the core evaluate path uses `--assurance trusted-local`.
- Optional adapter/backend installs use normal Python extras such as
  `pip install "invarlock[hf]"` rather than CLI install commands.

## Proof packs (portable evidence bundles)

Proof packs bundle reports + verification metadata into a distributable artifact.

- Guide: <https://github.com/invarlock/invarlock/blob/main/docs/user-guide/proof-packs.md>
- Verify from an installed wheel:
  `invarlock advanced proof-pack verify <dir> --strict`
- Repo harness alternative: `scripts/proof_packs/verify_pack.sh --pack <dir> --strict`

Note: `configs/` and most `scripts/` remain repo resources and are not included in
wheels. Installed wheels include the public contracts and the
`invarlock advanced proof-pack verify` verifier.

## Installation

```bash
# Minimal CLI (no torch/transformers)
pip install invarlock

# HF workflows (torch/transformers)
pip install "invarlock[hf]"
```

Optional extras: `invarlock[gpu]`, `invarlock[awq,gptq]`. Full setup: <https://github.com/invarlock/invarlock/blob/main/docs/user-guide/getting-started.md>.

## Documentation

- Quickstart: <https://github.com/invarlock/invarlock/blob/main/docs/user-guide/quickstart.md>
- Compare & evaluate (BYOE): <https://github.com/invarlock/invarlock/blob/main/docs/user-guide/compare-and-evaluate.md>
- Reading a report: <https://github.com/invarlock/invarlock/blob/main/docs/user-guide/reading-report.md>
- CLI reference: <https://github.com/invarlock/invarlock/blob/main/docs/reference/cli.md>
- Assurance case: <https://github.com/invarlock/invarlock/blob/main/docs/assurance/00-assurance-case.md>
- Threat model: <https://github.com/invarlock/invarlock/blob/main/docs/security/threat-model.md>

## Community

- Questions/ideas: <https://github.com/invarlock/invarlock/discussions>
- Bug reports: <https://github.com/invarlock/invarlock/issues>
- Contact: <mailto:support@invarlock.dev>

## Citation

If you use InvarLock in scientific work, please cite it (canonical metadata is in `CITATION.cff`):

```bibtex
@software{invarlock,
  title  = {InvarLock: Edit-agnostic robustness evaluation reports for weight edits},
  author = {{InvarLock}},
  url    = {https://github.com/invarlock/invarlock},
}
```

## Limitations

- InvarLock evaluates an edited model relative to a baseline under a specific configuration; results are not “global” guarantees.
- Not a content-safety/alignment tool.
- Native Windows is not supported (use WSL2 or Linux).

## Support matrix

<!-- markdownlint-disable MD060 -->
| Platform               | Status          | Notes                                     |
| ---------------------- | --------------- | ----------------------------------------- |
| Python 3.12+           | ✅ Required      |                                           |
| Linux                  | ✅ Full          | Primary dev target                        |
| macOS (Intel/M-series) | ✅ Full          | MPS supported (default on Apple Silicon)  |
| Windows                | ❌ Not supported | Use WSL2 or a Linux container if required |
| CUDA                   | ✅ Recommended   | For larger models                         |
| CPU                    | ✅ Fallback      | Slower but functional                     |
<!-- markdownlint-enable MD060 -->

## Project status

InvarLock is pre‑1.0. Until 1.0, minor releases may include breaking changes. See [`CHANGELOG.md`](CHANGELOG.md).

For guidance on where to ask questions, how to report bugs, and what to expect in terms of response times, see
[`SUPPORT.md`](SUPPORT.md).

## Contributing

- Contributing guide: <https://github.com/invarlock/invarlock/blob/main/CONTRIBUTING.md>
- Fast local checks (repo clone):
  - `make` targets auto-select Python 3.12+, preferring an active 3.12 env, `python3.12`, then the Conda env `invarlock-py312` when present.
  - `make dev-install`
  - `make test`
  - `make lint`
  - `make docs-live`

## License

Apache-2.0 — see `LICENSE`.
