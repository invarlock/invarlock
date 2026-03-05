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
evaluation windows, enforces a guard pipeline (invariants → spectral → RMT → variance), and
produces a machine‑readable Evaluation Report you can gate in CI.

## Why InvarLock?

- **Quality gates for weight edits**: catch regressions before deployment.
- **Statistical guarantees**: paired primary metrics with confidence intervals.
- **Auditable evidence**: deterministic pairing metadata + policy digests in `evaluation.report.json`.
- **CI/CD-friendly**: stable exit codes, `--json` outputs, and portable “proof packs”.
- **Offline-first**: network is disabled by default; enable downloads per command.

## Who is this for?

- ML engineers shipping quantized/pruned checkpoints.
- MLOps teams building CI quality gates and reviewable artifacts.
- Researchers validating compression/edit methods with reproducible, paired eval.

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

```bash
# HF adapter stack (torch/transformers)
pip install "invarlock[hf]"

# Version + report schema (when available)
invarlock --version

# Compare baseline vs subject (downloads require explicit network enable)
INVARLOCK_ALLOW_NETWORK=1 invarlock evaluate \
  --baseline gpt2 \
  --subject  gpt2 \
  --adapter auto \
  --profile dev \
  --quiet

# Validate the evaluation report
invarlock verify reports/eval/evaluation.report.json

# Render HTML for sharing
invarlock report html -i reports/eval/evaluation.report.json -o reports/eval/evaluation.html
```

Example output (abridged; counts vary by profile/config):

```text
INVARLOCK v<version> · EVALUATE
Baseline: gpt2 -> Subject: gpt2 · Profile: dev
Status: PASS · Gates: <passed>/<total> passed
Primary metric ratio: <ratio>
Output: reports/eval/evaluation.report.json
```

## Proof packs (portable evidence bundles)

Proof packs bundle reports + verification metadata into a distributable artifact.

- Guide: <https://github.com/invarlock/invarlock/blob/main/docs/user-guide/proof-packs.md>
- Verify: `scripts/proof_packs/verify_pack.sh --pack <dir> --strict` (or `PACK_STRICT_MODE=1 ...`)

Note: `configs/` and `scripts/` are repo resources and are not shipped in wheels; clone the repo to use
presets and proof-pack helpers.

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
- Assurance case: <https://github.com/invarlock/invarlock/blob/main/docs/assurance/00-safety-case.md>
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
  author = {{InvarLock Maintainers}},
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
  - `make dev-install`
  - `make test`
  - `make lint`

## License

Apache-2.0 — see `LICENSE`.
