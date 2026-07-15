# Getting Started

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Install InvarLock and complete the core evaluate → verify → report flow. |
| **Audience** | New users setting up their first local or CI evaluation. |
| **Python** | 3.12+ recommended (CI uses 3.13). |
| **Install** | `pip install invarlock` for verification/reporting; add `invarlock[hf]` only for Hugging Face-backed evaluation. |
| **Next step** | [Quickstart](quickstart.md) for copy-paste commands. |

This guide covers installation, environment setup, and the smallest useful
InvarLock workflow: compare a baseline against a subject, verify the
container-backed report, and render HTML for review. The same top-level loop
also underpins the included image-text path when you use the explicit
multimodal preset and provider configuration. The minimal install is enough for
`doctor`, `verify`, and `report html`; use `invarlock[hf]` only when you need
`evaluate` to load Hugging Face models. Treat `evaluate -> verify -> report html`
as the first path to get green before you reach for deeper report-analysis
commands.

Choose the path that matches the reviewed artifact. The checkpoint workflow
below covers Hugging Face/PyTorch inputs. For a deployed GGUF file or
TensorRT-LLM engine, start with [Native Runtime
Providers](native-runtime-providers.md), which authenticates the native
artifact and compares policy-scoped `exact_match` behavior on a reviewed record
schedule.

## Install InvarLock

```bash
# Minimal core (no torch; CLI + schema/verification tools)
pip install invarlock

# Recommended for model-loading and evaluation workflows
pip install "invarlock[hf]"

# Full extras bundle
pip install "invarlock[all]"
```

### Install via pipx

```bash
pipx install --python python3.12 "invarlock[hf]"
```

## Initialize Environment

```bash
conda create -n invarlock python=3.12 -y
conda activate invarlock
pip install "invarlock[hf]"
```

## Verify Installation And Container Runtime

The default strict path requires a running Docker or Podman engine that the
current account can invoke. Check the engine directly, then use `doctor` for
Python, dependency, and accelerator diagnostics:

```bash
# Use `podman info` instead when Podman is your selected engine.
docker info
invarlock doctor
```

## Network Access

InvarLock blocks outbound network by default. When the baseline, edited subject,
or dataset must be downloaded, opt in on the evaluation command with
`--allow-network`, as in the template below.

For offline use, pre-download assets and enforce offline reads with
`HF_DATASETS_OFFLINE=1`. You can also relocate your Hugging Face cache via
`HF_HOME` and `HF_DATASETS_CACHE`.

## First Evaluation

The default `evaluate` path runs model-loading steps inside the runtime
container and emits `runtime.manifest.json` beside the evaluation report. Point
the subject at the actual checkpoint produced by your external edit pipeline;
the template does not manufacture an edited-checkpoint claim from two unrelated
pretrained models.

```bash
BASELINE_CHECKPOINT=/path/to/original-checkpoint
EDITED_SUBJECT_CHECKPOINT=/path/to/checkpoint-produced-by-your-edit-pipeline

INVARLOCK_DEDUP_TEXTS=1 invarlock evaluate --allow-network \
  --baseline "$BASELINE_CHECKPOINT" \
  --subject "$EDITED_SUBJECT_CHECKPOINT" \
  --baseline-adapter auto --subject-adapter auto \
  --profile ci \
  --assurance strict \
  --verbose \
  --report-out reports/eval
```

`--verbose` prints `Baseline report: ...`; the path is also recorded at
`provenance.baseline.report_path` in `evaluation.report.json`. A report-local
`Status: PASS` is provisional: a generated strict report remains
`pending_verifier` until the separate strict verifier exits `0`.

When either checkpoint is a remote model ID, add its immutable 40–64 character
lowercase hexadecimal commit with `--baseline-revision` or
`--subject-revision`. Local checkpoint directories are content-hashed
automatically, so they do not take revision flags.

From a repository checkout, add `--preset configs/...` when a checked-in preset
is appropriate. The wheel-first onboarding path uses direct flags and built-in
adapter defaults because repository preset paths are not included in wheels.

## Verify And Render

```bash
: "${TRUSTED_RUNTIME_IMAGE_DIGEST:?Set this from independently reviewed policy}"
BASELINE_RUN_REPORT=/path/to/baseline/run/report.json
ACCEPTANCE_POLICY_PACK=/path/to/acceptance/policy-pack.json
invarlock verify \
  --profile ci \
  --assurance strict \
  --baseline "$BASELINE_RUN_REPORT" \
  --policy-pack "$ACCEPTANCE_POLICY_PACK" \
  --expected-runtime-image-digest "$TRUSTED_RUNTIME_IMAGE_DIGEST" \
  reports/eval/evaluation.report.json
invarlock report html -i reports/eval/evaluation.report.json -o reports/eval/evaluation.html
```

These commands validate the paired math, schema, and runtime provenance, then
render a shareable HTML artifact from the same report.
Obtain the policy pack and digest from reviewed policy channels independently
of the report bundle and its runtime manifest.
The submitted report cannot create an independent trust anchor by building the
pack itself or copying the digest from the manifest. Policy build tooling
serializes the caller's decision; it does not confer authorization. See
[Policy packs](../reference/contracts.md#policy-packs) and the [Runtime
Provenance Guide](../security/runtime-provenance-guide.md).
Set `BASELINE_RUN_REPORT` to the retained raw `report.json` shown by
`evaluate --verbose` or `provenance.baseline.report_path`; strict verification
rejects reconstructed metric fragments.

Artifact model:

| Artifact | Produced by | Primary consumers |
| --- | --- | --- |
| `evaluation.report.json` | `invarlock evaluate`, `invarlock report generate --format report` | `invarlock verify`, `invarlock report html`, `invarlock report validate`, `invarlock report explain --evaluation-report`, `invarlock advanced runtime-verify` |
| `report.json` | Baseline/subject run directories under `runs/...` | `invarlock report generate`, `invarlock report explain --subject-report ... --baseline-report ...` |

## Execution Modes

- `evaluate` defaults to the runtime container (`--execution-mode container`).
- Use `--execution-mode host` only for host-side workflows that intentionally
  bypass container execution.
- `verify` expects `runtime.manifest.json` next to container-backed evaluation reports.

## Learning Paths

| Persona | Path |
| --- | --- |
| **First-time user** | Getting Started → [Quickstart](quickstart.md) → [Compare & evaluate](compare-and-evaluate.md) |
| **Python developer** | Getting Started → [Primary Metric Smoke](primary-metric-smoke.md) → [API Guide](../reference/api-guide.md) |
| **Custom data user** | Getting Started → [Bring Your Own Data](bring-your-own-data.md) → [Config Gallery](config-gallery.md) |
| **Validation engineer** | Getting Started → [Evidence Packs](evidence-packs.md) → [Public Contracts](../reference/contracts.md) |
| **Integration author** | Getting Started → [Integration Examples](integrations.md) → [Compare & evaluate (BYOE)](compare-and-evaluate.md) |
| **Native runtime operator** | Getting Started → [Native Runtime Providers](native-runtime-providers.md) → [Runtime Providers reference](../reference/runtime-providers.md) |
| **Knowledge/self-edit workflow owner** | Getting Started → [Knowledge & self-edit workflows](knowledge-and-self-edit-workflows.md) → [Compare & evaluate (BYOE)](compare-and-evaluate.md) |
| **Security auditor** | Getting Started → [Threat Model](../security/threat-model.md) → [Best Practices](../security/best-practices.md) |

## Advanced Workflows

The simplified public CLI keeps the core path at the top level. Non-core
surfaces live under `invarlock advanced`:

- `invarlock advanced evidence-pack ...`
- `invarlock advanced policy ...`
- `invarlock advanced plugins ...`
- `invarlock advanced runtime-behavior ...`
- `invarlock advanced calibrate ...`

Installed packages also include the evidence-pack verifier, so bundles can be
inspected without cloning the repository:

```bash
invarlock advanced evidence-pack verify <pack> \
  --strict --report-assurance strict \
  --policy-pack "$ACCEPTANCE_POLICY_PACK" \
  --expected-runtime-image-digest "$TRUSTED_RUNTIME_IMAGE_DIGEST"
```

Optional adapter and backend installs use Python extras such as
`pip install "invarlock[awq,gptq]"`; they are not managed through CLI
install or uninstall commands. The `awq` and `gptq` extras use
GPTQModel-backed subject loading.

## Device Support

InvarLock defaults to `--device auto`, probing **CUDA → MPS → CPU** in that
order. All guard calculations and reports are device-agnostic; CUDA is
recommended for larger release-tier workloads, while CPU and MPS remain useful
for local smoke and portability runs.

- `invarlock doctor` reports detected accelerators.
- Use `--device cpu` to force portability runs.
- Use `--profile ci_cpu` for a reduced-window CPU preset when you need a fast
  validation lane.

## Next Steps

| I want to... | Start here |
| --- | --- |
| evaluate my own edited checkpoint workflow | [Compare & evaluate (BYOE)](compare-and-evaluate.md) |
| evaluate a subject from a knowledge-edit or self-edit workflow | [Knowledge & self-edit workflows](knowledge-and-self-edit-workflows.md) |
| attach evidence to an external edit toolchain | [Integration Examples](integrations.md) |
| compare deployed GGUF or TensorRT-LLM artifacts | [Native Runtime Providers](native-runtime-providers.md) |
| understand the CLI commands | [Quickstart](quickstart.md) |
| bring my own evaluation dataset | [Bring Your Own Data](bring-your-own-data.md) |
| see example outputs | [Example Reports](example-reports.md) |
| understand what's in a report | [Reading a report](reading-report.md) |
| use InvarLock programmatically | [API Guide](../reference/api-guide.md) |
| understand the assurance scope | [Assurance Case](../assurance/00-assurance-case.md) |
| set up secure production deployment | [Security Best Practices](../security/best-practices.md) |
