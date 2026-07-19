<p align="center">
  <picture>
    <source
      media="(prefers-color-scheme: dark)"
      srcset="https://raw.githubusercontent.com/invarlock/invarlock/main/docs/assets/invarlock-logo-dark.svg"
    />
    <img
      src="https://raw.githubusercontent.com/invarlock/invarlock/main/docs/assets/invarlock-logo.svg"
      alt="InvarLock"
    />
  </picture>
</p>

<p align="center"><em>Run paired release-regression checks. Verify the evidence independently. Render one clear report.</em></p>

<p align="center">
  <a href="https://github.com/invarlock/invarlock/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/invarlock/invarlock/ci.yml?branch=main&label=CI&logo=github&labelColor=18150f" /></a>
  <a href="https://pypi.org/project/invarlock/"><img alt="PyPI" src="https://img.shields.io/pypi/v/invarlock?label=PyPI&logo=pypi&labelColor=18150f&color=1f3a7a" /></a>
  <a href="https://invarlock.github.io/invarlock/"><img alt="Docs" src="https://img.shields.io/badge/docs-quickstart-1f3a7a?labelColor=18150f" /></a>
  <a href="https://github.com/invarlock/invarlock/blob/main/LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-1f3a7a?labelColor=18150f" /></a>
  <a href="https://www.python.org/downloads/release/python-3120/"><img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-1f3a7a?logo=python&logoColor=f4efe3&labelColor=18150f" /></a>
</p>

InvarLock is an open-source assurance engine for one paired
baseline-versus-subject release-regression decision. A closed request pins the
two model artifacts, a local JSONL evaluation source, runtime settings, one
built-in metric or scorer binding, and policy. `evaluate` runs both sides on
the same deterministic schedule, publishes a signed evidence bundle, and
records whether the selected paired interval satisfies the policy. A separate
verifier replays the
bundle against independently supplied trust anchors.

```bash
invarlock evaluate request.yaml
invarlock verify evidence/
invarlock report evidence/
```

<p align="center">
  <img
    src="docs/assets/evaluation-verification-flow.svg"
    alt="A pinned paired evaluation request runs baseline and subject providers, publishes signed evidence, is independently verified, and is rendered as a report"
    width="100%"
  />
</p>

## The release-regression decision

Both sides score the same authenticated records in the same order. InvarLock
derives one of two built-in paired comparisons:

| Metric | Point comparison | Policy verdict |
| --- | --- | --- |
| `exact_match` | Subject accuracy minus baseline accuracy, with paired regression/improvement counts and an exact McNemar test | Lower bound of the paired Newcombe 95% interval is at least `delta_min_pp` |
| `normalized_nll_per_utf8_byte` | Ratio of arithmetic means of per-record byte-normalized expected-continuation NLL | Upper bound of the paired schedule-resampling interval is at most `ratio_max` |

For exact match, InvarLock reports baseline-pass to subject-fail regressions,
baseline-fail to subject-pass improvements, the exact two-sided McNemar
probability, and a continuity-corrected paired Newcombe 95% effect-size
interval. For normalized
NLL, it uses 2,048 paired percentile-bootstrap replicates whose index draws are
derived from the authenticated schedule digest. In both cases the policy reads
the conservative interval bound, not the point value alone.

A policy may also require a minimum paired-record count and a maximum interval
width. Those controls are supplied together. When present, the report passes
only when the metric bound, record-count minimum, and precision ceiling all
pass. Preflight can prove the schedule count before execution; it reports the
interval-width check as pending until paired results exist.

Normalized NLL is teacher-forced expected-continuation likelihood regression.
It does not measure general model quality. When both artifacts bind the same
authenticated tokenizer and every pair has the same positive target-token
count, the verifier also renders a token-weighted perplexity ratio as a derived
likelihood interpretation. That derived value has no threshold, interval, or
verdict authority.

For a task-specific deterministic text scorer, `comparison` can select one
fully bound `scorer_extension` instead of a built-in `metric`. The runtime still
collects authenticated expected output, output text, and output digest facts.
An explicitly authorized scorer replays those facts into one `[0,1]`
higher-is-better value per record; core owns the arithmetic means, paired
percentage-point delta, deterministic interval, and policy decision. This
boundary can support separately supplied deterministic F1, structured
extraction, or VQA answer normalization. No such scorer package ships here.
The public CLI loads the exact installed scorer bound by the request only when
`--allow-installed-scorers` is supplied to both `evaluate` and `verify`.

Executable SQL or code scoring, model-based semantic similarity, network or
human scoring, external models, and LLM judges are outside this acceptance
contract. Judge results can be attached as authenticated observations until a
separate deterministic replay contract and calibration justify more.

## Run a comparison

Install the built-in Hugging Face provider and prepare:

- local baseline and subject snapshots;
- a digest-pinned JSONL file with prompt, expected-output, and optional stable
  ID fields;
- a one-metric policy file;
- a digest-addressed InvarLock runtime image available to Docker or Podman; and
- an Ed25519 evidence-signing key available only to the host transaction.

```bash
python -m pip install "invarlock[hf]"
```

Run requests bind a dataset object. `evaluate` authenticates the JSONL bytes
and deterministically prepares the canonical paired schedule inside the
transaction:

```yaml
format_version: invarlock/evaluation-request-v1
comparison:
  baseline:
    artifact:
      path: artifacts/baseline
      model_id: acme/baseline
      locator: hf://acme/baseline@0123456789abcdef0123456789abcdef01234567
    runtime:
      provider: hf_transformers
      settings:
        batch_size: 1
        checkpoint_tree_sha256: "1111111111111111111111111111111111111111111111111111111111111111"
        context_length: 2048
        immutable_revision: 0123456789abcdef0123456789abcdef01234567
        max_output_tokens: 64
        offline: true
        seed: 7
        timeout_seconds: 300
        tokenizer_metadata_sha256: "3333333333333333333333333333333333333333333333333333333333333333"
  subject:
    artifact:
      path: artifacts/subject
      model_id: acme/subject
      locator: hf://acme/subject@fedcba9876543210fedcba9876543210fedcba98
    runtime:
      provider: hf_transformers
      settings:
        batch_size: 1
        checkpoint_tree_sha256: "2222222222222222222222222222222222222222222222222222222222222222"
        context_length: 2048
        immutable_revision: fedcba9876543210fedcba9876543210fedcba98
        max_output_tokens: 64
        offline: true
        seed: 7
        timeout_seconds: 300
        tokenizer_metadata_sha256: "3333333333333333333333333333333333333333333333333333333333333333"
  dataset:
    path: inputs/release-regression.jsonl
    sha256: "4444444444444444444444444444444444444444444444444444444444444444"
    format: jsonl
    name: release-regression
    split: validation
    input_field: prompt
    expected_output_field: expected
    id_field: case_id
  policy: policy/acceptance.json
  task: text_causal
  metric: normalized_nll_per_utf8_byte
execution:
  mode: run
output:
  evidence: artifacts/evidence-001
```

`evaluate` always performs the complete execution-free validation before it
starts model runtimes. Use `--preflight` to stop after that validation and
inspect its machine-readable result without creating output:

```bash
invarlock evaluate request.yaml --signing-key evidence-signer.pem \
  --runtime-image registry.example/invarlock-runtime@sha256:... \
  --preflight --json
```

Preflight emits `invarlock/evaluation-preflight-v2` and checks configuration
and local availability. When the policy includes sample qualification, it also
reports the observed record count and leaves interval width explicitly
`pending_execution`. Continuing with the real evaluation is still required to
establish runtime execution, interval precision, and the policy result.

Replace the illustrative digests with values derived from the exact inputs.
Then invoke the host CLI. In run mode, the host prepares the authenticated
schedule and launches a separately pinned worker for each side; Docker is the
default engine and Podman is supported.

Build the authenticated Git source bundle as shown in the
[runtime-provider guide](docs/user-guide/runtime-providers.md#build-or-obtain-the-runtime-image),
then build and smoke-test the image that matches the intended device:

```bash
mkdir -p artifacts

# CPU, including Apple Silicon through the matching multi-architecture lock
make runtime-image \
  RUNTIME_SOURCE_COMMIT="$SOURCE_COMMIT" \
  RUNTIME_SOURCE_BUNDLE="$SOURCE_BUNDLE" \
  RUNTIME_SOURCE_BUNDLE_SHA256="$SOURCE_BUNDLE_SHA256" \
  RUNTIME_BUILD_STATEMENT="$PWD/artifacts/runtime-build-cpu.json"
make runtime-smoke

# x86_64 NVIDIA CUDA 12.8
make runtime-image-cuda \
  RUNTIME_SOURCE_COMMIT="$SOURCE_COMMIT" \
  RUNTIME_SOURCE_BUNDLE="$SOURCE_BUNDLE" \
  RUNTIME_SOURCE_BUNDLE_SHA256="$SOURCE_BUNDLE_SHA256" \
  RUNTIME_BUILD_STATEMENT="$PWD/artifacts/runtime-build-cuda.json"
make runtime-smoke-cuda
```

```bash
invarlock evaluate request.yaml \
  --signing-key evidence-signer.pem \
  --baseline-runtime-image registry.example/invarlock-runtime@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --baseline-runtime-image-digest sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --subject-runtime-image registry.example/invarlock-runtime@sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd \
  --subject-runtime-image-digest sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd \
  --container-engine docker \
  --baseline-runtime-device cuda:0 \
  --subject-runtime-device cuda:1
```

Shared `--runtime-image`, `--runtime-image-digest`, `--runtime-device`, and
`--runtime-entrypoint` options remain convenient defaults when both sides use
the same settings. `--runtime-device cuda` exposes a GPU only to an image that
already contains the CUDA runtime; it does not turn the CPU image into a CUDA
image.

Each worker receives its own artifact and support resources read-only plus an
isolated writable output directory. The host validates both outputs, publishes
the no-clobber evidence bundle, and signs it without exposing the private key to
either worker. Workers sharing a generic or identical CUDA device run
sequentially; explicitly different CUDA indexes can run in parallel.

## Verify and report

Verification supplies the expected artifact identities, canonical schedule,
policy, runtime identities, and evidence signer independently of the bundle.
Keep those inputs in one closed verifier-owned profile:

```bash
invarlock verify artifacts/evidence-001/ \
  --trust-profile trust/trust-inputs.json \
  --receipt verification.receipt.json

invarlock report artifacts/evidence-001/ --html evidence.html --explain
```

The evidence signer authenticates the comparison bytes. The verifier decides
whether those bytes satisfy the independently maintained anchors and signs a
separate receipt that binds the profile digest. `report` renders the
signature-authenticated comparison; independent verification remains the
acceptance record. The [CLI reference](docs/reference/cli.md#verify) defines
the closed profile and the equivalent explicit options.

## Import existing measurements

The repository's
[`examples/run/`](https://github.com/invarlock/invarlock/tree/main/examples/run)
journey creates two distinct tiny Hugging Face checkpoints and executes a real
CPU comparison through the pinned OCI path. The repository also includes an
offline import fixture for complete sidecars produced elsewhere. Import mode
requires the canonical schedule, both typed observations and runtime bindings,
and canonical paired records; InvarLock re-derives and checks the full
comparison before publication.

## Providers and diagnostics

Hugging Face Transformers is the built-in reference provider and supports both
built-in metrics. First-party optional GGUF/llama.cpp, TensorRT-LLM, and Hugging
Face vision-text packages add bounded runtime integrations without expanding
the core dependency set. The vision-text add-in is the first structured-content
integration and supports exact-match comparisons over authenticated prompt and
image parts.
See [runtime providers](https://github.com/invarlock/invarlock/blob/main/docs/user-guide/runtime-providers.md).

Spectral, random-matrix, and variance summaries live in the optional
`invarlock-diagnostics` package. They are observation-only diagnostics; the
selected paired comparison and policy exclusively determine acceptance. Their
canonical JSON can be attached to the signed bundle and appears in a separate report
section without changing the verdict. See
[diagnostics](https://github.com/invarlock/invarlock/blob/main/docs/user-guide/diagnostics.md).

## Documentation

- **Run and review:** [getting started](https://github.com/invarlock/invarlock/blob/main/docs/user-guide/getting-started.md),
  [evaluation requests](https://github.com/invarlock/invarlock/blob/main/docs/user-guide/evaluation-request.md),
  [schedule and policy](https://github.com/invarlock/invarlock/blob/main/docs/user-guide/schedule-and-policy.md), and
  [evidence and verification](https://github.com/invarlock/invarlock/blob/main/docs/user-guide/evidence-and-verification.md).
- **Understand the claim:** [assurance case](https://github.com/invarlock/invarlock/blob/main/docs/assurance/assurance-case.md),
  [decision semantics](https://github.com/invarlock/invarlock/blob/main/docs/assurance/decision-semantics.md), and
  [trust model](https://github.com/invarlock/invarlock/blob/main/docs/security/trust-model.md).
- **Integrate:** [CLI](https://github.com/invarlock/invarlock/blob/main/docs/reference/cli.md),
  [contracts](https://github.com/invarlock/invarlock/blob/main/docs/reference/contracts.md),
  [runtime providers](https://github.com/invarlock/invarlock/blob/main/docs/reference/runtime-providers.md), and
  [Python API](https://github.com/invarlock/invarlock/blob/main/docs/reference/api-guide.md).

InvarLock is pre-1.0. Canonical artifact formats carry explicit format versions;
the Python embedding facade may evolve between minor releases.

Questions and design discussions belong in
[GitHub Discussions](https://github.com/invarlock/invarlock/discussions). Report
bugs through [GitHub Issues](https://github.com/invarlock/invarlock/issues) and
security concerns through [SECURITY.md](https://github.com/invarlock/invarlock/blob/main/SECURITY.md).

Apache-2.0 — see the
[license](https://github.com/invarlock/invarlock/blob/main/LICENSE).
