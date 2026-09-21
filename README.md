<p align="center">
  <picture>
    <source
      media="(prefers-color-scheme: dark)"
      srcset="docs/assets/invarlock-logo-dark.svg"
    />
    <img
      src="docs/assets/invarlock-logo.svg"
      alt="InvarLock"
      width="420"
    />
  </picture>
</p>

<p align="center"><em>Evaluate model changes. Verify the evidence. Share the result.</em></p>

<p align="center">
  <a href="https://github.com/invarlock/invarlock/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/invarlock/invarlock/ci.yml?branch=main&label=CI&logo=github&labelColor=18150f" /></a>
  <a href="https://pypi.org/project/invarlock/"><img alt="PyPI" src="https://img.shields.io/pypi/v/invarlock?label=PyPI&logo=pypi&labelColor=18150f&color=1f3a7a" /></a>
  <a href="https://invarlock.github.io/invarlock/"><img alt="Docs" src="https://img.shields.io/badge/docs-quickstart-1f3a7a?labelColor=18150f" /></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-1f3a7a?labelColor=18150f" /></a>
  <a href="https://www.python.org/downloads/release/python-3120/"><img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-1f3a7a?logo=python&logoColor=f4efe3&labelColor=18150f" /></a>
</p>

**InvarLock evaluates model changes and produces evidence that another team can
verify independently.** Compare a candidate (the *subject*) with an approved
baseline, using tests and acceptance thresholds you choose. Run a supported
comparison through InvarLock, or bring per-case records from your existing
evaluation workflow.

The result is a signed evidence bundle, a separate verification receipt and a
report explaining the comparison. Your customer or internal reviewer can check
the retained result against agreed inputs and policy without rerunning model
inference.

- **Three native scorers:** exact match, normalized NLL and bounded LLM judging.
- **Existing workflows:** supported evaluator exports and a Python capture SDK.
- **Offline verification:** replay the supported analysis with recipient-owned
  trust inputs; no provider credentials needed.
- **Reviewable results:** model and service identities, measured changes,
  uncertainty bounds and the checks that passed or failed.

## Quickstart

Try a signed evidence check on a regular CPU. **Python 3.12+** is required;
no GPU, model download, API key or container engine is needed for this example.

From a checkout of this repository:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install .
python examples/quickstart/run.py \
  --fixture examples/acceptance-handoff/golden
```

The command verifies retained evidence against the example's separate trust
inputs, issues a new signed receipt and writes an HTML report. It prints:

```text
PASS signed evidence verified
Decision: pass
```

Open `invarlock-quickstart-output/evidence.html`. The same directory contains
`verification.receipt.json` and `verification.result.json`. This demonstrates
verification and reporting of a fixed comparison; it makes no new model calls.

<details>
<summary>Use a published wheel instead</summary>

Start in an empty directory and download the examples matching the installed
release:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install invarlock
INVARLOCK_VERSION="$(python -c 'from importlib.metadata import version; print(version("invarlock"))')"

curl -fsSLO \
  "https://github.com/invarlock/invarlock/archive/refs/tags/v${INVARLOCK_VERSION}.tar.gz" &&
tar -xzf "v${INVARLOCK_VERSION}.tar.gz" --strip-components=3 \
  "invarlock-${INVARLOCK_VERSION}/examples/quickstart" \
  "invarlock-${INVARLOCK_VERSION}/examples/acceptance-handoff/golden" &&

python run.py --fixture golden
```

</details>

Use documentation and examples from the same release as a published wheel.
For local builds, use the exact source checkout that built the package; retain
its commit and wheel digest when sharing it. A missing release archive is an
error, never a reason to substitute another version. See
[matching wheels and examples](docs/user-guide/getting-started.md#matching-wheels-and-examples).

To collect new judge ratings, install `python -m pip install "invarlock[judge]"`.
Judge scoring is built into the core; this extra adds the matching collector and
its pinned provider SDKs. Offline import, verification and reporting need only
`invarlock`. See [judge setup](docs/reference/judge-measurements.md#frozen-answer-requests-and-preflight).

## What can you use it for?

| Your task | Starting point |
| --- | --- |
| Check a fine-tune, quantized model or runtime change against a baseline | [Model-change workflows](docs/user-guide/change-scenarios.md) |
| Add a verifiable comparison to an existing evaluator or CI pipeline | [Captured results](docs/user-guide/captured-results.md) |
| Grade frozen answers under a task-specific rubric | [Judge scoring](docs/reference/judge-measurements.md) |
| Recheck a hosted service after a change or on a schedule | [Hosted-service requalification](docs/user-guide/hosted-service-requalification.md) |
| Send evidence to a customer or internal release reviewer | [Evidence and verification](docs/user-guide/evidence-and-verification.md) |

These workflows fit teams that repeatedly need to produce, check or retain
evidence supporting a model change: model suppliers, fine-tuning and optimization
teams, and internal AI teams with a release-review process.

## One workflow: evaluate, verify, report

<p align="center">
  <img
    src="docs/assets/evaluation-verification-flow.svg"
    alt="Native comparisons, captured records and frozen answers feed evaluation; independent verification and reporting use the resulting evidence"
    width="100%"
  />
</p>

For a captured deterministic comparison, prepare a request, signing key and
independent trust profile:

```bash
invarlock evaluate request.yaml --signing-key signing-key.pem --preflight --json
invarlock evaluate request.yaml --signing-key signing-key.pem
invarlock verify evidence/ --trust-profile trust/trust-inputs.json \
  --receipt verification.receipt.json
invarlock report evidence/ --html report.html
```

Use the evidence destination declared in your request. The
[captured-results guide](docs/user-guide/captured-results.md)
provides complete setup instructions, keys and trust-profile preparation.
Native runs also need runtime resources; judge receipt issuance uses its own
verifier-key and identity options. Follow the linked guide for that workflow.

| Command | What it does |
| --- | --- |
| `evaluate` | Runs or imports a declared comparison, applies policy and publishes evidence. Preflight checks setup without execution. |
| `verify` | Checks signatures, bound identities and supported analysis against the recipient's independent expectations; can issue a signed receipt. |
| `report` | Explains the identities, results, uncertainty and policy checks in the retained evidence. Rendering does not replace independent verification. |

The HTML report supports multiple metrics and slices, with an overview and
per-result detail. Terminal and Markdown outputs support review; JSON and JUnit
support automation where the selected workflow provides them. See the
[CLI reference](docs/reference/cli.md)
for output options and policy exit codes.

## Choose a scorer

The same native and captured entry points support three built-in scorers, each
with its own required observations and statistical treatment:

| Scorer | Required observations | Comparison |
| --- | --- | --- |
| Exact match (`exact_match`) | Answers and reference answers for paired cases | Accuracy change with a paired uncertainty interval |
| Normalized NLL (`normalized_nll_per_utf8_byte`) | Reference-continuation log probabilities, UTF-8 byte counts and bound tokenizer metadata | Ratio of mean byte-normalized NLL, with paired resampling |
| Judge (`judge`) | Frozen task text and answers, declared rubric and retained judge calls | Bounded ratings aggregated by declared independent units |

Policy uses the conservative uncertainty bound, with the configured sample,
precision and quality requirements. A point estimate alone does not decide the
result. [Schedule and policy](docs/user-guide/schedule-and-policy.md)
explains the statistical scope and thresholds.

Judge collection is built into core; `invarlock[judge]` installs its pinned
provider SDKs. Collection uses explicit call, token, cost, timeout and checkpoint
limits. Importing retained ratings, verification and reporting work offline in the core wheel. Start with
`invarlock evaluate --init my-judge --example native-judge` and replace the
illustrative model pins with your actual runtime inputs. See
[judge scoring](docs/reference/judge-measurements.md)
for the supported rubric, reference and collection profiles.

Task-specific deterministic extensions cover normalized labels, numeric
tolerances, structured fields and token overlap.
[Evidence sets](docs/reference/evidence-sets.md)
can require multiple complementary checks on the same frozen answers while
preserving each check's meaning.

## Keep your evaluator or run the comparison here

**Use existing records.** Dedicated profiles cover all 19 evaluators in the
maintained shortlist. Export supported native JSON from your existing environment,
then select `adapter: evaluator-native-json` in InvarLock. No InvarLock installation
is needed in the evaluator environment, and recipients need no evaluator SDK or
account. If you prefer a Python helper, `invarlock.engine.export_evaluator_result`
prepares a complete export for `adapter: evaluator-json`. Both routes use the same
`evaluate`, `verify` and `report` commands.

The [capture guide](docs/reference/evaluation-records.md#dedicated-evaluator-exports)
covers source shapes, complete case IDs, custom numeric metrics and metadata
slices. Exact match needs answers and references; normalized NLL needs actual
continuation measurements; judging needs task/answer text and complete judge
measurements. An aggregate score cannot substitute for the required per-case
records. Structured inputs can use an explicit text projection.
The [qualification matrix](docs/reference/evaluator-qualification.md)
distinguishes installed support, replay authority and retained runtime evidence
for each declared profile.

**Run a native comparison.** Hugging Face Transformers is the built-in runtime.
GGUF/llama.cpp, TensorRT-LLM and Hugging Face vision-text providers are included
in core, with optional execution dependencies. Native run mode uses a caller-authorized,
digest-addressed Docker or Podman image. Follow the
[getting-started guide](docs/user-guide/getting-started.md)
for artifact pins, runtime setup and independent verification inputs.
Host-side model preparation uses the matching checkout's
[HF runtime group](docs/user-guide/runtime-providers.md#hugging-face-transformers)
after bootstrapping its verified hardened Accelerate wheel.
The [import request](examples/request.yaml)
uses complete retained provider sidecars and omits `--runtime-image` and `--runtime-image-digest`.

**Recheck a hosted service.** Your harness records fresh executions, service
configuration and observation windows. InvarLock compares those captured facts
and verifies the resulting evidence offline. Hosted identity describes the
observed service; it does not claim access to hidden model weights. Your scheduler
initiates periodic comparisons. Verifying old evidence does not measure the
service again.

## Inspect real retained examples

The repository includes signed evidence, receipts and replay instructions from
actual model runs. Start with the [practical change walkthroughs](examples/README.md#inspect-a-practical-model-change)
for quantization, extraction instructions or a checkpoint replacement. Each
reference establishes its declared workflow and scope:

| Reference | Retained work |
| --- | --- |
| [Native model and runtime comparisons](public_evidence) | Pinned text, GGUF, vision-text and TensorRT-LLM comparisons |
| [Evaluator handoffs](examples/evaluator-qualification/signed-transactions/README.md) | 400-record Qwen3.5 9B Harness and Inspect journeys; a Gemma instruction-to-QAT comparison |
| [Hosted HTTP capture](examples/hosted-service/references/mistral-7b-http/README.md) | 400 paired cases from distinct Mistral 7B base and instruction checkpoints behind a local HTTP service |
| [Likelihood comparison](examples/captured-results/references/mistral-7b-likelihood/README.md) | Distinct Mistral 7B checkpoints on 400 fixed narrative continuations |
| [Bounded judge comparison](examples/judge-measurements/references/k2-32b-luna-xhigh-heldout/README.md) | 10,260 retained ratings across 1,710 QA and extraction cases, with offline replay |

A comparative pass is not proof of adequate task quality or representative
production performance. The local HTTP example does not qualify an external
provider; repeated judge ratings are not additional independent cases.

## What verification establishes

A recipient supplies its expected identities, policy and signer trust through a
channel independent of the submitted bundle. InvarLock checks the package and
reconstructs the supported analysis against those expectations. Native runtime
bindings and imported observations retain different provenance claims.

Verification checks retained evidence; it does not independently rerun the
original model execution. It does not establish that the chosen benchmark covers
production traffic, that a rubric captures every requirement, or that a passing
comparison authorizes deployment. Read the
[assurance case](docs/assurance/assurance-case.md)
and [trust model](docs/security/trust-model.md)
for the precise guarantees and assumptions.

For artifact-delivery automation, optional
[acceptance attestations](docs/reference/acceptance-attestations.md)
and [OPA/Rego or CUE policies](docs/reference/policy-engine-interop.md)
consume the authenticated result under recipient-controlled policy.
v0.13 evidence and receipts remain verifiable and ingestible; acceptance always
uses the recipient's current policy. InvarLock is pre-1.0, with explicit artifact
format versions and a Python API that may evolve between minor releases.

## Documentation and contributing

- [Getting started](docs/user-guide/getting-started.md) · [Examples](examples) · [CLI](docs/reference/cli.md) · [Python API](docs/reference/api-guide.md)
- [Contributing](CONTRIBUTING.md) for development setup and required checks.
- [Discussions](https://github.com/invarlock/invarlock/discussions) for questions and integration ideas; [Issues](https://github.com/invarlock/invarlock/issues) for reproducible bugs.
- [Security policy](SECURITY.md) for private vulnerability reports.

Apache-2.0. See [LICENSE](LICENSE)
and [third-party notices](THIRD_PARTY_NOTICES.md)
for dependency and retained-data terms.
