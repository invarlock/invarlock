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

<p align="center"><em>Run or import paired release-regression evidence. Verify it independently. Hand off a recipient-controlled decision.</em></p>

<p align="center">
  <a href="https://github.com/invarlock/invarlock/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/invarlock/invarlock/ci.yml?branch=main&label=CI&logo=github&labelColor=18150f" /></a>
  <a href="https://pypi.org/project/invarlock/"><img alt="PyPI" src="https://img.shields.io/pypi/v/invarlock?label=PyPI&logo=pypi&labelColor=18150f&color=1f3a7a" /></a>
  <a href="https://invarlock.github.io/invarlock/"><img alt="Docs" src="https://img.shields.io/badge/docs-quickstart-1f3a7a?labelColor=18150f" /></a>
  <a href="https://github.com/invarlock/invarlock/blob/main/LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-1f3a7a?labelColor=18150f" /></a>
  <a href="https://www.python.org/downloads/release/python-3120/"><img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-1f3a7a?logo=python&logoColor=f4efe3&labelColor=18150f" /></a>
</p>

Using independently supplied trust inputs, an artifact recipient can check
whether one exact model derivative satisfies an agreed release-regression
policy.

InvarLock is an open-source assurance engine for one paired
baseline-versus-subject decision. It can execute both sides on the same
deterministic schedule or import complete authenticated per-record material
from another controlled evaluation. A closed request binds the artifacts,
evaluation source or schedule, runtime and evaluator identities, scoring
contract, and policy. InvarLock recomputes the permitted comparison and
publishes a signed evidence bundle; a separate verifier replays that bundle
against independently supplied trust anchors.

## Evidence paths

Native execution and independently replayable import converge on the same signed evidence,
independent verification, and reporting transaction:

<p align="center">
  <img
    src="https://raw.githubusercontent.com/invarlock/invarlock/main/docs/assets/evaluation-verification-flow.svg"
    alt="A pinned paired evaluation request runs baseline and subject providers, publishes signed evidence, is independently verified, and is rendered as a report"
    width="100%"
  />
</p>

For artifact delivery, the workflow continues from technical verification to a
recipient-controlled decision:

> authenticated evaluation → portable signed evidence → independent technical
> verification → recipient-controlled acceptance

An evaluation operator publishes the evidence; the artifact recipient controls
the acceptance policy and decision.

```bash
invarlock evaluate request.yaml
invarlock verify evidence/
invarlock report evidence/
```

Native execution receives pinned artifacts, evaluation source, runtime,
deterministic scoring, and policy; InvarLock runs both sides and derives the
paired result. External evaluator adapters remain in the example and
integration layer and normalize source exports through evaluator-neutral
contracts; the core exposes evaluator-neutral contracts only. Their status is
recorded on three independent axes:

| Axis | Values | Meaning |
| --- | --- | --- |
| Adapter support | Maintained or external | Whether an adapter and pinned upstream entry point are maintained; this grants no decision authority |
| Replay authority | Independently replayable or observation-only | Whether complete ordered facts can be deterministically recomputed, or can only be preserved as authenticated context |
| Signed-journey maturity | Retained or not yet demonstrated | Whether a model-running signed `evaluate` → `verify` → `report` OCI transaction has been retained |

The stable qualification result expresses replay authority as
`verdict_authority` or `observation_only`. Adapter support and signed-journey
maturity are catalog metadata, not fields that an imported result can claim
for itself.

## Decision boundary

InvarLock answers one precise question: whether a subject artifact satisfies an
agreed release-regression policy relative to a baseline, using authenticated
evidence and independently supplied trust anchors. It makes that decision
reproducible, portable, and suitable for recipient-controlled approval.

The decision remains bounded by the supplied evidence and identities. Broader
deployment, safety, compliance, and organizational decisions remain with their
corresponding controls and reviewers. See the
[threat model](https://github.com/invarlock/invarlock/blob/main/docs/security/threat-model.md#explicit-non-goals)
and [assurance case](https://github.com/invarlock/invarlock/blob/main/docs/assurance/assurance-case.md)
for the complete claim boundary and assumptions.

## Try the signed handoff locally

The five-minute wheel workflow verifies retained signed evidence against
independent anchors, issues a fresh verifier-signed receipt, and renders an HTML
report. It needs only Python 3.12 or newer and a regular CPU.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install "invarlock==0.15.0"

curl -fsSLO \
  https://github.com/invarlock/invarlock/archive/refs/tags/v0.15.0.tar.gz
tar -xzf v0.15.0.tar.gz --strip-components=3 \
  invarlock-0.15.0/examples/quickstart \
  invarlock-0.15.0/examples/acceptance-handoff/golden

python run.py --fixture golden
```

The command prints `Decision: pass` and the paths to the signed receipt,
machine-readable verification result, and human report. The versioned example
files stay outside the package; the command imports only the installed wheel.
The fuller [offline handoff example](https://github.com/invarlock/invarlock/tree/main/examples/acceptance-handoff)
also builds fixture evidence and exercises ten fail-closed recipient scenarios.

## Inspect published evidence

The repository carries strictly verified signed evidence packs across the
built-in text runtime and first-party vision-text and TensorRT-LLM runtimes.
Each uses a pinned public qualification suite and includes an independently
signed verification receipt.

```bash
make public-evidence-audit
invarlock report \
  public_evidence/evidence/mistral-7b-weight-scale-hf/evidence
python -m json.tool \
  public_evidence/evidence/mistral-7b-weight-scale-hf/verification.receipt.json
```

Start with the
[public evidence index](https://github.com/invarlock/invarlock/tree/main/public_evidence)
for the maintained inventory and interpretation limits. Reporting authenticates
and explains the signed bundle. Acceptance remains a separate,
recipient-controlled decision under current policy. Technical verification
obtains the expected artifact, schedule, evaluated policy, runtime, evidence
signer, and verifier anchors through channels independent of the submitted
pack.

## Run, verify, and report

Install the built-in Hugging Face provider for a native text comparison:

```bash
python -m pip install "invarlock[hf]"
```

For the native command below, start from the
[complete run request](https://github.com/invarlock/invarlock/blob/main/docs/user-guide/evaluation-request.md#complete-run-request)
and replace every illustrative digest with one derived from the exact input.
Import mode instead starts from the repository's
[schema-valid import request](https://github.com/invarlock/invarlock/blob/main/examples/request.yaml)
and omits `--runtime-image` and `--runtime-image-digest`; the request binds the
complete imported provider sidecars.

```bash
invarlock evaluate request.yaml \
  --signing-key evidence-signer.pem \
  --runtime-image registry.example/invarlock-runtime@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --runtime-image-digest sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

invarlock verify artifacts/evidence-001/ \
  --trust-profile trust/trust-inputs.json \
  --receipt verification.receipt.json

invarlock report artifacts/evidence-001/ --html evidence.html --explain
```

`evaluate --preflight --json` returns the machine-readable result of complete
execution-free validation. Native run mode delegates to a caller-authorized,
digest-addressed Docker or Podman image. Import mode authenticates and replays
complete provider sidecars and ordered per-record evidence locally.
The [getting-started guide](https://github.com/invarlock/invarlock/blob/main/docs/user-guide/getting-started.md)
and [runtime-provider guide](https://github.com/invarlock/invarlock/blob/main/docs/user-guide/runtime-providers.md)
cover request construction, image preparation, device selection, host-only
signing keys, and independently derived verifier inputs.

## The release-regression decision

Both sides score the same authenticated records in the same order. InvarLock
derives one of two built-in paired comparisons:

| Metric | Point comparison | Policy verdict |
| --- | --- | --- |
| `exact_match` | Subject accuracy minus baseline accuracy, with paired regression and improvement counts | Lower bound of the paired Newcombe 95% interval is at least `delta_min_pp` |
| `normalized_nll_per_utf8_byte` | Ratio of arithmetic means of per-record byte-normalized expected-continuation NLL | Upper bound of the paired schedule-resampling interval is at most `ratio_max` |

The conservative interval bound controls the policy; the point value remains
descriptive. A policy may also require a minimum paired-record count and
maximum interval width. Exact match includes an exact two-sided McNemar test;
normalized NLL uses 2,048 deterministic paired schedule-resampling replicates.
Normalized NLL specifically measures expected-continuation likelihood
regression; broader model-quality claims require other evidence.

For task-specific deterministic scoring, a request may bind one authorized
scorer extension. The extension derives one replayable value per record while
the core retains ownership of pairing, aggregation, intervals, and policy.
See [schedule and policy](https://github.com/invarlock/invarlock/blob/main/docs/user-guide/schedule-and-policy.md)
for the full statistical contract and threshold guidance.

## Import and qualify evaluator results

The core exposes one evaluator-neutral qualification contract as versioned
JSON, the `invarlock-qualify-evaluator` companion CLI, and
`invarlock.engine.qualify_evaluator_export` for Python callers. Open-source or
proprietary evaluators reached through an SDK, CLI, or API normalize into that
same boundary outside the core.

Independent replay requires complete ordered per-record evidence that passes
identity, provenance, schedule, and deterministic recomputation requirements.
Aggregate-only outputs and unsupported judge results remain observation-only.
A signed observation proves what was supplied; a replayable source additionally
requires the identity, schedule, and recomputation guarantees above.

The maintained
[evaluator qualification matrix](https://github.com/invarlock/invarlock/blob/main/docs/reference/evaluator-qualification.md)
groups recognizable upstream evaluators by role and records adapter support,
source version, replay authority, and retained signed-journey maturity. Each
independently replayable import starts with
retained output from a pinned real model evaluation, passes through a
source-shaped adapter, and completes the closed import replay. LM Evaluation
Harness and Inspect AI additionally include retained 400-record native signed
OCI transactions over a shared deterministic corpus. Their adapters and
profiles are example-owned; new profiles extend the same evaluator-neutral
engine contract. The retained
[flagship proof map](https://github.com/invarlock/invarlock/blob/main/examples/evaluator-qualification/signed-transactions/README.md#flagship-proof-map)
links each upstream output, qualified import, signed transaction, and verifier
receipt.

The
[`examples/integrations/`](https://github.com/invarlock/invarlock/tree/main/examples/integrations)
directory contains maintained artifact-producing journeys for Hugging Face,
PEFT, TorchAO, GGUF/llama.cpp, TensorRT-LLM, Hugging Face vision-text, and LM
Evaluation Harness, plus the compact Inspect AI bridge. OpenAI Evals has a
maintained native adapter without retained signed-journey evidence. The
[model-change workflow guide](https://github.com/invarlock/invarlock/blob/main/docs/user-guide/change-scenarios.md)
maps common model and runtime changes to native execution, optional-runtime, or
import boundaries.

## Hand off acceptance

The detailed signed verification receipt remains the replayable technical
result. An optional in-toto/DSSE acceptance attestation transports that result
and its exact subject binding. The artifact recipient still applies separate
envelope and receipt trust, independent envelope and evidence freshness,
contract-version, signer-status, and verdict policy.

Recipients can consume the acceptance envelope with a standalone reference
verifier and maintained OPA/Rego or CUE policy configuration. The example
authenticates the envelope and embedded receipt, then applies current recipient
policy. Its conformance fixtures cover an accepted delivery, policy rejection,
subject tampering, an untrusted signer, stale evidence, and an unsupported
contract.

Acceptance-policy interoperability applies current recipient policy to the
authenticated projection. `invarlock verify` performs complete evidence-pack
replay. See the
[policy-engine interoperability reference](https://github.com/invarlock/invarlock/blob/main/docs/reference/policy-engine-interop.md).

> **Compatibility note:** v0.13 evidence and receipts remain permanently
> verifiable and permanently ingestible as first-class dossier inputs. Every
> acceptance outcome remains controlled by the recipient's current policy.

## Providers and diagnostics

Hugging Face Transformers is the built-in reference provider and supports both
built-in metrics. First-party optional GGUF/llama.cpp, TensorRT-LLM, and Hugging
Face vision-text packages are independently installable runtime integrations
with their own dependency sets. The vision-text add-in supports exact-match
comparisons over authenticated prompt and image parts. See
[runtime providers](https://github.com/invarlock/invarlock/blob/main/docs/user-guide/runtime-providers.md).

Spectral, random-matrix, and variance summaries live in the optional
`invarlock-diagnostics` package. They are observation-only diagnostics. Their
canonical JSON can be attached to the signed bundle and appears in a separate
report section. The selected paired comparison and policy remain the sole
technical-verdict inputs. See
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
  [acceptance attestations](https://github.com/invarlock/invarlock/blob/main/docs/reference/acceptance-attestations.md),
  [compatibility covenant](https://github.com/invarlock/invarlock/blob/main/docs/reference/compatibility.md),
  [evaluator qualification](https://github.com/invarlock/invarlock/blob/main/docs/reference/evaluator-qualification.md),
  [policy-engine interoperability](https://github.com/invarlock/invarlock/blob/main/docs/reference/policy-engine-interop.md),
  [runtime providers](https://github.com/invarlock/invarlock/blob/main/docs/reference/runtime-providers.md), and
  [Python API](https://github.com/invarlock/invarlock/blob/main/docs/reference/api-guide.md).

InvarLock is pre-1.0. Canonical artifact formats carry explicit format versions;
the Python embedding facade may evolve between minor releases.

Questions and design discussions belong in
[GitHub Discussions](https://github.com/invarlock/invarlock/discussions). Report
bugs through [GitHub Issues](https://github.com/invarlock/invarlock/issues) and
security concerns through
[SECURITY.md](https://github.com/invarlock/invarlock/blob/main/SECURITY.md).

If you ship or receive derived model artifacts and want to co-publish a real
evidence-and-receipt handoff using the open engine, start a
[design-partner discussion](https://github.com/invarlock/invarlock/discussions/new?category=ideas).

Apache-2.0 — see the
[license](https://github.com/invarlock/invarlock/blob/main/LICENSE).
