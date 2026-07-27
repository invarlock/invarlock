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

An artifact recipient can independently check whether one exact model
derivative satisfies an agreed release-regression policy without trusting the
system that delivered it.

InvarLock is an open-source assurance engine for one paired
baseline-versus-subject decision. It can execute both sides on the same
deterministic schedule or import complete authenticated per-record material
from another controlled evaluation. A closed request binds the artifacts,
evaluation source or schedule, runtime and evaluator identities, scoring
contract, and policy. InvarLock recomputes the permitted comparison and
publishes a signed evidence bundle; a separate verifier replays that bundle
against independently supplied trust anchors.

## Evidence paths

| Path | What enters InvarLock | Decision authority |
| --- | --- | --- |
| Native execution | Pinned artifacts, evaluation source, runtime, metric or deterministic scorer, and policy | InvarLock runs both sides and derives the paired result |
| Qualified import | Complete ordered per-record results, provenance, identities, schedule, and runtime bindings | InvarLock authenticates the import and recomputes the supported result |
| Authenticated observation | Aggregate-only results, external judges, or other non-replayable context | Preserved as signed context; never allowed to determine the verdict |

External evaluator adapters normalize source exports through the same
versioned JSON, CLI, and Python qualification contracts. They live in the
example and integration layer rather than becoming
evaluator-specific engine plugins.

For artifact delivery, the workflow is:

> authenticated evaluation → portable signed evidence → independent technical
> verification → recipient-controlled acceptance

An evaluation operator publishes the evidence; the artifact recipient controls
the acceptance policy and decision.

```bash
invarlock evaluate request.yaml
invarlock verify evidence/
invarlock report evidence/
```

<p align="center">
  <img
    src="https://raw.githubusercontent.com/invarlock/invarlock/main/docs/assets/evaluation-verification-flow.svg"
    alt="A pinned paired evaluation request runs baseline and subject providers, publishes signed evidence, is independently verified, and is rendered as a report"
    width="100%"
  />
</p>

## Try the signed handoff locally

The service-free acceptance example runs a complete signed evaluation
transaction over fixture artifacts and imported per-record results, independent
technical verification, and recipient-policy handoff. It also demonstrates
fail-closed rejection for changed artifacts, tampered evidence, untrusted or
revoked signers, stale evidence, and contradictory envelope content.

```bash
git clone https://github.com/invarlock/invarlock.git
cd invarlock
python -m pip install -e .
make example-acceptance-handoff
```

The command uses a temporary workspace and needs no model download, GPU, OCI
engine, or running InvarLock service after installation. See the
[offline handoff example](https://github.com/invarlock/invarlock/tree/main/examples/acceptance-handoff).

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
for the maintained inventory and interpretation limits. Rendering authenticates
and explains the signed bundle; it is not an independent acceptance decision.
Actual verification also requires artifact, schedule, policy, runtime, evidence
signer, and verifier anchors obtained through channels independent of the
submitted pack.

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

`evaluate --preflight --json` performs the complete execution-free validation
without starting a runtime or creating evidence. Native run mode delegates to a
caller-authorized, digest-addressed Docker or Podman image. Import mode requires
complete provider sidecars and ordered per-record evidence but no model runtime.
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

The policy reads the conservative interval bound, not the point value alone. It
may also require a minimum paired-record count and maximum interval width.
Exact match includes an exact two-sided McNemar test; normalized NLL uses 2,048
deterministic paired schedule-resampling replicates. Normalized NLL measures
expected-continuation likelihood regression, not general model quality.

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

Complete ordered per-record evidence may receive verdict authority only when
identity, provenance, schedule, and deterministic recomputation requirements
pass. Aggregate-only outputs and unsupported judge results fail closed to
verdict authority and remain observation-only. A signed observation proves
what was supplied; it does not make that source replayable.

The maintained
[evaluator qualification matrix](https://github.com/invarlock/invarlock/blob/main/docs/reference/evaluator-qualification.md)
groups recognizable upstream evaluators by role and records their source
version, evidence granularity, identity and provenance binding, replay status,
and authority boundary. Each authoritative import demonstration starts with
retained output from a pinned real model evaluation, passes through a
source-shaped adapter, and completes the closed import replay. The matrix
separately records model-running signed journeys. These are example-owned
adapters and profiles, not evaluator-specific engine plugins or a permanent
catalog ceiling.

The
[`examples/integrations/`](https://github.com/invarlock/invarlock/tree/main/examples/integrations)
directory contains maintained artifact-producing journeys for Hugging Face,
PEFT, TorchAO, GGUF/llama.cpp, TensorRT-LLM, Hugging Face vision-text, and LM
Evaluation Harness. The
[model-change workflow guide](https://github.com/invarlock/invarlock/blob/main/docs/user-guide/change-scenarios.md)
maps common model and runtime changes to native execution, optional-runtime, or
import boundaries.

## Hand off acceptance

The detailed signed verification receipt remains the replayable technical
result. An optional in-toto/DSSE acceptance attestation transports that result
and its exact subject binding. The artifact recipient still applies separate
envelope and receipt trust, independent envelope and evidence freshness,
contract-version, signer-status, and verdict policy.

Recipients can consume the acceptance envelope without an InvarLock service or
policy-engine plugin. The maintained example authenticates the envelope and
embedded receipt with a standalone reference verifier, then applies current
recipient policy in both OPA/Rego and CUE. Its conformance fixtures cover an
accepted delivery, policy rejection, subject tampering, an untrusted signer,
stale evidence, and an unsupported contract.

This is acceptance-policy interoperability, not complete evidence replay.
Recipients use `invarlock verify` when they need to replay every evidence-pack
invariant. See the
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
`invarlock-diagnostics` package. They are observation-only diagnostics; the
selected paired comparison and policy exclusively determine acceptance. Their
canonical JSON can be attached to the signed bundle and appears in a separate
report section without changing the verdict. See
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
