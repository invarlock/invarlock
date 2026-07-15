# Native Runtime Providers

## Overview

| Aspect | Details |
| --- | --- |
| **Purpose** | Compare authenticated GGUF/llama.cpp and TensorRT-LLM behavior without converting the artifacts back into Hugging Face checkpoints. |
| **Audience** | Model-release and runtime-validation engineers operating native deployment artifacts. |
| **Status** | First-party experimental workflow under `invarlock advanced runtime-behavior`. |
| **Claim** | Directed, policy-scoped `exact_match` behavior on one authenticated record schedule. |
| **Start here** | [Runnable native-provider pair](https://github.com/invarlock/invarlock/tree/main/examples/integrations/runtime_providers). |

Use the native-provider path when the artifact being approved is the deployed
GGUF file or TensorRT-LLM engine itself. Use the normal `invarlock evaluate →
invarlock verify → invarlock report html` path for Hugging Face checkpoint
regression and guard evidence.

## Supported Boundary

| Provider | Current boundary | Required backend |
| --- | --- | --- |
| `llama_cpp` | Linux CPU, one authenticated GGUF file | Qualified pinned llama.cpp OCI image |
| `tensorrt_llm` | Linux NVIDIA GPU, TensorRT-LLM 1.2.1, maintained qualification on compute capability 9.0 | Qualified pinned NVIDIA-derived OCI image |

Connector discovery is metadata-only:

```bash
invarlock advanced plugins runtime-providers
```

`Connector ready; runtime not probed` means the Python connector is importable.
It does not mean the backend, image, GPU, or model artifact has passed local
qualification.

## Operator Sequence

The transaction has two separate gates:

1. **Platform qualification** builds and tests the pinned native image on the
   intended execution platform. It protects stable image selection.
2. **Behavioral evidence** authenticates the real baseline and subject inputs,
   produces one immutable side per role, and independently verifies the pair
   against a directed policy.

A platform canary is not behavioral evidence for a release schedule. Complete
all of these stages for the artifacts under review:

```text
records + dataset identity
  → authenticated schedule
  → baseline binding + subject binding
  → directed policy
  → baseline side + subject side
  → positive pair receipt
```

The maintained runnable wrapper performs that full sequence and supplies the
required container boundary:

```bash
bash examples/integrations/runtime_providers/run_native_pair.sh
```

Its provider containers run as the invoking uid and gid, leaving the strict
`0600` outputs readable by the host-side control commands. Each container has a
read-only root and one persistent writable output bind mount; a bounded
non-executable in-memory `/tmp` filesystem provides runtime scratch. The
wrapper checks the schedule, both bindings, policy, both complete side
directories, and final receipt before reporting success.

See the [example README](https://github.com/invarlock/invarlock/tree/main/examples/integrations/runtime_providers)
for settings derivation, exact environment variables, mixed-provider
operation, and the produced file layout.

## Acceptance Inputs

Keep these inputs separate from a subject producer whenever the organizational
review boundary requires independent approval:

- the record schedule and expected outputs;
- the directed policy thresholds;
- the reviewed runtime-image digest for each role; and
- the decision to accept the resulting pair receipt.

Each provider settings file binds the native artifact, backend, decoding
settings, and relevant tokenizer or build identity. The example derives those
settings inside the reviewed image and refuses to overwrite an existing file.
Do not copy a mutable image tag or an identity claimed by the submitted subject
into reviewer-owned policy.

## Reading Outcomes

`run-side` writes a side only after reloading its report, runtime manifest,
provider receipt, artifact identity, observation, schedule, and directed policy
bindings. `verify-pair` then reloads both sides and writes a positive receipt
only when the pair passes.

All native workflow command failures use exit status `2`. In JSON mode, a
failure returns one versioned error object and does not emit a positive pair
receipt. Preserve failed workspaces for diagnosis, but do not treat a partial
side directory or platform qualification result as accepted evidence.

## Build The Release-Evidence Asset

After both platform qualification summaries and a positive pair receipt have
been reviewed, a maintainer can build the compact release-asset carrier from an
authenticated checkout of the matching source:

```bash
python scripts/release/runtime_release_evidence.py build \
  --source-commit <full-40-hex-source-commit> \
  --source-archive-sha256 <source-archive-64-hex-sha256> \
  --qualification llama_cpp:cpu-reference=artifacts/qualification/gguf-runtime-blackbox-summary.json \
  --qualification tensorrt_llm:pair-a=artifacts/qualification/tensorrt-llm-pair-a-summary.json \
  --qualification tensorrt_llm:pair-b=artifacts/qualification/tensorrt-llm-pair-b-summary.json \
  --behavior artifacts/native-runtime-pair/paired-receipt.json \
  --output artifacts/release/invarlock-runtime-evidence.tar.gz

python scripts/release/runtime_release_evidence.py validate \
  --asset artifacts/release/invarlock-runtime-evidence.tar.gz \
  --expected-source-commit <full-40-hex-source-commit> \
  --expected-source-archive-sha256 <source-archive-64-hex-sha256> \
  --expected-provider llama_cpp \
  --expected-provider tensorrt_llm \
  --expected-qualification llama_cpp:cpu-reference \
  --expected-qualification tensorrt_llm:pair-a \
  --expected-qualification tensorrt_llm:pair-b \
  --require-behavioral-claim
```

Use a lowercase path-free qualification name to distinguish independently
reviewed runs of the same provider. A repeated provider must use a name on
every entry, and the validator can pin the complete provider-and-name set.
Naming does not by itself prove independent execution. Deterministic independent
runs may legitimately produce byte-identical summaries, so the carrier accepts
that result without adding meaningless entropy. Review the private run
provenance before accepting an independence claim. Do not put host names,
addresses, operator identities, or filesystem paths in a name.

The original `PROVIDER=SUMMARY` form remains valid for one qualification per
provider, preserving existing single-run automation and assets. Do not mix the
original and named forms for the same provider in one asset.

Use the resulting `tar.gz` as the release-asset carrier after independent
validation. It is not source-tree evidence and does not contain model files,
native engines, raw logs, host paths, or an additional behavioral claim. The
asset only carries the closed qualification and pair-receipt material supplied
to the builder.

## Scope And Next Steps

The native receipt covers literal typed exact-match behavior on the selected
records. It does not claim:

- weight, tensor, activation, or quantization equivalence;
- log-probability, latency, throughput, or numerical equivalence;
- model quality outside the authenticated schedule;
- correctness of the backend or export process; or
- attestation of a potentially compromised execution host.

For the complete settings fields, image build and qualification targets, and
trust boundary, continue to [Runtime Providers](../reference/runtime-providers.md).
For release-approval ownership and evidence handling, use the
[Strict Assurance Checklist](../assurance/15-strict-assurance-checklist.md).
