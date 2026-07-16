# Environment variables

InvarLock keeps environment configuration narrow. Comparison intent belongs in
`request.yaml`. Environment variables are alternatives for command options that
supply secrets, independent verification anchors, OCI launch choices, and
caller-owned provider resources.

!!! info "Reference"

    - **Surface:** Public CLI, OCI launcher, provider-resource, and security
      environment variables
    - **Stability:** Documented variables are public operator inputs;
      repository-test and add-in-internal variables are excluded
    - **Use this page when:** Supplying key paths, independent verifier anchors,
      runtime-image identity, device selection, or caller-owned provider support

An explicit command option wins over its environment alternative. Empty values
do not satisfy required paths, digests, fingerprints, or identities.

## Transaction inputs

| Variable | Command | Meaning |
| --- | --- | --- |
| `INVARLOCK_SIGNING_KEY` | `evaluate` | Ed25519 evidence-signing private-key path |
| `INVARLOCK_POLICY` | `verify` | Independently sourced policy file |
| `INVARLOCK_EXPECTED_BASELINE_ARTIFACT` | `verify` | Approved baseline artifact-identity `sha256:...` digest |
| `INVARLOCK_EXPECTED_SUBJECT_ARTIFACT` | `verify` | Approved subject artifact-identity `sha256:...` digest |
| `INVARLOCK_EXPECTED_SCHEDULE` | `verify` | Approved canonical schedule `sha256:...` digest |
| `INVARLOCK_EXPECTED_BASELINE_RUNTIME` | `verify` | Expected baseline `sha256:...` runtime-image digest |
| `INVARLOCK_EXPECTED_SUBJECT_RUNTIME` | `verify` | Expected subject `sha256:...` runtime-image digest |
| `INVARLOCK_EXPECTED_SIGNER` | `verify` | Expected Ed25519 evidence-signer fingerprint |
| `INVARLOCK_VERIFIER_SIGNING_KEY` | `verify` | Ed25519 verifier private-key path |
| `INVARLOCK_VERIFIER_IDENTITY` | `verify` | Stable verifier identity included in the receipt |

The receipt destination intentionally has no environment alternative. It must
be an explicit new path outside the immutable evidence pack.

Variables contain key paths, not private-key bytes. Keep the evidence-signing
key in a separate, caller-controlled location. Run-mode workers never receive
that key: the host validates their side results and performs evidence signing
and publication itself.

## Host OCI worker launch

When the host CLI receives a valid `execution.mode: run` request, it
automatically launches the transaction in Docker or Podman. These variables
mirror the `evaluate` options:

| Variable | Default | Meaning |
| --- | --- | --- |
| `INVARLOCK_RUNTIME_IMAGE` | None | Local OCI reference; use `repository@sha256:...` or an exact `sha256:...` image identity |
| `INVARLOCK_RUNTIME_IMAGE_DIGEST` | Embedded image digest when present | Pinned lowercase `sha256:...` runtime-image identity |
| `INVARLOCK_BASELINE_RUNTIME_IMAGE` | Common image | Baseline image override |
| `INVARLOCK_BASELINE_RUNTIME_IMAGE_DIGEST` | Baseline embedded or common digest | Baseline image-digest override |
| `INVARLOCK_SUBJECT_RUNTIME_IMAGE` | Common image | Subject image override |
| `INVARLOCK_SUBJECT_RUNTIME_IMAGE_DIGEST` | Subject embedded or common digest | Subject image-digest override |
| `INVARLOCK_CONTAINER_ENGINE` | `docker` | Closed engine selection: `docker` or `podman` |
| `INVARLOCK_RUNTIME_DEVICE` | `cpu` | Shared device: `cpu`, `cuda`, or `cuda:<index>` |
| `INVARLOCK_BASELINE_RUNTIME_DEVICE` | Shared device | Optional baseline override |
| `INVARLOCK_SUBJECT_RUNTIME_DEVICE` | Shared device | Optional subject override |
| `INVARLOCK_RUNTIME_ENTRYPOINT` | `auto` | Shared entrypoint profile: `auto`, `python`, or `nvidia` |
| `INVARLOCK_BASELINE_RUNTIME_ENTRYPOINT` | Shared profile | Optional baseline entrypoint override |
| `INVARLOCK_SUBJECT_RUNTIME_ENTRYPOINT` | Shared profile | Optional subject entrypoint override |

If the image reference contains `@sha256:...`, it must agree with
`INVARLOCK_RUNTIME_IMAGE_DIGEST` when both are supplied. If the image reference
has no embedded digest, the separate digest is required and the launcher forms
the immutable reference. Mutable tags without a pinned digest are rejected.

Each side image must already exist locally. The launcher uses `--pull=never`,
disables networking, sets a read-only container root, drops capabilities,
enables `no-new-privileges`, bounds process count, and provides a temporary
filesystem. The prepared job and schedule, side artifact, and side support
resources are mounted read-only. Each worker receives its own writable output
directory and cannot read the other side's artifact. The host validates the
closed side results before it computes, signs, or publishes evidence.

`cuda` exposes all available GPUs to the selected side worker. An indexed
selection such as `cuda:1` exposes that host GPU as `cuda` inside that worker.
CPU/CPU and distinct explicit CUDA indexes may run in parallel. Generic CUDA,
the same explicit CUDA index, and CPU/CUDA pairs run sequentially. These rules
control worker scheduling; the authenticated schedule and record IDs establish
the measurement pairing.

The `auto` entrypoint selects the NVIDIA entrypoint profile for TensorRT-LLM
and the Python profile for the other first-party providers. Image, digest,
device, and entrypoint defaults are resolved independently for baseline and
subject after applying their per-side overrides.

Image-digest agreement identifies declared image bytes. It is not execution
attestation and does not prove which physical host executed them.

## Canonical Hugging Face runtime images

The repository has distinct CPU and CUDA image definitions. They are not
interchangeable:

| Runtime | Definition | Build | Smoke check |
| --- | --- | --- | --- |
| CPU | `runtime/Dockerfile` | `make runtime-image` | `make runtime-smoke` |
| x86_64 CUDA 12.8 | `runtime/Dockerfile.cuda` | `make runtime-image-cuda` | `make runtime-smoke-cuda` |

`--runtime-device cuda` and the matching environment variables expose a GPU to
the selected worker. They do not install CUDA-enabled PyTorch in an image.
Select an image built from `runtime/Dockerfile.cuda` for CUDA Hugging Face
evaluation, and require `make runtime-smoke-cuda` on the intended NVIDIA host
before relying on that image/device combination.

## Example run environment

```bash
export INVARLOCK_RUNTIME_IMAGE='registry.example/invarlock-runtime-cuda@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
export INVARLOCK_RUNTIME_IMAGE_DIGEST='sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
export INVARLOCK_CONTAINER_ENGINE=docker
export INVARLOCK_RUNTIME_DEVICE=cuda
export INVARLOCK_SIGNING_KEY=keys/evidence-signer.pem

invarlock evaluate release-check/request.yaml
```

Replace the illustrative digest with the exact local CUDA image identity. Keep
the signing-key path in a caller-controlled location; the host reads it only
when signing the validated evidence bundle.

Equivalent explicit options are often clearer for one-off runs:

```bash
invarlock evaluate release-check/request.yaml \
  --signing-key keys/evidence-signer.pem \
  --runtime-image registry.example/invarlock-runtime-cuda@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --runtime-image-digest sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --container-engine docker \
  --runtime-device cuda
```

Import requests do not launch OCI workers and do not require these image,
engine, device, or entrypoint inputs.

## Optional provider resources

The GGUF (`llama_cpp`) add-in requires all three values together:

| Variable | Meaning |
| --- | --- |
| `INVARLOCK_GGUF_RESOURCE_ROOT` | Caller-owned absolute root containing artifact and support files |
| `INVARLOCK_GGUF_BACKEND_EXECUTABLE` | Safe relative `backend_executable` resource beneath that root |
| `INVARLOCK_GGUF_BACKEND_SOURCE` | Safe relative `backend_source` resource beneath that root |

The TensorRT-LLM add-in requires all three values together:

| Variable | Meaning |
| --- | --- |
| `INVARLOCK_TENSORRT_LLM_RESOURCE_ROOT` | Caller-owned absolute root containing engine and support files |
| `INVARLOCK_TENSORRT_LLM_TOKENIZER_CONTRACT` | Safe relative tokenizer-contract resource beneath that root |
| `INVARLOCK_TENSORRT_LLM_RUNNER_EXECUTABLE` | Safe relative runner resource beneath that root |

The Hugging Face vision-text add-in requires both values together:

| Variable | Meaning |
| --- | --- |
| `INVARLOCK_HF_VISION_TEXT_RESOURCE_ROOT` | Caller-owned absolute root containing both artifacts and image content |
| `INVARLOCK_HF_VISION_TEXT_CONTENT_STORE` | Safe relative image-content directory beneath that root |

Resource roots must be absolute non-symbolic directories. Support values must
be portable relative paths beneath the root. Missing members, extra partial
configuration, path escape, and symbolic-link transitions fail closed.

The primary artifact path remains in `request.yaml`. Executable, source, and
tokenizer-support resources remain caller-controlled because a submitted
request is data, not authorization to choose host code.

## Security switches

These variables are false unless set to `1`, `true`, `yes`, or `on`:

| Variable | Effect outside the strict run path |
| --- | --- |
| `INVARLOCK_ALLOW_NETWORK` | Removes the root CLI outbound-socket guard |
| `INVARLOCK_ALLOW_REMOTE_CODE` | Records permission for remote code in runtime provenance |
| `INVARLOCK_ALLOW_THIRD_PARTY_PLUGINS` | Permits discovery of non-first-party runtime entry points |
| `INVARLOCK_ALLOW_INSTALLED_SCORERS` | Equivalent to `--allow-installed-scorers`; authorizes host-side loading of the exact scorer bound by a request and policy |

The strict run path explicitly disables network, remote code, and third-party
runtime discovery inside each side worker. An installed scorer instead runs in
the host evaluation or verification process after provider facts are
authenticated. Authorize it only after reviewing the exact descriptor and
package identity pinned by the request and policy. Independent verification
must make its own authorization decision.

## Validation and failure behavior

| Problem | Transaction behavior |
| --- | --- |
| Missing required command option and environment alternative | Command rejects before completion |
| Key path is missing, symbolic, or not a regular file | Transaction rejects before key content is loaded |
| Signing key is missing, symbolic, or not a regular host file | Transaction rejects before signing |
| Runtime image is mutable, unavailable, or disagrees with its digest | OCI launch rejects |
| Container engine is unavailable or not `docker`/`podman` | OCI launch rejects |
| Invalid or conflicting device selection | OCI launch rejects |
| Partial GGUF, TensorRT-LLM, or vision-text resource group | Runtime construction rejects |
| Support path escapes its root or crosses a link | No-follow resource validation rejects |
| Provider does not support the request metric | Request loading rejects |

## Operational guidance

- Prefer explicit options for one-off invocations and narrowly scoped
  environment variables in automation.
- Keep private-key bytes out of environment variables.
- Keep artifact, schedule, runtime, signer, and policy anchors outside the
  submitted bundle and source them through a separate trusted path.
- Keep dataset mapping, policy, built-in metric or scorer binding, and provider settings in the
  authenticated request, not environment variables.
- Record the exact image digest and engine/device selection with the comparison
  record; do not infer them from a mutable shell profile later.

Variables used only by repository tests, image builds, or add-in runner
internals are not part of this public inventory. Treat this page and
`invarlock COMMAND --help` as the supported operator surface.

## Related documentation

- [Command-line interface](cli.md) lists the matching command options.
- [Evaluation request](../user-guide/evaluation-request.md) separates comparison
  intent from host authority.
- [Runtime providers](runtime-providers.md) defines provider-specific identity
  and resource requirements.
- [Evaluation lifecycle](lifecycle.md) describes validation and publication
  boundaries.
